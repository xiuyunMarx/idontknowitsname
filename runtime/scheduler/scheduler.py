"""Speculation scheduling policy: WHO may prefill WHEN, calibrated by profiler.py.

Owns all speculative-prefill admission state so GuardServer stays thin:

  real traffic   — tracked via real_begin()/real_end(); the in-flight count
                   defines the engine's mode.
  idle mode      — no real request in flight (call gaps, tool execution, think
                   time): speculative prefills run up to `idle_concurrency` at a
                   time; an arriving real call overlaps at most those in-flight
                   chunks, and priority scheduling lets it jump every queued one.
  busy mode      — a real request is prefilling/decoding: speculation is capped
                   at `busy_budget`, the CALIBRATED number of concurrent warms
                   this device+model tolerates within the TBT slack (measured 0
                   for Qwen2.5-3B on RTX 3090, 1 for 0.5B). Budget 0 = park until
                   the next idle window — which is exactly the tool-execution /
                   inter-call gap of real agent workflows.

Calibration reports are cached per (device, model, slack) so repeated server
boots skip the ~35 s sweep.
"""

import asyncio
import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from vllm.v1.engine.async_llm import AsyncLLM

from .profiler import ScheduleProfiler


class Scheduler:
    def __init__(self, engine: Optional[AsyncLLM] = None, idle_concurrency: int = 2, idle_grace: float = 0.03):
        self.engine = engine
        self.profiler = ScheduleProfiler(engine) if engine is not None else None
        self.profile_report: Optional[Dict[str, Any]] = None
        self.busy_budget = 0  # concurrent speculative prefills allowed under real traffic (safe default)
        self.idle_concurrency = idle_concurrency
        # Hysteresis: an idle window admits speculation only after persisting for `idle_grace` seconds.
        self.idle_grace = idle_grace
        self._idle_since = 0.0  # 0.0 => "idle since forever" (boot state)
        self._real = 0  # in-flight real requests
        self._spec = 0  # in-flight speculative prefills
        self._cond = asyncio.Condition()
        # Preemption: when a real request arrives while more speculation is in
        # flight than the busy budget allows, the current era's event fires and
        # those holders must cancel their engine request and re-enter the gate
        # (partial prefix stays cached, so the retry is cheap).
        self._preempt = asyncio.Event()
        self.stats: Dict[str, Any] = {"idle_slots": 0, "busy_slots": 0, "parked": 0, "park_time_s": 0.0, "preempt_signals": 0}

    # -------------------------------------------------------------- calibration
    def _cache_path(self, cache_dir: str, tbt_slack: float) -> Path:
        model = getattr(getattr(self.engine, "model_config", None), "model", "unknown")
        dev = self.profiler.device_prop.name if self.profiler else "cpu"
        key = hashlib.sha1(f"{dev}|{model}|{tbt_slack}".encode()).hexdigest()[:12]
        return Path(cache_dir).expanduser() / f"tbt-profile-{key}.json"

    async def calibrate(self, tbt_slack: float = 0.10, cache_dir: str = "~/.cache/proactive_prefill", **kw: Any) -> Dict[str, Any]:
        """Run (or load) the engine-based interference sweep and set busy_budget."""
        assert self.profiler is not None, "calibration needs the engine"
        path = self._cache_path(cache_dir, tbt_slack)
        if path.exists():
            report: Dict[str, Any] = json.loads(path.read_text())
            print(f"[scheduler] loaded calibration cache {path.name}: busy budget {report['safe_concurrent_prefills']}")
        else:
            report = await self.profiler._profile(tbt_slack=tbt_slack, **kw)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1))
        self.profile_report = report
        self.busy_budget = int(report["safe_concurrent_prefills"])
        return report

    # ------------------------------------------------------------- real traffic
    async def real_begin(self) -> None:
        async with self._cond:
            self._real += 1
            if self._spec > self.busy_budget:
                ev, self._preempt = self._preempt, asyncio.Event()
                ev.set()  # yield: in-flight speculation beyond budget must abort
                self.stats["preempt_signals"] += 1

    @contextlib.asynccontextmanager
    async def real(self) -> AsyncIterator[None]:
        """Wrap one real request; its presence defines busy mode."""
        await self.real_begin()
        try:
            yield
        finally:
            await self.real_end()

    async def real_end(self) -> None:
        async with self._cond:
            self._real -= 1
            if self._real <= 0:
                self._real = 0
                self._idle_since = time.perf_counter()
                asyncio.create_task(self._grace_wake(self._idle_since))

    async def _grace_wake(self, stamp: float) -> None:
        """Wake parked speculation once the idle window has outlived the grace
        period (and only if it is still the same window)."""
        await asyncio.sleep(self.idle_grace)
        async with self._cond:
            if self._real == 0 and self._idle_since == stamp:
                self._cond.notify_all()

    # -------------------------------------------------------------- speculation
    def _cap(self) -> int:
        if self._real == 0 and (time.perf_counter() - self._idle_since) >= self.idle_grace:
            return self.idle_concurrency
        return self.busy_budget

    @contextlib.asynccontextmanager
    async def spec_slot(self) -> AsyncIterator[asyncio.Event]:
        """Admission gate for one speculative prefill. Parks while the engine is
        busy beyond the calibrated budget; woken when an idle window opens.
        Yields the era's preempt event: the holder must race its engine request
        against it and abort+retry when it fires."""
        t0 = time.perf_counter()
        async with self._cond:
            parked = self._spec >= self._cap()
            if parked:
                self.stats["parked"] += 1
            await self._cond.wait_for(lambda: self._spec < self._cap())
            self._spec += 1
            self.stats["idle_slots" if self._real == 0 else "busy_slots"] += 1
            ev = self._preempt
        self.stats["park_time_s"] += time.perf_counter() - t0
        try:
            yield ev
        finally:
            async with self._cond:
                self._spec -= 1
                self._cond.notify_all()


scheduler = Scheduler  # lowercase alias
