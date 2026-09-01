"""Deadline-driven KV planner: Belady with repair.

Every predicted future call becomes a Job carrying a deadline (its expected
arrival) and its work split into creation (uncached tokens) and promotion (host
tokens). Jobs run just-in-time — earliest deadline first, released at their
latest start or whenever the engine goes idle — one at a time inside the
engine's speculative-token budget, in stride-aligned chunks so a real admission
preempts cleanly and a partially-resolved prompt extends incrementally.

Under memory pressure the planner also warm-touches the highest-value cached
prefixes about to fall off the device LRU (a prefill of a resident prefix costs
~zero uncached tokens and refreshes the radix cache's LRU position), steering
the engine's eviction toward blocks that are cheap to repair before their next
use. An eviction the planner can repair in time is free; the objective is to
minimize Σ p(call) × exposed prefill at its arrival.

The engine is duck-typed (serving / spec_allowance / prefill / probe / cost /
ledger / device_profile / promote / _stride); the module imports no engine code.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Job:
    key: str                    # dedup identity: f"{sid}|{kind}|{site}"
    sid: str                    # session id
    epoch: int                  # session epoch at submission; a bump voids the job
    site: str                   # target callsite key
    kind: str                   # "create" | "promote" | "probe" | "touch"
    toks: List[int]             # full target token prefix
    prompt: Optional[str]       # probe jobs carry the rendered text
    p: float                    # probability the call happens
    value: float                # p × exposed-prefill tokens saved if this job lands
    deadline: float             # monotonic: expected arrival, safety-tightened
    t90: float                  # monotonic: pessimistic arrival, for staleness GC
    work: int                   # uncached tokens at submission
    host: int                   # promotable host-tier tokens at submission
    done_upto: int = 0          # chunk progress: token index already prefilled
    chunk_cap: int = 0          # per-pick chunk limit (one-shot lane); 0 = full CHUNK
    attempts: int = 0           # aborted executions since the last progress
    state: str = "queued"       # queued | running | done | void
    on_result: Optional[Callable[[Any, "Job"], None]] = None  # probe follow-up hook


class KVPlanner:
    TICK_S = 0.05               # planner heartbeat between wake events
    SAFETY_K = 0.5              # deadline tightening per unit of arrival spread
    CHUNK = 512                 # prefill chunk, rounded to the engine stride
    MAX_ATTEMPTS = 3            # aborted executions before a job is dropped
    STALE_SLACK_S = 5.0         # past t90 by this much: the call never came
    STEER_AT = 0.85             # device-fill fraction that arms eviction steering
    STEER_TAIL = 0.25           # LRU tail fraction considered endangered
    MAX_TOUCH_PER_TICK = 4
    BACKOFF_ABORTS = 3          # consecutive kills that mean the engine has no headroom
    RELEASE_SLACK_S = 0.5       # release ahead of the latest start

    def __init__(self, engine, manage_only: bool = False) -> None:
        self.engine = engine
        self.manage_only = manage_only  # only promotion/steering/probe: no bulk creation
        self._jobs: Dict[str, Dict[str, Job]] = {}   # sid -> job key -> job
        self._wake = asyncio.Event()
        self._last_over = 0   # ledger over-drift watermark: rising = the engine is evicting
        self._aborts_row = 0  # congestion signal: every recent execution was killed

    # ------------------------------------------------------------------ intake
    def submit(self, sid: str, epoch: int, jobs: List[Job], extend: bool = False) -> None:
        """Adopt a session's fresh job set. Replace semantics: a previously queued
        job not re-submitted is void — its branch collapsed. A re-submitted key with
        longer toks keeps its chunk progress (progressive prefill) and refreshed
        timing; `extend` merges instead of replacing (probe follow-ups add jobs)."""
        held = self._jobs.setdefault(sid, {})
        for j in jobs:
            prev = held.get(j.key)
            if prev is not None and prev.state in ("queued", "running") and prev.epoch == j.epoch:
                if len(j.toks) > len(prev.toks):
                    prev.toks, prev.prompt = j.toks, j.prompt
                    prev.attempts = 0
                prev.p, prev.value = j.p, j.value
                prev.deadline, prev.t90 = j.deadline, j.t90
                prev.work, prev.host = j.work, j.host
                prev.on_result = j.on_result or prev.on_result
                continue
            if prev is not None and prev.state == "running":
                prev.state = "void"      # stale epoch still executing
            held[j.key] = j
        if not extend:
            keep = {j.key for j in jobs}
            for k, prev in held.items():
                if k not in keep and prev.state in ("queued", "running"):
                    prev.state = "void"

    def note_arrival(self, sid: str, site: str, t_arrive: float) -> None:
        """Ground-truth feedback: a call just arrived. Log how its plan stood —
        deadline error (positive = the call came after our deadline, the healthy
        direction) and prefill progress — so a run audits prediction validity
        end to end. `done` jobs are kept in the table for exactly this readout."""
        js = [j for j in self._jobs.get(sid, {}).values()
              if j.site == site and j.kind != "touch"]
        if not js:
            print(f"[timing] {sid} unplanned site={site[:48]}", flush=True)
            return
        # a loop revisits the same site: only the newest epoch's jobs describe
        # THIS arrival — an older iteration's done job would report stale numbers
        newest = max(j.epoch for j in js)
        j = max((j for j in js if j.epoch == newest), key=lambda x: x.done_upto)
        print(f"[timing] {sid} err_s={t_arrive - j.deadline:+.2f} "
              f"done={j.done_upto}/{len(j.toks)} kind={j.kind} state={j.state} "
              f"site={site[:48]}", flush=True)

    def void_session(self, sid: str, keep_epoch: int) -> None:
        """The session advanced: everything planned before `keep_epoch` is stale."""
        for j in self._jobs.get(sid, {}).values():
            if j.epoch < keep_epoch and j.state in ("queued", "running"):
                j.state = "void"

    def drop_session(self, sid: str) -> None:
        for j in self._jobs.pop(sid, {}).values():
            j.state = "void"

    def wake(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------------ the loop
    async def run(self) -> None:
        """Single-flight executor: the only admission path into the speculative
        budget, so budget accounting cannot race."""
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.TICK_S)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            now = time.monotonic()
            self._gc(now)
            self.steer(now)
            job = self._pick(now)
            if job is not None:
                await self._execute(job)

    def _latest_start(self, job: Job, now: float) -> float:
        """Deadline minus the remaining work, each tier at its own profiled rate:
        creation at the loaded prefill rate (the job runs beside real decodes, so
        the idle rate would release it too late), promotion at the host-load rate,
        device residency free."""
        prof = self.engine.device_profile
        unc, host = self.engine.cost(job.toks)
        return job.deadline - unc / prof.prefill_tps_loaded - host / prof.promote_tps

    def _released(self, job: Job, now: float) -> bool:
        """Release in idle window or before the deadline"""
        return (not self.engine.serving
                or now >= self._latest_start(job, now) - self.RELEASE_SLACK_S)

    def _pick(self, now: float) -> Optional[Job]:
        """EDF over the released AND affordable jobs — an unaffordable early deadline
        must not starve promotion work behind it. A job whose cached (device+host)
        frontier extends past its progress is always affordable: promoting or touching
        it costs no uncached compute, which is what the budget is denominated in.
        Uncached compute is additionally timing-gated: a chunk starts only when it can
        finish before any other session's predicted next arrival (the earliest queued
        deadline is that proxy) — a chunk that cannot is guaranteed to be killed by
        the very request it delays, and the abort lands on that request's TTFT."""

        congested = self._aborts_row >= self.BACKOFF_ABORTS and self.engine.serving
        best: Optional[Job] = None
        allowance = self.engine.spec_allowance()
        oneshot = getattr(self.engine, "oneshot_allowance", self.engine.spec_allowance)
        rate = self.engine.device_profile.prefill_tps_loaded
        arrival: Dict[str, float] = {}   # sid -> earliest predicted next call
        for sid, held in self._jobs.items():
            ds = [j.deadline for j in held.values()
                  if j.state == "queued" and j.kind != "touch"]
            if ds:
                arrival[sid] = min(ds)
        for sid, held in self._jobs.items():
            horizon = min((d for s, d in arrival.items() if s != sid), default=float("inf"))
            for j in held.values():
                if j.state != "queued" or not self._released(j, now):
                    continue
                unc = self.engine.cost(j.toks)[0]
                slotless = len(j.toks) - unc > j.done_upto   # promotion/touch: an RPC, no request
                if congested and j.kind != "probe" and not slotless:
                    continue
                if best is not None and (j.deadline, -j.value) >= (best.deadline, -best.value):
                    continue
                if j.kind == "probe":
                    if unc > 0 and (oneshot() < unc or now + unc / rate > horizon):
                        continue
                elif len(j.toks) - unc <= j.done_upto:  # next chunk is uncached compute
                    need = min(self._chunk(), max(1, unc))
                    # a queued probe on the same site makes this its ammunition:
                    # small enabling chunks may ride the one-shot lane, shrinking
                    # the probe's uncached suffix below its own admission cap
                    enabling = any(p.kind == "probe" and p.state == "queued"
                                   and p.site == j.site for p in held.values())
                    if self.manage_only and not enabling:
                        continue
                    j.chunk_cap = 0
                    if allowance < need:
                        stride = getattr(self.engine, "_stride", 1)
                        cap = (oneshot() if enabling else 0) // stride * stride
                        if cap <= 0 or now + cap / rate > horizon:
                            continue
                        j.chunk_cap = cap
                    elif now + need / rate > horizon:
                        continue
                best = j
        return best

    def _chunk(self) -> int:
        stride = getattr(self.engine, "_stride", 1)
        return max(stride, self.CHUNK // stride * stride)

    async def _execute(self, job: Job) -> None:
        if job.state != "queued":    # voided between pick and start
            return
        job.state = "running"
        rid = f"plan-{job.kind}-{uuid.uuid4().hex[:8]}"
        if job.kind == "probe":
            res = await self.engine.probe(job.prompt, rid)
            if job.state == "void":  # epoch bumped mid-flight: landed KV is harmless
                return
            if res is None:
                print(f"[probe-miss] {job.site[:48]} attempt={job.attempts + 1}", flush=True)
                self._retry(job)
                return
            self._aborts_row = 0
            job.state = "done"
            if job.on_result is not None:
                job.on_result(res, job)
            return
        unc = self.engine.cost(job.toks)[0]
        cached_end = len(job.toks) - unc
        if cached_end > job.done_upto:
            end = cached_end   # the whole cached frontier at once: no request, no slot
            ok = await self.engine.promote(job.toks[:end], rid)
        else:
            end = min(len(job.toks), job.done_upto + (job.chunk_cap or self._chunk()))
            ok = await self.engine.prefill(job.toks[:end], rid)
        if job.state == "void":
            return
        if not ok:
            self._retry(job)
            return
        self._aborts_row = 0
        job.done_upto = end
        job.attempts = 0
        if end >= len(job.toks):
            job.state = "done"
        else:
            job.state = "queued"
            self.wake()              # next chunk without waiting a tick

    def _retry(self, job: Job) -> None:
        self._aborts_row += 1
        job.attempts += 1
        job.state = "void" if job.attempts >= self.MAX_ATTEMPTS else "queued"

    def _gc(self, now: float) -> None:
        # done jobs stay until replaced or the session drops: note_arrival reads
        # them back when the predicted call lands
        for sid, held in list(self._jobs.items()):
            for k, j in list(held.items()):
                if j.state == "void":
                    del held[k]
                elif j.state == "queued" and now > j.t90 + self.STALE_SLACK_S:
                    j.state = "void"  # the predicted call never came
            if not held:
                del self._jobs[sid]

    # ------------------------------------------------------------------ eviction steering
    def steer(self, now: float) -> None:
        """D2: under memory pressure, warm-touch the most valuable cached prefixes
        sitting in the device LRU tail. score = p × resident / time-until-use — the
        blocks whose eviction would be expensive to repair before their deadline."""
        ledger = getattr(self.engine, "ledger", None)
        if ledger is None:
            return
        # The shadow under-counts device pressure (decode KV shares the real pool), so
        # arm on either signal: shadow near capacity, or calibration just observed the
        # engine demoting blocks we believed device-resident.
        over = ledger.drift["over"]
        pressured = (ledger.device_used_tokens() >= self.STEER_AT * ledger.device_cap_tokens()
                     or over > self._last_over)
        self._last_over = over
        if not pressured:
            return
        endangered = ledger.device_tail(self.STEER_TAIL)
        if not endangered:
            return
        cands = []
        for held in self._jobs.values():
            for j in held.values():
                if j.state != "queued" or j.kind == "touch":
                    continue
                n = ledger.resident_prefix(j.toks)
                if n == 0 or not any(h in endangered for h in ledger.hashes(j.toks[:n])):
                    continue
                cands.append((j.p * n / max(j.deadline - now, 1.0), j, n))
        cands.sort(key=lambda c: -c[0])
        for score, j, n in cands[:self.MAX_TOUCH_PER_TICK]:
            key = f"{j.sid}|touch|{j.site}"
            held = self._jobs.setdefault(j.sid, {})
            if key in held and held[key].state in ("queued", "running"):
                continue
            held[key] = Job(key=key, sid=j.sid, epoch=j.epoch, site=j.site, kind="touch",
                            toks=j.toks[:n], prompt=None, p=j.p, value=score,
                            deadline=now, t90=now + self.STALE_SLACK_S, work=0, host=0)
            print(f"[steer] {j.site} resident={n} score={score:.1f}", flush=True)
