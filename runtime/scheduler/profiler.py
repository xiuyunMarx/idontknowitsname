"""Engine-based calibration of speculative-prefill interference.

Measures, ON THE SERVING ENGINE ITSELF, how much concurrent speculative prefill
an in-flight decode tolerates before its TBT (time-between-tokens) degrades past
a slack threshold. Everything the real system does — chunked-prefill mixing,
priority scheduling, attention kernels, eager mode — is in the measurement path,
because the measurement IS the engine.

Standalone:  python3 -m runtime.scheduler.profiler --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import asyncio
import random
import time
import uuid
from typing import Any, Dict, Optional, Sequence

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


class ScheduleProfiler:
    def __init__(self, engine: AsyncLLM):
        self.engine = engine
        self.device_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.safe_concurrent_prefills: Optional[int] = None
        self.safe_prefill_tokens: Optional[int] = None
        self.profile_report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ probes
    @staticmethod
    def _random_prompt(n_words: int) -> str:
        return " ".join(f"{random.randrange(16 ** 4):04x}" for _ in range(n_words))

    async def _decode_probe_tbt(self, tag: str, decode_tokens: int = 128) -> float:
        """Median steady-state inter-token gap (seconds) of one decode stream."""
        sp = SamplingParams(max_tokens=decode_tokens, ignore_eos=True, temperature=0.0)
        gaps = []
        last: Optional[float] = None
        async for _ in self.engine.generate(self._random_prompt(24), sp, f"probe-{tag}-{uuid.uuid4().hex[:6]}"):
            now = time.perf_counter()
            if last is not None:
                gaps.append(now - last)
            last = now
        gaps = gaps[8:]  # drop ramp-up steps
        gaps.sort()
        return gaps[len(gaps) // 2] if gaps else float("inf")

    def _injector(self, stop: asyncio.Event, idx: int, prompt_words: int, tag: str):
        """One background loop keeping a speculative-warm-shaped prefill in flight
        (priority=1, unique random prompt so prefix caching cannot swallow it,
        max_tokens=1 — exactly the shape of our invariant warms)."""
        async def run() -> None:
            sp = SamplingParams(max_tokens=1, temperature=0.0)
            while not stop.is_set():
                rid = f"inj-{tag}-{idx}-{uuid.uuid4().hex[:6]}"
                async for _ in self.engine.generate(self._random_prompt(prompt_words), sp, rid, priority=1):
                    pass
        return asyncio.create_task(run())

    # ----------------------------------------------------------------- profile
    async def _profile(self, tbt_slack: float = 0.10,
                       levels: Sequence[int] = (0, 1, 2, 4, 8),
                       prefill_words: int = 512,
                       decode_tokens: int = 128,
                       verbose: bool = True) -> Dict[str, Any]:
        """Sweep concurrent-injector levels on the live engine and derive the safe
        speculation budget. Run once at startup, before real traffic — the sweep
        owns the engine while it runs (~30 s). `prefill_words` ~512 hex words is
        roughly a 1k-token warm, the shape of our invariant prefixes."""
        assert self.engine is not None, "engine-based profiling needs the engine"
        curve: Dict[int, float] = {}
        for n in levels:
            stop = asyncio.Event()
            tasks = [self._injector(stop, i, prefill_words, f"l{n}") for i in range(n)]
            if n:
                await asyncio.sleep(0.5)  # let injectors reach steady state
            curve[n] = await self._decode_probe_tbt(f"l{n}", decode_tokens)
            stop.set()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0.3)  # drain in-flight injections
        base = curve[levels[0]]
        safe = max((n for n in curve if curve[n] <= base * (1.0 + tbt_slack)), default=0)
        # nominal tokens per hex-word prompt: ~2 tokens/word (word + space pieces)
        est_tokens_per_injector = prefill_words * 2
        self.safe_concurrent_prefills = safe
        self.safe_prefill_tokens = safe * est_tokens_per_injector
        self.profile_report = {
            "device": self.device_prop.name,
            "model": getattr(getattr(self.engine, "model_config", None), "model", "?"),
            "tbt_curve_ms": {n: round(t * 1e3, 2) for n, t in curve.items()},
            "baseline_tbt_ms": round(base * 1e3, 2),
            "tbt_slack": tbt_slack,
            "injector_nominal_tokens": est_tokens_per_injector,
            "safe_concurrent_prefills": safe,
            "safe_prefill_tokens": self.safe_prefill_tokens,
        }
        if verbose:
            r = self.profile_report
            print(f"[profiler] {r['device']} · {r['model']}: baseline TBT {r['baseline_tbt_ms']}ms; curve {r['tbt_curve_ms']}")
            print(f"[profiler] safe concurrent speculative prefills: {safe} (~{self.safe_prefill_tokens} tok) at {tbt_slack:.0%} slack")
        return self.profile_report

    def profile_sync(self, **kw: Any) -> Dict[str, Any]:
        """Standalone entry point when no event loop is running."""
        return asyncio.run(self._profile(**kw))


def main() -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    ap = argparse.ArgumentParser(description="Calibrate speculative-prefill interference on a live engine")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tbt-slack", type=float, default=0.10)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    args = ap.parse_args()

    async def run() -> None:
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, enable_prefix_caching=True, enable_chunked_prefill=True,
            max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
            enforce_eager=True, scheduling_policy="priority"))
        try:
            await ScheduleProfiler(engine)._profile(tbt_slack=args.tbt_slack)
        finally:
            engine.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
