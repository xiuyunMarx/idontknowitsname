import asyncio
import json
import random
import statistics
import time
import uuid
from typing import Dict, List, Tuple

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logprobs import Logprob
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from console_helper.debug_output import console_debug, console_log, console_warn, console_error

class ModelEngine:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        self.model_name = model_name
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_prefix_caching=True,
                scheduling_policy="priority",
            )
        )
        self.sp = SamplingParams(temperature=0.7)
        # decode concurrency -> uncached speculative tokens one engine step may carry
        # without pushing TBT past the profiled slack. Interference is a per-step
        # quantity: a speculative request of c uncached tokens lands whole in one
        # step (c << max_num_batched_tokens), so bounding c bounds that step.
        self.spec_tokens_per_step: Dict[int, int] = {}
        self.profile_curve: List[Dict[str, float]] = []  # raw (b, n) -> TBT sweep, kept for plots
        self._decode_tasks = 0
        self._real_prefills = 0  # real requests still before their first token
        self._spec_inflight = False  # at most one speculative request at a time

    def tokenize(self, prompt: str) -> List[int]:
        return self.engine.get_tokenizer().encode(prompt)  # type: ignore[union-attr]

    def count_tokens(self, prompt: str) -> int:
        return len(self.tokenize(prompt))

    def spec_allowance(self) -> int:
        """Uncached tokens the next speculative request may carry right now.
        Idle engine: unbounded 
        A real prefill in flight already owns the step: nothing.
        Otherwise the decode-only batch has compute slack up to the profiled per-step figure for its concurrency."""
        if self._spec_inflight:
            return 0
        if not self.engine.output_processor.has_unfinished_requests():
            return 1 << 30
        if self._real_prefills > 0:
            return 0
        return self.spec_tokens_per_step.get(self._decode_tasks, 0)

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"model": self.model_name, "unit": "tokens_per_step",
                       "spec_tokens_per_step": self.spec_tokens_per_step,
                       "curve": self.profile_curve}, f, indent=1)

    def load_profile(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        # Older profiles budgeted in-flight tokens; that unit does not bound a step.
        if data.get("model") != self.model_name or data.get("unit") != "tokens_per_step":
            return False
        self.spec_tokens_per_step = {int(k): int(v) for k, v in data["spec_tokens_per_step"].items()}
        self.profile_curve = data.get("curve", [])
        return True

    def render(self, messages: List[Dict[str, str]]) -> str:
        return self.engine.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #type: ignore

    async def generate(
        self,
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        text = ""
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        out_tokens = 0
        self._real_prefills += 1  # A real call prefills until its first token, then decodes.
        try:
            async for output in self.engine.generate(
                prompt, sampling_params or self.sp, request_id
            ):
                if output.outputs:
                    if first_token:
                        self._real_prefills -= 1
                        self._decode_tasks += 1
                        first_token = False
                        first_token_at = time.perf_counter()
                        console_debug(f"[serve] {request_id} duration_ms={(first_token_at - started) * 1000:.2f} cached_tokens={output.num_cached_tokens or 0} prompt_tokens={len(output.prompt_token_ids or [])}")
                    text = output.outputs[0].text
                    out_tokens = len(output.outputs[0].token_ids)
        finally:
            # Exactly one decrement per increment, whichever phase we ended in.
            if first_token:
                self._real_prefills -= 1
            else:
                self._decode_tasks -= 1
                console_debug(f"[decode] {request_id} decode_ms={(time.perf_counter() - first_token_at) * 1000:.2f} output_tokens={out_tokens}")
        return text

    async def prefill(self, prefill_prompt: str | List[int], request_id: str, cost: int | None = None) -> None:
        """`cost` is the caller's estimate of uncached tokens; defaults to full length.
        A token-id list warms exactly that prefix of a longer prompt's tokens —
        how chunked speculation bounds each request to a small uncached slice."""
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        started = time.perf_counter()
        if isinstance(prefill_prompt, list):
            cost = len(prefill_prompt) if cost is None else cost
            prefill_prompt = TokensPrompt(prompt_token_ids=prefill_prompt)
        else:
            cost = self.count_tokens(prefill_prompt) if cost is None else cost
        self._spec_inflight = True
        try:
            async for _ in self.engine.generate(
                prefill_prompt, sampling_params, request_id, priority=1
            ):
                pass
        finally:
            self._spec_inflight = False
        print(
            f"[prefill] {request_id} cost_tokens={cost} duration_ms={(time.perf_counter() - started) * 1000:.2f}",
            flush=True,
        )

    async def probe(self, prompt: str, request_id: str, cost: int = 64) -> Dict[int, Logprob]:
        """Greedy one-token probe; returns the top logprobs at the first position.
        The prompt rides the router's own just-computed prefix, so its uncached
        cost is a few think-block tokens."""
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        self._spec_inflight = True
        try:
            async for output in self.engine.generate(prompt, sampling_params, request_id, priority=1):
                if output.outputs and output.outputs[0].logprobs:
                    return output.outputs[0].logprobs[0]
        finally:
            self._spec_inflight = False
        return {}

    def _random_ids(self, num_tokens: int) -> List[int]:
        """Exactly `num_tokens` ids no earlier request has seen — an uncached prompt
        of known length (a random string tokenizes to an unpredictable count)."""
        vocab = len(self.engine.get_tokenizer())  # type: ignore[arg-type]
        return [random.randrange(1000, vocab - 1000) for _ in range(num_tokens)]

    async def _measure_tbt(
        self,
        num_decode_tasks: int,
        step_tokens: int,
        *,
        decode_tokens: int,
        warmup_tokens: int,
    ) -> Dict[str, float]:
        """TBT of `num_decode_tasks` decode streams while one speculative request of
        `step_tokens` uncached tokens at a time is injected back-to-back — the
        production pattern (`prefill` is awaited one at a time, priority 1).
        Returns pooled mean and p95 gap in ms, and the injector's duty (requests
        completed per decode step) so the curve shows how many steps were hit."""
        if num_decode_tasks < 1 or step_tokens < 0:
            raise ValueError("num_decode_tasks must be >= 1 and step_tokens >= 0")

        stop = asyncio.Event()
        decode_started = asyncio.Event()
        started_count = 0
        injected = 0

        async def decode_worker(index: int) -> List[float]:
            nonlocal started_count
            sampling = SamplingParams(max_tokens=decode_tokens, temperature=0.0, ignore_eos=True)
            timestamps: List[float] = []
            async for _ in self.engine.generate(
                TokensPrompt(prompt_token_ids=self._random_ids(24)), sampling,
                f"profile-decode-{index}-{uuid.uuid4().hex}",
            ):
                timestamps.append(time.perf_counter())
                if len(timestamps) == 1:
                    started_count += 1
                    if started_count == num_decode_tasks:
                        decode_started.set()
            return [timestamps[i] - timestamps[i - 1] for i in range(warmup_tokens + 1, len(timestamps))]

        async def injector() -> None:
            nonlocal injected
            await decode_started.wait()
            sampling = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            while not stop.is_set():
                async for _ in self.engine.generate(
                    TokensPrompt(prompt_token_ids=self._random_ids(step_tokens)), sampling,
                    f"profile-spec-{uuid.uuid4().hex}", priority=1,
                ):
                    pass
                injected += 1

        decoders = [asyncio.create_task(decode_worker(i)) for i in range(num_decode_tasks)]
        inj = asyncio.create_task(injector()) if step_tokens > 0 else None
        try:
            gaps_by_stream = await asyncio.gather(*decoders)
        finally:
            stop.set()
            decode_started.set()  # release the injector if a decoder failed at startup
            if inj is not None:
                await asyncio.gather(inj, return_exceptions=True)

        gaps = sorted(g for stream in gaps_by_stream for g in stream)
        if len(gaps) < num_decode_tasks * 4:
            raise RuntimeError("not enough generated tokens to measure TBT; increase decode_tokens")
        steps = decode_tokens - warmup_tokens - 1
        return {
            "tbt_mean_ms": statistics.mean(gaps) * 1000,
            "tbt_p95_ms": gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))] * 1000,
            "duty": injected / steps,
        }

    async def _profile(
        self,
        *,
        tbt_slack: float = 0.10,
        max_decode_tasks: int = 16,
        step_tokens: Tuple[int, ...] = (16, 32, 48, 64, 96, 128, 192, 256),
        decode_tokens: int = 128,
        warmup_tokens: int = 8,
        stat: str = "tbt_mean_ms",
    ) -> None:
        """Find, per decode concurrency b, the largest uncached token count N one
        speculative request may carry while `stat` TBT stays within `tbt_slack`
        of the unloaded baseline. The unit is tokens per engine step: the request
        lands whole in one step, so N bounds that step's extra compute.

        `stat` is the gate; `aggregate.py`'s no-harm check uses the mean, so the
        default matches it. The median is deliberately not offered: an injector
        that hits fewer than half the steps leaves it untouched.
        """
        if tbt_slack < 0:
            raise ValueError("tbt_slack must be >= 0")
        if max_decode_tasks < 1:
            raise ValueError("max_decode_tasks must be >= 1")
        if decode_tokens <= warmup_tokens + 2:
            raise ValueError("decode_tokens must exceed warmup_tokens by at least 3")
        if stat not in ("tbt_mean_ms", "tbt_p95_ms"):
            raise ValueError("stat must be tbt_mean_ms or tbt_p95_ms")

        self.spec_tokens_per_step.clear()
        self.profile_curve.clear()

        # Exclude model/CUDA first-request initialization from the baseline.
        await self._measure_tbt(1, 0, decode_tokens=max(16, warmup_tokens + 3),
                                warmup_tokens=min(warmup_tokens, 4))

        for b in range(1, max_decode_tasks + 1):
            base = await self._measure_tbt(b, 0, decode_tokens=decode_tokens, warmup_tokens=warmup_tokens)
            self.profile_curve.append({"b": b, "n": 0, **base})
            limit = base[stat] * (1.0 + tbt_slack)
            safe = 0
            for n in step_tokens:
                m = await self._measure_tbt(b, n, decode_tokens=decode_tokens, warmup_tokens=warmup_tokens)
                self.profile_curve.append({"b": b, "n": n, **m})
                print(f"[profile] b={b} n={n} tbt_mean={m['tbt_mean_ms']:.2f}ms "
                      f"(+{m['tbt_mean_ms'] / base['tbt_mean_ms'] - 1:.1%}) "
                      f"tbt_p95={m['tbt_p95_ms']:.2f}ms duty={m['duty']:.2f}", flush=True)
                if m[stat] > limit:
                    break
                safe = n
            self.spec_tokens_per_step[b] = safe
            print(f"[profile] decode_tasks={b} baseline_tbt={base['tbt_mean_ms']:.2f}ms "
                  f"p95={base['tbt_p95_ms']:.2f}ms spec_tokens_per_step={safe}", flush=True)
            if safe == 0:
                break
