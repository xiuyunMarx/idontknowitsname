import asyncio
import json
import random
import statistics
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logprobs import Logprob
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from console_helper.debug_output import console_debug, console_log, console_warn, console_error

class ModelEngine:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        self.model_name = model_name
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model_name, #type: ignore[arg-type]
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_prefix_caching=True,
                scheduling_policy="priority",
            )
        )
        self.sp = SamplingParams(temperature=0.7)
        self.spec_tokens_per_step: Dict[int, int] = {} # uncached speculative prefill tokens one engine step may carry
        self.profile_curve: List[Dict[str, float]] = []  # raw (b, n) -> TBT sweep, kept for plots
        self._decode_tasks = 0
        self._real_prefills = 0  # real requests still before their first token
        self._spec_inflight = False  # at most one speculative request at a time
        self._spec_task: Optional[asyncio.Task] = None  # consumer task of the speculative request in flight

    def tokenize(self, prompt: str) -> List[int]:
        return self.engine.get_tokenizer().encode(prompt)  # type: ignore[union-attr]

    def spec_allowance(self) -> int:
        """Uncached tokens the next speculative request may carry right now.
        A real prefill in flight already owns the step: nothing.
        Idle engine: unbounded.
        Otherwise the decode-only batch has compute slack up to the profiled per-step figure for its concurrency.

        The real-prefill test comes before the idle test on purpose: a real
        request is counted here from the moment generate() is entered, whereas
        the engine's own view of it lags admission (AsyncLLM.add_request awaits
        before the output processor registers the request), so an idle-looking
        engine may already have a real request on the way."""
        if self._spec_inflight or self._real_prefills > 0:
            return 0
        if not self.engine.output_processor.has_unfinished_requests():
            return 1 << 30
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

    @staticmethod
    def _with_salt(prompt: Any, cache_salt: Optional[str]) -> Any:
        """Attach a prefix-cache salt: requests with different salts never share
        cached blocks, which is how one workflow instance's KV cache is kept
        invisible to every other instance (FaaS-style isolation)."""
        if not cache_salt:
            return prompt
        if isinstance(prompt, str):
            return TextPrompt(prompt=prompt, cache_salt=cache_salt)  # type: ignore[call-arg]
        if isinstance(prompt, dict):
            return {**prompt, "cache_salt": cache_salt}
        return prompt

    async def generate(
        self,
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams | None = None,
        cache_salt: Optional[str] = None,
    ) -> str:
        text = ""
        prompt = self._with_salt(prompt, cache_salt)
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        out_tokens = 0
        self._real_prefills += 1  # A real call prefills until its first token, then decodes.
        self.kill_speculation()  # the step belongs to this request now
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

    def kill_speculation(self) -> bool:
        """Abort the speculative request in flight, if any. Called on every real admission."""
        task = self._spec_task
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _speculative(self, prompt: Any, sampling_params: SamplingParams, request_id: str) -> Any:
        """Run one priority-1 request to completion and return its last output,
        or None if kill_speculation() aborted it. The request is consumed in a
        task of its own so a real admission can cancel exactly that request:
        AsyncLLM.generate aborts the request when its consumer is cancelled."""
        async def consume():
            last = None
            async for out in self.engine.generate(prompt, sampling_params, request_id, priority=1):
                last = out
            return last

        self._spec_inflight = True
        self._spec_task = task = asyncio.create_task(consume())
        try:
            return await task
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if not task.cancelled() or (me is not None and me.cancelling()):
                raise  # the caller itself is being cancelled (shutdown), not the request
            return None
        finally:
            self._spec_inflight = False
            self._spec_task = None

    async def prefill(self, prefill_prompt: str | List[int], request_id: str, cost: int | None = None,
                      cache_salt: Optional[str] = None) -> bool:
        """`cost` is the caller's estimate of uncached tokens; defaults to full length.
        Returns False when a real admission killed the request before it completed;
        the caller must not count those tokens as warm."""
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        start_time = time.perf_counter_ns()
        if isinstance(prefill_prompt, list):
            cost = len(prefill_prompt) if cost is None else cost
            prefill_prompt = TokensPrompt(prompt_token_ids=prefill_prompt) #type: ignore[call-arg]
        elif cost is None:
            cost = len(self.tokenize(prefill_prompt))
        prefill_prompt = self._with_salt(prefill_prompt, cache_salt)
        done = await self._speculative(prefill_prompt, sampling_params, request_id) is not None
        # An aborted request keeps its cost in the log line: it may have spent a step
        # before the abort landed, so the spend accounting errs conservative.
        print(
            f"[prefill] {request_id} cost_tokens={cost} duration_ms={(time.perf_counter_ns() - start_time) / 1e6:.2f}"
            + ("" if done else " aborted=1"),
            flush=True,
        )
        return done

    async def probe(self, prompt: str, request_id: str, cost: int = 64,
                    cache_salt: Optional[str] = None) -> Optional[Dict[int, Logprob]]:
        """Greedy one-token probe; returns the top logprobs at the first position.
        The prompt rides the router's own just-computed prefix, so its uncached
        cost is a few think-block tokens. None when a real prefill owns the step
        or a real admission killed the probe: the caller should probe again later."""
        if self._real_prefills > 0:
            return None
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        output = await self._speculative(self._with_salt(prompt, cache_salt), sampling_params, request_id)
        if output is None:
            return None
        if output.outputs and output.outputs[0].logprobs:
            return output.outputs[0].logprobs[0]
        return {}

    async def warmup(self) -> None:
        """Bring a fresh process to the already-serving state the cold-start regime
        assumes, before any tenant can connect. The first request of a process pays
        CUDA/kernel initialisation, the first chat-template render, and AsyncLLM's
        one-off get_supported_tasks RPC (the only await on the admission path that
        yields before the engine sees the request). Random token ids keep the
        prefix cache free of anything a real prompt could hit."""
        started = time.perf_counter()
        self.render([{"role": "system", "content": "warmup"}, {"role": "user", "content": "warmup"}])
        decode = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
        for n in (64, 512):  # a short and a long prefill, each followed by a few decode steps
            async for _ in self.engine.generate(
                TokensPrompt(prompt_token_ids=self._random_ids(n)), decode, f"warmup-real-{n}-{uuid.uuid4().hex}"  # type: ignore[call-arg]
            ):
                pass
        await self._speculative(TokensPrompt(prompt_token_ids=self._random_ids(64)),  # type: ignore[call-arg]
                                SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
                                f"warmup-spec-{uuid.uuid4().hex}")
        await self._speculative(TokensPrompt(prompt_token_ids=self._random_ids(64)),  # type: ignore[call-arg]
                                SamplingParams(max_tokens=1, temperature=0.0, logprobs=20),
                                f"warmup-probe-{uuid.uuid4().hex}")
        print(f"[warmup] engine ready in {(time.perf_counter() - started) * 1000:.0f} ms", flush=True)

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
