from model_runner.model_engine import InstanceEngine
import asyncio
import random
import statistics
import time
import uuid
import json
from typing import Any, Dict, List, Optional, Tuple
from vllm import SamplingParams
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logprobs import Logprob

class Engine:
    def __init__(self, model_name: str, engine: InstanceEngine) -> None:
        self.name = model_name
        self.engine = engine
        
        self._inflight_prefill:int = 0
        self._inflight_decode:int = 0
        self._spec_tokens_per_step:Dict[int, int] = {} # uncached spec prefill tokens one step may carry, keyed by decode concurrency
        self._spec_tasks: Dict[asyncio.Task, int] = {}  # speculative consumers in flight -> their uncached-token cost
        self.tier: str = "gpu"  # gpu | host | ssd
        self.sp = SamplingParams(temperature=0.7)
        self.last: Dict[str, float] = {}  # stats of the last real request (ttft_ms, cached_tokens, prompt_tokens)


    @classmethod
    async def create(cls, model_name: str, **engine_kwargs) -> "Engine":
        engine_kwargs.setdefault("scheduling_policy", "priority")  # spec requests run at priority 1
        engine = await InstanceEngine.create(model=model_name, **engine_kwargs)
        return cls(model_name, engine)


    
    # ---- tiers: gpu (resident) / host (pinned) / ssd -------------------------

    async def offload(self) -> dict:
        """gpu -> host."""
        await self.drain_speculation()  # offload refuses unfinished requests
        res = await self.engine.offload()
        self.tier = "host"
        return res

    async def kill(self) -> dict:
        """any -> ssd (host copy dropped)."""
        await self.drain_speculation()
        res = await self.engine.kill()
        self.tier = "ssd"
        return res

    async def prepare(self) -> dict:
        """ssd -> host."""
        res = await self.engine.prepare()
        self.tier = "host"
        return res

    async def load(self, prefill=None) -> dict:
        """host -> gpu (via prepare when on ssd), prefilling `prefill` under the weight stream."""
        if self.tier == "ssd":
            await self.prepare()
        res = await self.engine.load(prefill)
        self.tier = "gpu"
        return res

    @property
    def resident(self) -> bool:
        return self.engine.resident

    @property
    def busy(self) -> bool:
        return self._inflight_prefill + self._inflight_decode + len(self._spec_tasks) > 0

    # ---- tokenizer helpers ---------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        return self.engine.get_tokenizer().encode(prompt)  # type: ignore[union-attr]

    def render(self, messages: List[Dict[str, str]]) -> str:
        return self.engine.get_tokenizer().apply_chat_template(  
            messages, tokenize=False, add_generation_prompt=True) # type: ignore[union-attr]

    @staticmethod
    def _with_salt(prompt: Any, cache_salt: Optional[str]) -> Any:
        """Prefix-cache salt: requests with different salts never share cached
        blocks, so one tenant's KV cache is invisible to every other tenant."""
        if not cache_salt:
            return prompt
        if isinstance(prompt, str):
            return TextPrompt(prompt=prompt, cache_salt=cache_salt)  # type: ignore[call-arg]
        if isinstance(prompt, dict):
            return {**prompt, "cache_salt": cache_salt}
        return prompt

    # ---- speculative budget --------------------------------------------------

    def spec_allowance(self) -> int:
        """Uncached tokens a new speculative request may carry right now: the per-step
        budget at the current decode concurrency minus what in-flight speculation holds."""
        if self._inflight_prefill > 0:
            return 0  # a real prefill owns the step
        if self._inflight_decode == 0:
            return 1 << 30  # no real decode to protect
        return self._spec_tokens_per_step.get(self._inflight_decode, 0) - sum(self._spec_tasks.values())

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"model": self.name, "unit": "tokens_per_step",
                       "spec_tokens_per_step": self._spec_tokens_per_step}, f, indent=1)

    def load_profile(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("model") != self.name or data.get("unit") != "tokens_per_step":
            return False
        self._spec_tokens_per_step = {int(k): int(v) for k, v in data["spec_tokens_per_step"].items()}
        return True

    # ---- real requests -------------------------------------------------------

    async def generate(self, prompt: str, request_id: str,
                       sampling_params: Optional[SamplingParams] = None,
                       cache_salt: Optional[str] = None) -> str:
        """A real request: prefill until its first token, then decode. Any speculative
        request in flight is killed on admission so the prefill step is not shared."""
        text = ""
        out_tokens = 0
        prompt = self._with_salt(prompt, cache_salt)
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        self._inflight_prefill += 1
        self.kill_speculation()
        try:
            async for output in self.engine.generate(prompt, sampling_params or self.sp, request_id):
                if not output.outputs:
                    continue
                if first_token:
                    first_token = False
                    first_token_at = time.perf_counter()
                    self._inflight_prefill -= 1
                    self._inflight_decode += 1
                    self.last = {"ttft_ms": (first_token_at - started) * 1000,
                                 "cached_tokens": output.num_cached_tokens or 0,
                                 "prompt_tokens": len(output.prompt_token_ids or [])}
                    print(f"[serve] {request_id} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                            for k, v in self.last.items()), flush=True)
                text = output.outputs[0].text
                out_tokens = len(output.outputs[0].token_ids)
        finally:
            # exactly one decrement per increment, whichever phase we ended in
            if first_token:
                self._inflight_prefill -= 1
            else:
                self._inflight_decode -= 1
                print(f"[decode] {request_id} decode_ms={(time.perf_counter() - first_token_at) * 1000:.2f} "
                      f"output_tokens={out_tokens}", flush=True)
        return text

    # ---- speculative requests ------------------------------------------------

    def kill_speculation(self) -> int:
        """Abort every speculative request in flight; returns how many."""
        live = [t for t in self._spec_tasks if not t.done()]
        for t in live:
            t.cancel()
        return len(live)

    async def drain_speculation(self) -> None:
        """kill_speculation, then wait until the aborts have actually gone through."""
        self.kill_speculation()
        if self._spec_tasks:
            await asyncio.gather(*list(self._spec_tasks), return_exceptions=True)

    async def _speculative(self, prompt: Any, sampling_params: SamplingParams, request_id: str, cost: int = 0) -> Any:
        """Run one priority-1 request to completion; return its last output, or None
        if kill_speculation() aborted it. Consumed in its own task so a real admission
        can cancel exactly this request (AsyncLLM aborts it when the consumer is cancelled).
        `cost` is held against the speculation budget until the request ends."""
        async def consume():
            last = None
            async for out in self.engine.generate(prompt, sampling_params, request_id, priority=1):
                last = out
            return last

        task = asyncio.create_task(consume())
        self._spec_tasks[task] = cost
        try:
            return await task
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if not task.cancelled() or (me is not None and me.cancelling()):
                raise  # the caller itself is being cancelled (shutdown), not the request
            return None
        finally:
            self._spec_tasks.pop(task, None)

    async def prefill(self, prompt: str | List[int], request_id: str, cost: Optional[int] = None,
                      cache_salt: Optional[str] = None) -> bool:
        """Speculative prefill: one priority-1 request that leaves the prompt's KV in the
        prefix cache. Returns False if a real admission killed it; the caller must not
        count those tokens as warm."""
        started = time.perf_counter()
        if isinstance(prompt, list):
            cost = len(prompt) if cost is None else cost
            prompt = TokensPrompt(prompt_token_ids=prompt)  # type: ignore[call-arg,assignment]
        elif cost is None:
            cost = len(self.tokenize(prompt))
        sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        done = await self._speculative(self._with_salt(prompt, cache_salt), sp, request_id, cost) is not None
        print(f"[prefill] {request_id} cost_tokens={cost} duration_ms={(time.perf_counter() - started) * 1000:.2f}"
              + ("" if done else " aborted=1"), flush=True)
        return done

    async def probe(self, prompt: str, request_id: str,
                    cache_salt: Optional[str] = None) -> Optional[Dict[int, Logprob]]:
        """Greedy one-token probe: top-20 logprobs at the first position. None when the
        budget cannot take it or a real admission killed the probe; retry later."""
        cost = len(self.tokenize(prompt))
        if self.spec_allowance() < cost:
            return None
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        output = await self._speculative(self._with_salt(prompt, cache_salt), sp, request_id, cost)
        if output is None:
            return None
        if output.outputs and output.outputs[0].logprobs:
            return output.outputs[0].logprobs[0]
        return {}

    # ---- warmup / profiling --------------------------------------------------

    async def warmup(self) -> None:
        """Pay first-request costs (CUDA/kernel init, first chat-template render,
        AsyncLLM's one-off RPCs) before any tenant connects. Random token ids keep
        the prefix cache free of anything a real prompt could hit."""
        started = time.perf_counter()
        self.render([{"role": "system", "content": "warmup"}, {"role": "user", "content": "warmup"}])
        decode = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
        for n in (64, 512):
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
        """`num_tokens` ids no earlier request has seen: an uncached prompt of exact length."""
        vocab = len(self.engine.get_tokenizer())  # type: ignore[arg-type]
        return [random.randrange(1000, vocab - 1000) for _ in range(num_tokens)]

    async def _measure_tbt(self, num_decode: int, num_prefill: int, sampling_params: SamplingParams,
                           injectors: int = 1) -> dict:
        """TBT of `num_decode` decode streams (each `sampling_params.max_tokens` long)
        while priority-1 prefill requests of `num_prefill` uncached tokens are
        injected back-to-back, one at a time. `num_prefill == 0` measures the baseline.
        Returns tbt_mean_ms, tbt_p95_ms and duty (injections completed per decode step)."""
        if num_decode < 1 or num_prefill < 0:
            raise ValueError("num_decode must be >= 1 and num_prefill >= 0")
        decode_tokens = sampling_params.max_tokens or 0
        warmup_tokens = min(8, max(0, decode_tokens - 4))
        if decode_tokens < warmup_tokens + 3:
            raise ValueError("sampling_params.max_tokens too small to measure TBT")

        stop = asyncio.Event()
        decode_started = asyncio.Event()
        started = 0
        injected = 0

        async def decode_worker(index: int) -> List[float]:
            nonlocal started
            ts: List[float] = []
            async for _ in self.engine.generate(
                TokensPrompt(prompt_token_ids=self._random_ids(24)), sampling_params,  # type: ignore[call-arg]
                f"profile-decode-{index}-{uuid.uuid4().hex}",
            ):
                ts.append(time.perf_counter())
                if len(ts) == 1:
                    started += 1
                    if started == num_decode:
                        decode_started.set()
            return [ts[i] - ts[i - 1] for i in range(warmup_tokens + 1, len(ts))]

        async def injector() -> None:
            nonlocal injected
            await decode_started.wait()
            sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            while not stop.is_set():
                async for _ in self.engine.generate(
                    TokensPrompt(prompt_token_ids=self._random_ids(num_prefill)), sp,  # type: ignore[call-arg]
                    f"profile-prefill-{uuid.uuid4().hex}", priority=1,
                ):
                    pass
                injected += 1

        decoders = [asyncio.create_task(decode_worker(i)) for i in range(num_decode)]
        injs = [asyncio.create_task(injector()) for _ in range(injectors)] if num_prefill > 0 else []
        try:
            gaps_by_stream = await asyncio.gather(*decoders)
        finally:
            stop.set()
            decode_started.set()  # release the injectors if a decoder died at startup
            await asyncio.gather(*injs, return_exceptions=True)

        gaps = sorted(g for stream in gaps_by_stream for g in stream)
        if len(gaps) < num_decode * 4:
            raise RuntimeError("not enough tokens to measure TBT; raise sampling_params.max_tokens")
        steps = decode_tokens - warmup_tokens - 1
        return {
            "tbt_mean_ms": statistics.mean(gaps) * 1000,
            "tbt_p95_ms": gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))] * 1000,
            "duty": injected / steps,
        }

    async def _profile(self, tbt_slack: float = 0.05, max_decode: int = 16,
                       prefill_sizes: Tuple[int, ...] = (16, 32, 48, 64, 96, 128, 192, 256),
                       decode_tokens: int = 128) -> Dict[int, int]:
        """Per decode concurrency b, the largest uncached prefill size one priority-1
        request may carry while mean TBT stays within `tbt_slack` of the b-stream
        baseline. Fills and returns `_spec_tokens_per_step`."""
        sp = SamplingParams(max_tokens=decode_tokens, temperature=0.0, ignore_eos=True)
        self._spec_tokens_per_step.clear()
        # Pay first-request CUDA/kernel init outside the baseline.
        await self._measure_tbt(1, 0, SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))

        for b in range(1, max_decode + 1):
            base = await self._measure_tbt(b, 0, sp)
            limit = base["tbt_mean_ms"] * (1.0 + tbt_slack)
            safe = 0
            for n in prefill_sizes:
                m = await self._measure_tbt(b, n, sp)
                print(f"[profile] b={b} n={n} tbt_mean={m['tbt_mean_ms']:.2f}ms "
                      f"(+{m['tbt_mean_ms'] / base['tbt_mean_ms'] - 1:.1%}) "
                      f"p95={m['tbt_p95_ms']:.2f}ms duty={m['duty']:.2f}", flush=True)
                if m["tbt_mean_ms"] > limit:
                    break
                safe = n
            self._spec_tokens_per_step[b] = safe
            print(f"[profile] decode={b} baseline_tbt={base['tbt_mean_ms']:.2f}ms spec_tokens_per_step={safe}", flush=True)
            if safe == 0:
                break
        return self._spec_tokens_per_step
