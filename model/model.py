"""SGLang Engine wrapped for serving: real generate/decode plus speculative prefill and
probes that stay within a profiled per-step token budget.

    engine = Engine("Qwen/Qwen3-8B")
    text = await engine.generate(prompt, request_id)
    await engine.prefill(prompt, request_id)   # KV lands in the radix cache
"""

import asyncio
import random
import statistics
import time
import uuid
import json
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from sglang.srt.entrypoints.engine import Engine as SGLangEngine

# Host-DRAM KV tier (HiCache L2): GPU evictions demote here, prefetch promotes back.
# HiCache asserts the host pool is larger than the device pool, so this has to stay
# above whatever `mem_fraction_static` leaves for KV on the card.
HOST_KV_GB = 8
LANDED_STRIDE = 8   # bookkeeping granularity for prefixes we have already computed

# SGLang schedules the HIGHEST priority value first — the opposite of vLLM. A tool-loop
# turn of a call already in flight outranks a fresh call, which outranks speculation.
PRIORITY_CONT = 2
PRIORITY_REAL = 1
PRIORITY_SPEC = 0


class Logprob(NamedTuple):
    """One entry of a probe's top-k distribution."""
    logprob: float


class Engine:
    def __init__(self, model_name: str, **engine_kwargs) -> None:
        engine_kwargs.setdefault("enable_priority_scheduling", True)
        # Two-tier KV cache is always on: a block evicted from the GPU survives in
        # host memory, so a predicted prefix is promoted over PCIe instead of
        # recomputed. `host_cache_gb=0` disables the tier (measurement baseline).
        host_gb = int(engine_kwargs.pop("host_cache_gb", HOST_KV_GB))
        if host_gb > 0:
            engine_kwargs.setdefault("enable_hierarchical_cache", True)
            engine_kwargs.setdefault("hicache_size", host_gb)  # gigabytes; overrides hicache_ratio
        self.name = model_name
        self.engine = SGLangEngine(model_path=model_name, **engine_kwargs)

        self._inflight_prefill:int = 0
        self._inflight_decode:int = 0
        self._spec_tokens_per_step:Dict[int, int] = {} # uncached spec prefill tokens one step may carry, keyed by decode concurrency
        self.DEFAULT_SPEC_TOKENS_PER_STEP = 32
        self.SPEC_TIMEOUT_S = 20.0
        self.tbt_ms: Optional[float] = None  # profiled single-stream decode step time (planner's time unit)
        self._spec_tasks: Dict[asyncio.Task, int] = {}  # speculative consumers in flight -> their uncached-token cost
        self._landed: Dict[int, float] = {}   # chain hash of a computed prefix -> when it landed
        self.LANDED_MAX = 1 << 16
        self.sp: Dict[str, Any] = {"temperature": 0.7}
        # stats of the last real request: ttft_ms, prompt_tokens, and the cached total split
        # by tier (cached_device / cached_host) — a host hit is a promotion, not a recompute.
        self.last: Dict[str, float] = {}


    @property
    def busy(self) -> bool:
        return self._inflight_prefill + self._inflight_decode + len(self._spec_tasks) > 0

    @property
    def serving(self) -> bool:
        """Real requests in flight (speculation alone does not pin an engine)."""
        return self._inflight_prefill + self._inflight_decode > 0

    def shutdown(self) -> None:
        self.engine.shutdown()

    # ---- tokenizer helpers ---------------------------------------------------

    @property
    def _tokenizer(self):
        return self.engine.tokenizer_manager.tokenizer

    def tokenize(self, prompt: str) -> List[int]:
        return self._tokenizer.encode(prompt)

    def render(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Disable thinking. Multimodal content lists are flattened to their text
        parts first — the chat template renders non-string content as empty.
        `tools` must be passed for native-tools requests so the rendered bytes
        match what the client-side template would produce."""
        msgs = [dict(m, content="\n".join(str(p.get("text", "")) for p in m["content"]
                                          if isinstance(p, dict) and p.get("type") == "text"))
                if isinstance(m.get("content"), list) else m for m in messages]
        return self._tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # ---- speculative budget --------------------------------------------------

    def spec_allowance(self) -> int:
        """How many more speculative tokens can we accept right now without interfering with normal model work"""
        if self._inflight_prefill > 0:
            return 0  # a real prefill owns the step
        if self._inflight_decode == 0:
            return 1 << 30  # no real decode to protect
        budget = self._spec_tokens_per_step.get(self._inflight_decode)
        if budget is None:  # unprofiled concurrency: a conservative default instead of no speculation at all
            budget = self.DEFAULT_SPEC_TOKENS_PER_STEP if not self._spec_tokens_per_step else 0
        return budget - sum(self._spec_tasks.values())

    @property
    def _stride(self) -> int:
        """Bookkeeping step, rounded up to a whole number of radix-cache pages so that
        every prefix we record is one the cache can actually reuse."""
        page = getattr(self.engine.server_args, "page_size", None) or 1
        return page * max(1, LANDED_STRIDE // page)

    def _prefix_hashes(self, toks: List[int]):
        """Chain hash of each `_stride`-aligned prefix of `toks`, shortest first."""
        step, h = self._stride, 0
        for i in range(0, len(toks) - step + 1, step):
            h = hash((h, tuple(toks[i:i + step])))
            yield h

    def note_landed(self, toks: List[int]) -> None:
        """Record that this prompt's KV was computed, so a later prompt sharing its
        prefix is not charged again for it."""
        now = time.monotonic()
        for h in self._prefix_hashes(toks):
            self._landed[h] = now
        if len(self._landed) > self.LANDED_MAX:  # drop the older half
            cut = statistics.median(self._landed.values())
            self._landed = {k: v for k, v in self._landed.items() if v >= cut}

    def uncached_cost(self, toks: List[int]) -> int:
        """Tokens the engine would actually have to compute for `toks`: everything past
        the longest prefix we have already landed. The speculation budget is denominated
        in *uncached* tokens, so charging the full length rejects work that is nearly
        free — exactly the promotion-shaped work worth doing when the GPU is busy.
        Conservative by construction: a partly-cached stride counts as uncached."""
        matched = 0
        for i, h in enumerate(self._prefix_hashes(toks), start=1):
            if h not in self._landed:
                break
            matched = i * self._stride
        return len(toks) - matched

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"model": self.name, "unit": "tokens_per_step", "tbt_ms": self.tbt_ms,
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
        self.tbt_ms = data.get("tbt_ms") or self.tbt_ms
        return True

    # ---- real requests -------------------------------------------------------

    async def generate(self, prompt: str, request_id: str,
                       sampling_params: Optional[Dict[str, Any]] = None,
                       priority: int = PRIORITY_REAL,
                       progress: Optional[Callable[[float, int], None]] = None) -> str:
        """A real request: prefill until its first token, then decode. Any speculative
        request in flight is killed on admission so the prefill step is not shared.
        Higher `priority` is scheduled first; speculation stays at PRIORITY_SPEC.
        `progress(first_token_at, out_tokens)` is called on every output."""
        text = ""
        out_tokens = 0
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        self._inflight_prefill += 1
        self.kill_speculation()
        try:
            async for out in await self.engine.async_generate(
                    prompt=prompt, sampling_params=sampling_params or self.sp,
                    rid=request_id, priority=priority, stream=True):
                meta = out.get("meta_info") or {}
                if first_token:
                    first_token = False
                    first_token_at = time.perf_counter()
                    self._inflight_prefill -= 1
                    self._inflight_decode += 1
                    self.last = self._cache_stats(meta)
                    self.last["ttft_ms"] = (first_token_at - started) * 1000
                    print(f"[serve] {request_id} " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                            for k, v in self.last.items()), flush=True)
                text = out.get("text") or ""          # cumulative: incremental_streaming_output is off
                out_tokens = len(out.get("output_ids") or ())
                if progress is not None:
                    progress(first_token_at, out_tokens)
        finally:
            # exactly one decrement per increment, whichever phase we ended in
            if first_token:
                self._inflight_prefill -= 1
            else:
                self._inflight_decode -= 1
                self.note_landed(self.tokenize(prompt))  # a served prompt is cached too
                print(f"[decode] {request_id} decode_ms={(time.perf_counter() - first_token_at) * 1000:.2f} "
                      f"output_tokens={out_tokens}", flush=True)
        return text

    @staticmethod
    def _cache_stats(meta: Dict[str, Any]) -> Dict[str, float]:
        """Prompt tokens served from cache, split by tier. `cached_host` are tokens the
        hierarchical cache promoted from host DRAM instead of recomputing — the number
        that says whether the promotion path is doing any work at all."""
        detail = meta.get("cached_tokens_details") or {}
        return {"prompt_tokens": meta.get("prompt_tokens") or 0,
                "cached_tokens": meta.get("cached_tokens") or 0,
                "cached_device": detail.get("device") or 0,
                "cached_host": detail.get("host") or 0}

    # ---- speculative requests ------------------------------------------------

    def kill_speculation(self) -> int:
        """Abort every speculative request in flight; returns how many."""
        live = [t for t in self._spec_tasks if not t.done()]
        for t in live:
            t.cancel()
        return len(live)

    async def _speculative(self, prompt: Optional[str], input_ids: Optional[List[int]],
                           sampling_params: Dict[str, Any], request_id: str, cost: int = 0,
                           **kwargs) -> Any:
        """Run one PRIORITY_SPEC request to completion; return its last output, or None
        if kill_speculation() aborted it. Consumed in its own task so a real admission
        can cancel exactly this request. `cost` is held against the speculation budget
        until the request ends."""
        async def consume():
            last = None
            async for out in await self.engine.async_generate(
                    prompt=prompt, input_ids=input_ids, sampling_params=sampling_params,
                    rid=request_id, priority=PRIORITY_SPEC, stream=True, **kwargs):
                last = out
            return last

        async def bounded():
            try:
                return await asyncio.wait_for(consume(), self.SPEC_TIMEOUT_S)
            except asyncio.TimeoutError:
                print(f"[spec] {request_id} timed out after {self.SPEC_TIMEOUT_S}s; aborted", flush=True)
                self._abort(request_id)
                return None

        task = asyncio.create_task(bounded())
        self._spec_tasks[task] = cost
        try:
            return await task
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if not task.cancelled() or (me is not None and me.cancelling()):
                raise  # the caller itself is being cancelled (shutdown), not the request
            self._abort(request_id)
            return None
        finally:
            self._spec_tasks.pop(task, None)

    def _abort(self, request_id: str) -> None:
        """Cancelling the consumer does not reach the scheduler; tell it explicitly."""
        try:
            self.engine.tokenizer_manager.abort_request(rid=request_id)
        except Exception:
            pass

    async def prefill(self, prompt: str | List[int], request_id: str, cost: Optional[int] = None) -> bool:
        """Speculative prefill: one PRIORITY_SPEC request that leaves the prompt's KV in
        the radix cache. `cost` defaults to the tokens not already cached."""
        started = time.perf_counter()
        ids = list(prompt) if isinstance(prompt, list) else self.tokenize(prompt)
        if cost is None:
            cost = self.uncached_cost(ids)
        sp = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
        done = await self._speculative(None, ids, sp, request_id, cost) is not None
        if done:
            self.note_landed(ids)
        print(f"[prefill] {request_id} tokens={len(ids)} cost_tokens={cost} "
              f"duration_ms={(time.perf_counter() - started) * 1000:.2f}"
              + ("" if done else " aborted=1"), flush=True)
        return done

    async def probe(self, prompt: str, request_id: str) -> Optional[Dict[int, Logprob]]:
        """Greedy one-token probe: top-20 logprobs at the first position. None when the
        budget cannot take it or a real admission killed the probe; retry later."""
        ids = self.tokenize(prompt)
        cost = self.uncached_cost(ids)
        if self.spec_allowance() < cost:
            return None
        sp = {"max_new_tokens": 1, "temperature": 0.0}
        output = await self._speculative(None, ids, sp, request_id, cost,
                                         return_logprob=True, top_logprobs_num=20)
        if output is None:
            return None
        self.note_landed(ids)
        # meta_info["output_top_logprobs"] is one list per generated position, each entry
        # a list of (logprob, token_id, token_text|None).
        top = ((output.get("meta_info") or {}).get("output_top_logprobs") or [None])[0]
        return {tid: Logprob(lp) for lp, tid, *_ in top} if top else {}

    # ---- warmup --------------------------------------------------

    async def warmup(self) -> None:
        """Pay first-request costs."""
        started = time.perf_counter()
        self.render([{"role": "system", "content": "warmup"}, {"role": "user", "content": "warmup"}])
        decode = {"max_new_tokens": 4, "temperature": 0.0, "ignore_eos": True}
        for n in (64, 512):
            async for _ in await self.engine.async_generate(
                input_ids=self._random_ids(n), sampling_params=decode,
                rid=f"warmup-real-{n}-{uuid.uuid4().hex}", priority=PRIORITY_REAL, stream=True
            ):
                pass
        await self._speculative(None, self._random_ids(64),
                                {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
                                f"warmup-spec-{uuid.uuid4().hex}")
        await self._speculative(None, self._random_ids(64),
                                {"max_new_tokens": 1, "temperature": 0.0},
                                f"warmup-probe-{uuid.uuid4().hex}",
                                return_logprob=True, top_logprobs_num=20)
        print(f"[warmup] engine ready in {(time.perf_counter() - started) * 1000:.0f} ms", flush=True)

    # ------------------ Internal helpers ---------------------------------------------------
    def _random_ids(self, num_tokens: int) -> List[int]:
        """`num_tokens` ids no earlier request has seen: an uncached prompt of exact length."""
        vocab = len(self._tokenizer)
        return [random.randrange(1000, vocab - 1000) for _ in range(num_tokens)]

    async def _measure_tbt(self, num_decode: int, num_prefill: int, sampling_params: Dict[str, Any],
                           injectors: int = 1) -> dict:
        """TBT of `num_decode` decode streams (each `sampling_params["max_new_tokens"]` long)
        while PRIORITY_SPEC prefill requests of `num_prefill` uncached tokens are
        injected back-to-back, one at a time. `num_prefill == 0` measures the baseline.
        Returns tbt_mean_ms, tbt_p95_ms and duty (injections completed per decode step)."""
        if num_decode < 1 or num_prefill < 0:
            raise ValueError("num_decode must be >= 1 and num_prefill >= 0")
        decode_tokens = sampling_params.get("max_new_tokens") or 0
        warmup_tokens = min(8, max(0, decode_tokens - 4))
        if decode_tokens < warmup_tokens + 3:
            raise ValueError("sampling_params['max_new_tokens'] too small to measure TBT")

        stop = asyncio.Event()
        decode_started = asyncio.Event()
        started = 0
        injected = 0

        async def decode_worker(index: int) -> List[float]:
            nonlocal started
            ts: List[float] = []
            async for _ in await self.engine.async_generate(
                input_ids=self._random_ids(24), sampling_params=sampling_params,
                rid=f"profile-decode-{index}-{uuid.uuid4().hex}", priority=PRIORITY_REAL, stream=True,
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
            sp = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
            while not stop.is_set():
                async for _ in await self.engine.async_generate(
                    input_ids=self._random_ids(num_prefill), sampling_params=sp,
                    rid=f"profile-prefill-{uuid.uuid4().hex}", priority=PRIORITY_SPEC, stream=True,
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
            raise RuntimeError("not enough tokens to measure TBT; raise max_new_tokens")
        steps = decode_tokens - warmup_tokens - 1
        return {
            "tbt_mean_ms": statistics.mean(gaps) * 1000,
            "tbt_p95_ms": gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))] * 1000,
            "duty": injected / steps,
        }

    async def _profile(self, tbt_slack: float = 0.05, max_decode: int = 16,
                       prefill_sizes: Tuple[int, ...] = (16, 32, 48, 64, 96, 128, 192, 256),
                       decode_tokens: int = 128) -> Dict[int, int]:
        """Per decode concurrency b, the largest uncached prefill size one PRIORITY_SPEC
        request may carry while mean TBT stays within `tbt_slack` of the b-stream
        baseline. Fills and returns `_spec_tokens_per_step`."""
        sp = {"max_new_tokens": decode_tokens, "temperature": 0.0, "ignore_eos": True}
        self._spec_tokens_per_step.clear()
        # Pay first-request CUDA/kernel init outside the baseline.
        await self._measure_tbt(1, 0, {"max_new_tokens": 16, "temperature": 0.0, "ignore_eos": True})

        for b in range(1, max_decode + 1):
            base = await self._measure_tbt(b, 0, sp)
            if b == 1:
                self.tbt_ms = base["tbt_mean_ms"]
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
