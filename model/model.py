"""SGLang Engine wrapped for serving: real generate/decode plus speculative prefill and
probes that stay within a profiled per-step token budget.

    engine = Engine("Qwen/Qwen3-8B")
    text = await engine.generate(prompt, request_id)
    await engine.prefill(prompt, request_id)   # KV lands in the radix cache
"""

import asyncio
import functools
import random
import statistics
import time
import uuid
import json
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from sglang.srt.entrypoints.engine import Engine as SGLangEngine

from model.device_profiler import DeviceProfile
import model.promote  # noqa: F401  patches Scheduler.hicache_promote (see module doc)

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


class KVLedger:
    """Shadow model of the engine's two-tier KV cache, in stride-sized blocks.

    Chain-hashed prefixes mirror the radix cache's content addressing; dict order is
    the LRU order. The host tier is inclusive of the device tier (HiCache writes
    through), so a device overflow just drops the device entry — the host copy is
    what makes the eviction repairable by promotion. The shadow cannot see the
    engine's true eviction order, so every real request's measured cache split
    recalibrates it: drift is bounded by one request, not by the session."""

    def __init__(self, stride: int, device_cap_tokens: int, host_cap_tokens: int) -> None:
        self.stride = stride
        self.device_cap = max(1, device_cap_tokens // stride)   # capacities in blocks
        self.host_cap = host_cap_tokens // stride               # 0: host tier disabled
        self._device: "OrderedDict[int, float]" = OrderedDict() # hash -> landed_at; order = LRU
        self._host: "OrderedDict[int, float]" = OrderedDict()   # includes every device block
        self.drift = {"over": 0, "under": 0}                    # calibration corrections

    def hashes(self, toks: List[int]):
        """Chain hash of each stride-aligned prefix of `toks`, shortest first."""
        h = 0
        for i in range(0, len(toks) - self.stride + 1, self.stride):
            h = hash((h, tuple(toks[i:i + self.stride])))
            yield h

    def tier(self, h: int) -> str:
        return "device" if h in self._device else ("host" if h in self._host else "gone")

    def land(self, toks: List[int], now: float) -> None:
        """Record that every prefix block of `toks` was just computed on the device."""
        for h in self.hashes(toks):
            self._device[h] = now
            self._device.move_to_end(h)
            if self.host_cap:
                self._host[h] = now
                self._host.move_to_end(h)
        self._trim()

    def _trim(self) -> None:
        while len(self._device) > self.device_cap:  # LRU falls off; the host copy survives
            self._device.popitem(last=False)
        while self.host_cap and len(self._host) > self.host_cap:
            victim = next((h for h in self._host if h not in self._device), None)
            if victim is None:
                break
            del self._host[victim]

    def cost(self, toks: List[int]) -> Tuple[int, int]:
        """(uncached, host) token counts for `toks`. Uncached is everything past the
        longest hole-free cached prefix — a gone block kills the radix prefix match —
        and host counts the matched blocks that need promotion rather than sitting on
        the device. Conservative: a partly-cached stride counts as uncached."""
        matched = host = 0
        for h in self.hashes(toks):
            if h in self._device:
                pass
            elif h in self._host:
                host += self.stride
            else:
                break
            matched += self.stride
        return len(toks) - matched, host

    def resident_prefix(self, toks: List[int]) -> int:
        """Token count of the longest all-device prefix."""
        n = 0
        for h in self.hashes(toks):
            if h not in self._device:
                break
            n += self.stride
        return n

    def calibrate(self, toks: List[int], device_hit: int, host_hit: int, now: float) -> None:
        """Reconcile with a request's measured cache split: its first `device_hit`
        prompt tokens came from the device, the next `host_hit` from the host, the
        rest were recomputed. Fixes exactly those blocks; the caller lands the whole
        prompt afterwards (it is on the device now either way)."""
        edge_d = device_hit // self.stride * self.stride
        edge_h = (device_hit + host_hit) // self.stride * self.stride
        tiers = ("gone", "host", "device")
        off = 0
        for h in self.hashes(toks):
            truth = "device" if off < edge_d else ("host" if off < edge_h else "gone")
            off += self.stride
            said = self.tier(h)
            if said == truth:
                continue
            self.drift["over" if tiers.index(said) > tiers.index(truth) else "under"] += 1
            self._device.pop(h, None)
            self._host.pop(h, None)
            if truth == "device":
                self._device[h] = now
            if truth != "gone" and self.host_cap:
                self._host[h] = now
        self._trim()

    def device_used_tokens(self) -> int:
        return len(self._device) * self.stride

    def device_cap_tokens(self) -> int:
        return self.device_cap * self.stride


class Engine:
    def __init__(self, model_name: str, **engine_kwargs) -> None:
        engine_kwargs.setdefault("enable_priority_scheduling", True)
        host_gb = int(engine_kwargs.pop("host_cache_gb", HOST_KV_GB))
        device_kv_tokens = engine_kwargs.pop("device_kv_tokens", None)
        kv_bytes_per_token = engine_kwargs.pop("kv_bytes_per_token", None)
        if host_gb > 0:
            engine_kwargs.setdefault("enable_hierarchical_cache", True)
            engine_kwargs.setdefault("hicache_size", host_gb) 
            # "kernel" (sglang default): one copy kernel per layer. "direct" ran torch
            # gather/scatter per layer per K/V: ~0.75 GB/s host->device at c=4, 370 ms per
            # 2k-token demand load, and it slowed decode 25% (measured 2026-09-02).
            engine_kwargs.setdefault("hicache_io_backend", "kernel")
        self.name = model_name
        self.engine = SGLangEngine(model_path=model_name, **engine_kwargs)

        self._inflight_prefill:int = 0
        self._inflight_decode:int = 0
        self._spec_tokens_per_step:Dict[int, int] = {} # uncached spec prefill tokens one step may carry, keyed by decode concurrency
        self._oneshot_tokens_per_step:Dict[int, int] = {} # single isolated injection (probe lane), same keying
        self.device_profile: Optional[DeviceProfile] = None  # measured rates; see model.device_profiler
        self.DEFAULT_SPEC_TOKENS_PER_STEP = 32
        self.SPEC_TIMEOUT_S = 20.0
        self.tbt_ms: Optional[float] = None  # profiled single-stream decode step time (planner's time unit)
        self.promote_no_room = False  # last promote() deferral was for device space, not queue depth
        self.prefill_rate_tps: Optional[float] = None  # EMA of idle uncached-prefill throughput
        self._spec_tasks: Dict[asyncio.Task, int] = {}  # speculative consumers in flight -> their uncached-token cost
        if device_kv_tokens is None:
            init = getattr(self.engine, "_scheduler_init_result", None)
            infos = getattr(init, "scheduler_infos", None) or [{}]
            device_kv_tokens = infos[0].get("max_total_num_tokens") or (1 << 20)
        kvb = int(kv_bytes_per_token or self._kv_bytes_per_token())
        host_tokens = int(host_gb * 1e9 // kvb) if host_gb > 0 else 0  # sglang's host-pool math
        self.ledger = KVLedger(self._stride, int(device_kv_tokens), host_tokens)
        print(f"[ledger] device={device_kv_tokens} tokens host={host_tokens} tokens "
              f"stride={self._stride} kv_bytes_per_token={kvb}", flush=True)
        self.sp: Dict[str, Any] = {"temperature": 0.7}
        # stats of the last real request: ttft_ms, prompt_tokens, and the cached total split
        # by tier (cached_device / cached_host) — a host hit is a promotion, not a recompute.
        self.last: Dict[str, float] = {}

    def _kv_bytes_per_token(self) -> int:
        """2 (K,V) × layers × kv heads × head_dim × dtype bytes, mirroring sglang's
        host-pool sizing; falls back to Qwen3-8B's 147456."""
        try:
            cfg = self.engine.tokenizer_manager.model_config.hf_config #type: ignore
            heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
            head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            return 2 * cfg.num_hidden_layers * heads * head_dim * 2
        except Exception:
            return 147456


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
        return self.engine.tokenizer_manager.tokenizer #type: ignore

    def tokenize(self, prompt: str) -> List[int]:
        return self._tokenizer.encode(prompt) #type: ignore

    def render(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Disable thinking. Multimodal content lists are flattened to their text
        parts first — the chat template renders non-string content as empty.
        `tools` must be passed for native-tools requests so the rendered bytes
        match what the client-side template would produce ."""
        msgs = [dict(m, content="\n".join(str(p.get("text", "")) for p in m["content"]
                                          if isinstance(p, dict) and p.get("type") == "text"))
                if isinstance(m.get("content"), list) else m for m in messages]
        return self._tokenizer.apply_chat_template( #type: ignore
            msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False) #type: ignore

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

    def oneshot_allowance(self) -> int:
        """Budget for one isolated ride-along request (the probe pattern). The
        sustained-injection profile answers "can we do this every step"; a probe
        happens once per routing decision, and its cost is a one-off stall of a
        couple of decode steps — profiled separately in `_oneshot_tokens_per_step`.
        Same hard rules as the sustained lane: never beside a real prefill,
        unlimited when nothing decodes, one ride-along at a time."""
        if self._inflight_prefill > 0:
            return 0
        if self._inflight_decode == 0:
            return 1 << 30
        if self._spec_tasks or not self._oneshot_tokens_per_step:
            return 0
        keys = sorted(self._oneshot_tokens_per_step)
        b = self._inflight_decode
        key = next((k for k in keys if k >= b), keys[-1])
        return self._oneshot_tokens_per_step[key]

    @property
    def _stride(self) -> int:
        """Bookkeeping step, rounded up to a whole number of radix-cache pages so that
        every prefix we record is one the cache can actually reuse."""
        page = getattr(self.engine.server_args, "page_size", None) or 1
        return page * max(1, LANDED_STRIDE // page)

    def note_landed(self, toks: List[int]) -> None:
        """Record that this prompt's KV was computed, so a later prompt sharing its
        prefix is not charged again for it."""
        self.ledger.land(toks, time.monotonic())

    def uncached_cost(self, toks: List[int]) -> int:
        """Tokens the engine would actually have to compute for `toks`: everything past
        the longest prefix the ledger believes is still cached. The speculation budget
        is denominated in *uncached* tokens, so charging the full length rejects work
        that is nearly free — exactly the promotion-shaped work worth doing when the
        GPU is busy."""
        return self.ledger.cost(toks)[0]

    def cost(self, toks: List[int]) -> Tuple[int, int]:
        """(uncached, host) token counts: what must be computed vs what a host→device
        promotion covers. The planner prices creation and promotion differently."""
        return self.ledger.cost(toks)

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"model": self.name, "unit": "tokens_per_step", "tbt_ms": self.tbt_ms,
                       "spec_tokens_per_step": self._spec_tokens_per_step,
                       "oneshot_tokens_per_step": self._oneshot_tokens_per_step,
                       "device": self.device_profile.as_dict() if self.device_profile else None},
                      f, indent=1)

    def load_profile(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("model") != self.name or data.get("unit") != "tokens_per_step":
            return False
        if "oneshot_tokens_per_step" not in data or not data.get("device"):
            return False  # older profile: re-measure everything
        try:
            self.device_profile = DeviceProfile.from_dict(data["device"])
        except (KeyError, TypeError, ValueError):
            return False  # measured under an older definition: re-measure
        self._spec_tokens_per_step = {int(k): int(v) for k, v in data["spec_tokens_per_step"].items()}
        self._oneshot_tokens_per_step = {int(k): int(v)
                                         for k, v in data["oneshot_tokens_per_step"].items()}
        self.tbt_ms = data.get("tbt_ms") or self.tbt_ms
        return True

    # ---- real requests -------------------------------------------------------

    async def generate(self, prompt: str, request_id: str,
                       sampling_params: Optional[Dict[str, Any]] = None,
                       priority: int = PRIORITY_REAL,
                       progress: Optional[Callable[[float, int], None]] = None,
                       ids: Optional[List[int]] = None) -> str:
        """A real request: prefill until its first token, then decode. Any speculative
        request in flight is killed on admission so the prefill step is not shared.
        Higher `priority` is scheduled first; speculation stays at PRIORITY_SPEC.
        `progress(first_token_at, out_tokens)` is called on every output. `ids` is
        the prompt's tokenization when the caller already has it."""
        text = ""
        out_tokens = 0
        ids = ids if ids is not None else self.tokenize(prompt)
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        self._inflight_prefill += 1
        self.kill_speculation()
        try:
            async for out in await self.engine.async_generate(  #type: ignore
                    prompt=prompt, sampling_params=sampling_params or self.sp,
                    rid=request_id, priority=priority, stream=True):
                meta = out.get("meta_info") or {}
                if first_token:
                    first_token = False
                    first_token_at = time.perf_counter()
                    self._inflight_prefill -= 1
                    self._inflight_decode += 1
                    stats = self._cache_stats(meta)  # local: `last` is shared across requests
                    self._calibrate(ids, stats)
                    self.last = stats
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
                self.note_landed(ids)  # a served prompt is cached too
                print(f"[decode] {request_id} decode_ms={(time.perf_counter() - first_token_at) * 1000:.2f} "
                      f"output_tokens={out_tokens}", flush=True)
        return text

    def _calibrate(self, ids: List[int], stats: Dict[str, float]) -> None:
        """Reconcile the ledger with a request's measured cache split. Builds without
        per-tier details report only the total: count it as device."""
        dev, host = int(stats["cached_device"]), int(stats["cached_host"])
        if not dev and not host:
            dev = int(stats["cached_tokens"])
        before = dict(self.ledger.drift)
        self.ledger.calibrate(ids, dev, host, time.monotonic())
        d = {k: self.ledger.drift[k] - before[k] for k in before}
        if d["over"] or d["under"]:
            print(f"[ledger] drift +over={d['over']} +under={d['under']} "
                  f"(total {self.ledger.drift}) device_used={self.ledger.device_used_tokens()}",
                  flush=True)

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
            async for out in await self.engine.async_generate( #type: ignore
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
            self.engine.tokenizer_manager.abort_request(rid=request_id) #type: ignore
        except Exception:
            pass


    async def prefill(self, prompt: str | List[int], request_id: str, cost: Optional[int] = None) -> bool:
        """Speculative prefill: one PRIORITY_SPEC request that leaves the prompt's KV in
        the radix cache. `cost` defaults to the tokens not already cached."""
        started = time.perf_counter()
        ids = list(prompt) if isinstance(prompt, list) else self.tokenize(prompt)
        if cost is None:
            cost = self.uncached_cost(ids)
        was_idle = not self.serving
        sp = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
        out = await self._speculative(None, ids, sp, request_id, cost)
        done = out is not None
        duration = time.perf_counter() - started
        if done:
            meta = out.get("meta_info") or {}
            if meta.get("prompt_tokens"):
                self._calibrate(ids, self._cache_stats(meta))  # free ground truth
            self.note_landed(ids)
            if was_idle and cost > 64 and duration > 0:  # a clean sample of idle prefill throughput
                rate = cost / duration
                self.prefill_rate_tps = rate if self.prefill_rate_tps is None else \
                    0.7 * self.prefill_rate_tps + 0.3 * rate
        print(f"[prefill] {request_id} tokens={len(ids)} cost_tokens={cost} "
              f"duration_ms={duration * 1000:.2f}"
              + ("" if done else " aborted=1"), flush=True)
        return done

    async def promote(self, toks: List[int], request_id: str, wait: bool = False) -> Optional[bool]:
        """Host→device promotion (and LRU touch) of `toks`' cached prefix with no
        request: a scheduler RPC starts HiCache's own load-back (model.promote).
        Runs in a thread — the reply waits for the scheduler's next loop turn.
        None = deferred by the scheduler (real work first); retry later."""
        started = time.perf_counter()
        rpc = functools.partial(self.engine.collective_rpc, "hicache_promote",
                                token_ids=list(toks), rid=request_id, wait=wait)
        try:
            await asyncio.get_running_loop().run_in_executor(None, rpc)
        except AssertionError as e:
            if "deferred" in str(e):
                self.promote_no_room = "no room" in str(e)
                print(f"[promote-deferred] {request_id} {e}", flush=True)
                return None
            print(f"[promote] {request_id} failed: {e}", flush=True)
            return False
        self.note_landed(toks)   # the ledger's belief; the next real request calibrates
        print(f"[promote-rpc] {request_id} tokens={len(toks)} "
              f"duration_ms={(time.perf_counter() - started) * 1000:.2f}", flush=True)
        return True

    async def set_kv_priority(self, demote: List[Tuple[List[int], int]],
                              protect: List[Tuple[List[int], int]], request_id: str,
                              retire: List[Tuple[List[int], int]] = ()) -> bool:
        """Push the eviction map (model.promote.kv_priority) through a scheduler RPC."""
        rpc = functools.partial(self.engine.collective_rpc, "kv_priority",
                                demote=demote, protect=protect, rid=request_id, retire=list(retire))
        try:
            await asyncio.get_running_loop().run_in_executor(None, rpc)
        except AssertionError as e:
            print(f"[priority] {request_id} failed: {e}", flush=True)
            return False
        return True

    async def probe(self, prompt: str, request_id: str) -> Optional[Dict[int, Logprob]]:
        """Greedy one-token probe: top-20 logprobs at the first position. None when the
        budget cannot take it or a real admission killed the probe; retry later."""
        ids = self.tokenize(prompt)
        cost = self.uncached_cost(ids)
        if self.oneshot_allowance() < cost:  # probe rides the one-shot lane, not the sustained one
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
            async for _ in await self.engine.async_generate(  #type: ignore
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
        vocab = len(self._tokenizer)  #type: ignore
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
            async for _ in await self.engine.async_generate(  #type: ignore
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
                async for _ in await self.engine.async_generate(  #type: ignore
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

    async def _profile(self, tbt_slack: float = 0.15, max_decode: int = 16,
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

    async def _measure_oneshot(self, num_decode: int, num_prefill: int,
                               sampling_params: Dict[str, Any], shots: int = 3) -> dict:
        """Stall cost of a SINGLE isolated n-token PRIORITY_SPEC injection on running
        decode streams — the probe pattern, as opposed to `_measure_tbt`'s sustained
        back-to-back injection. Fires `shots` well-separated injections into one run;
        the quiet gaps of the same run are the baseline. Returns quiet p50/p95 and the
        worst gap that overlaps an injection window (all ms)."""
        stop = asyncio.Event()
        decode_started = asyncio.Event()
        started = 0
        windows: List[Tuple[float, float]] = []

        async def decode_worker(index: int) -> List[Tuple[float, float]]:
            nonlocal started
            ts: List[float] = []
            async for _ in await self.engine.async_generate(  #type: ignore
                    input_ids=self._random_ids(24), sampling_params=sampling_params,
                    rid=f"oneshot-decode-{index}-{uuid.uuid4().hex}",
                    priority=PRIORITY_REAL, stream=True):
                ts.append(time.perf_counter())
                if len(ts) == 1:
                    started += 1
                    if started == num_decode:
                        decode_started.set()
            return [(ts[i], ts[i] - ts[i - 1]) for i in range(9, len(ts))]

        async def shooter() -> None:
            await decode_started.wait()
            sp = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
            for _ in range(shots):
                await asyncio.sleep(0.35)   # let the streams run quiet between shots
                if stop.is_set():
                    return
                t0 = time.perf_counter()
                async for _ in await self.engine.async_generate(  #type: ignore
                        input_ids=self._random_ids(num_prefill), sampling_params=sp,
                        rid=f"oneshot-prefill-{uuid.uuid4().hex}",
                        priority=PRIORITY_SPEC, stream=True):
                    pass
                windows.append((t0, time.perf_counter()))

        decoders = [asyncio.create_task(decode_worker(i)) for i in range(num_decode)]
        shoot = asyncio.create_task(shooter())
        try:
            gaps_by_stream = await asyncio.gather(*decoders)
        finally:
            stop.set()
            decode_started.set()
            await asyncio.gather(shoot, return_exceptions=True)
        if not windows:
            raise RuntimeError("oneshot: no injection landed; raise max_new_tokens")
        pad = 0.05  # settle margin after an injection returns
        quiet: List[float] = []
        stall: List[float] = []
        for t, g in (x for stream in gaps_by_stream for x in stream):
            hit = any(t >= w0 and t - g <= w1 + pad for w0, w1 in windows)
            (stall if hit else quiet).append(g)
        quiet.sort()
        return {
            "quiet_p50_ms": quiet[len(quiet) // 2] * 1000 if quiet else 0.0,
            "quiet_p95_ms": quiet[min(len(quiet) - 1, int(0.95 * (len(quiet) - 1)))] * 1000 if quiet else 0.0,
            "stall_max_ms": max(stall) * 1000 if stall else 0.0,
            "shots": len(windows),
        }

    async def _profile_oneshot(self, bs: Tuple[int, ...] = (1, 2, 4, 8),
                               sizes: Tuple[int, ...] = (32, 64, 128, 256, 384, 512),
                               decode_tokens: int = 192) -> Dict[int, int]:
        """Per decode concurrency, the largest single-injection size whose worst-hit
        token is delayed by no more than ~two extra decode steps. Fills the probe
        admission lane `_oneshot_tokens_per_step`."""
        sp = {"max_new_tokens": decode_tokens, "temperature": 0.0, "ignore_eos": True}
        self._oneshot_tokens_per_step.clear()
        for b in bs:
            safe = 0
            for n in sizes:
                m = await self._measure_oneshot(b, n, sp)
                limit = max(3 * m["quiet_p50_ms"], m["quiet_p95_ms"] + m["quiet_p50_ms"])
                print(f"[profile-oneshot] b={b} n={n} stall_max={m['stall_max_ms']:.1f}ms "
                      f"quiet_p50={m['quiet_p50_ms']:.1f} p95={m['quiet_p95_ms']:.1f} "
                      f"shots={m['shots']}", flush=True)
                if m["stall_max_ms"] > limit:
                    break
                safe = n
            self._oneshot_tokens_per_step[b] = safe
            print(f"[profile-oneshot] decode={b} oneshot_tokens={safe}", flush=True)
            if safe == 0:
                break
        return self._oneshot_tokens_per_step
