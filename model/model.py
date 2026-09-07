"""SGLang Engine wrapped for serving: real generate/decode, a shadow KV ledger, and the
two scheduler RPCs the planner steers HiCache with (promote, kv_priority).

    engine = Engine("Qwen/Qwen3-8B")
    text = await engine.generate(prompt, request_id)
    await engine.promote(token_ids, request_id)   # host-tier prefix back onto the device
"""

import asyncio
import functools
import random
import time
import uuid
import json
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.engine import Engine as SGLangEngine #type: ignore

from model.device_profiler import DeviceProfile
import model.promote  # noqa: F401  patches Scheduler.hicache_promote (see module doc)

# Host-DRAM KV tier (HiCache L2): GPU evictions demote here, prefetch promotes back.
# HiCache asserts the host pool is larger than the device pool, so this has to stay
# above whatever `mem_fraction_static` leaves for KV on the card.
HOST_KV_GB = 8
LANDED_STRIDE = 8   # bookkeeping granularity for prefixes we have already computed

# SGLang schedules the HIGHEST priority value first — the opposite of vLLM. A tool-loop
# turn of a call already in flight outranks a fresh call.
PRIORITY_CONT = 2
PRIORITY_REAL = 1


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
        self.device_profile: Optional[DeviceProfile] = None  # measured rates; see model.device_profiler
        self.promote_no_room = False  # last promote() deferral was for device space, not queue depth
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
    def serving(self) -> bool:
        """Real requests in flight."""
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

    # ---- KV ledger -----------------------------------------------------------

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

    def cost(self, toks: List[int]) -> Tuple[int, int]:
        """(uncached, host) token counts: what must be computed vs what a host→device
        promotion covers."""
        return self.ledger.cost(toks)

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"model": self.name,
                       "device": self.device_profile.as_dict() if self.device_profile else None},
                      f, indent=1)

    def load_profile(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("model") != self.name or not data.get("device"):
            return False
        try:
            self.device_profile = DeviceProfile.from_dict(data["device"])
        except (KeyError, TypeError, ValueError):
            return False  # measured under an older definition: re-measure
        return True

    # ---- real requests -------------------------------------------------------

    async def generate(self, prompt: str, request_id: str,
                       sampling_params: Optional[Dict[str, Any]] = None,
                       priority: int = PRIORITY_REAL,
                       progress: Optional[Callable[[float, int], None]] = None,
                       ids: Optional[List[int]] = None) -> str:
        """A real request: prefill until its first token, then decode. Higher `priority`
        is scheduled first. `progress(first_token_at, out_tokens)` is called on every
        output. `ids` is the prompt's tokenization when the caller already has it."""
        text = ""
        out_tokens = 0
        ids = ids if ids is not None else self.tokenize(prompt)
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        self._inflight_prefill += 1
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

    # ---- scheduler RPCs ------------------------------------------------------

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
                              retire: List[Tuple[List[int], int]] = []) -> bool:
        """Push the eviction map (model.promote.kv_priority) through a scheduler RPC."""
        rpc = functools.partial(self.engine.collective_rpc, "kv_priority",
                                demote=demote, protect=protect, rid=request_id, retire=list(retire))
        try:
            await asyncio.get_running_loop().run_in_executor(None, rpc)
        except AssertionError as e:
            print(f"[priority] {request_id} failed: {e}", flush=True)
            return False
        return True

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
        print(f"[warmup] engine ready in {(time.perf_counter() - started) * 1000:.0f} ms", flush=True)

    # ------------------ Internal helpers ---------------------------------------------------
    def _random_ids(self, num_tokens: int) -> List[int]:
        """`num_tokens` ids no earlier request has seen: an uncached prompt of exact length."""
        vocab = len(self._tokenizer)  #type: ignore
        return [random.randrange(1000, vocab - 1000) for _ in range(num_tokens)]
