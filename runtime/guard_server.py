from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM
from runtime.incremental_feed import FeedStore, eval_provenance, extract_visit_ctx, extract_walker_fields, final_messages, partial_user_prompt, reorder_binding_zone
from runtime.scheduler.scheduler import Scheduler
from runtime.speculate import SpeculateCandidate
from static.static_parser import bfs_distances, entry_distances, extract_finish_output
from runtime.side_runtime import SideRuntime


class RawRepr:
    """Wraps a repr string sent over the wire so build_full_prompt's `{value!r}` reproduces the caller-side repr byte-identically."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


class Monitor:
    """Watches call activity and drives topology-based speculative warming.

    Fires on call START: the successors' invariants need no runtime values, so the
    earliest useful moment to warm them is the moment we know the current call is
    happening. Keeps an event log for overlap analysis.
    """

    def __init__(self, server: "GuardServer"):
        self.server = server
        self.events: List[Dict[str, Any]] = []

    def _log(self, kind: str, key: str, **extra: Any) -> None:
        self.events.append({"t": time.perf_counter(), "kind": kind, "key": key, **extra})

    def on_call_start(self, key: str) -> None:
        self.server.last_call_key = key  # current workflow position, for warm relevance
        self.server.probe_favored = set()  # the real position supersedes any probe guess
        self._log("call_start", key)

    def on_first_token(self, key: str, rid: str = "", walker_fields: Optional[Dict[str, str]] = None) -> None:
        """Speculate once the current call's first token is out: its TTFT-critical window is over, and the remaining decode (~95% of the call) is still ample   overlap for warming successors without interfering with prefill. Besides the
        spatial enqueue, this is the first provenance checkpoint: successor params
        sourced from state visible NOW (walker fields, constants) get RECORDED —
        engine-free session bookkeeping; the prefill rides the unified queue job.
        Field recordings stay topology-adjacent on purpose — walker fields are
        MUTABLE state, and adjacency doubles as a freshness heuristic (ret params
        have no such restriction; see on_call_end)."""
        if not self.server.speculate:
            return
        # Selection sites: fork the just-prefilled route prompt and sample the
        # choice distribution — the predicted candidate's calls get drainer
        # priority over their siblings (runtime/speculate.py).
        fn = self.server.side_rt.byllm_callsites.get(key)
        if self.server.probe and fn is not None and fn.decl.kind == "visit":
            self.server.speculator.launch(key, self.server.route_prompts.get(key, ""))
        # Record BEFORE enqueueing so the ring-1 jobs already carry their bindings.
        if self.server.prov_feed:
            for succ in self.server.side_rt.callsites_topo.get(key, []):
                self.server.record_provenance(succ, walker_fields=walker_fields)
        # Spatial speculation: enqueue EVERY call site reachable from here, nearest-first
        for succ, d in sorted(bfs_distances(self.server.side_rt.callsites_topo, key).items(), key=lambda kv: kv[1]):
            if succ == key:
                continue
            self._log("speculate", succ, after=key, dist=d)
            self.server.enqueue_spec(succ, reason=f"spec:{key}")

    def on_call_end(self, key: str, ttft: Optional[float], duration: float, rid: str = "", result_repr: Optional[str] = None) -> None:
        """Second provenance checkpoint: the call's parsed result is now known.
        DATAFLOW firing, not topology firing: consumers of ret(this call) come from
        the inverse provenance index, however many calls downstream they sit —
        `a=f(); b=g(); h(a,b)` records h.a the moment f ends. Safe at any distance:
        ret values are SSA-like (the variable never mutates after binding). Results
        accumulate in a store so a consumer fed at different times (h.a at f's end,
        h.b at g's end) evaluates against everything observed so far. Recording is
        engine-free; the PREFILL rides the consumer's unified queue job, which the
        admission gate (30ms idle grace) keeps out of the next call's prefill
        window and runs in the tool/think window right after this call if one
        exists — the only path by which the LAST producer's ret reaches its
        consumer in time. In a zero-gap flow the job just stays parked; the serve
        still replays the recorded session, losing only the warm."""
        self._log("call_end", key, ttft=ttft, duration=duration)
        if result_repr is None or not self.server.speculate or not self.server.prov_feed:
            return
        self.server.prov_results[key] = result_repr
        for ckey, _param in self.server.side_rt.ret_consumers.get(key, []):
            self._log("ret_record", ckey, after=key)
            self.server.record_provenance(ckey, results=self.server.prov_results)

    def on_warm(self, key: str, duration: float, reason: str) -> None:
        self._log("warm_done", key, duration=duration, reason=reason)

    def dump(self) -> None:
        t0 = self.events[0]["t"] if self.events else 0.0
        for e in self.events:
            extra = {k: v for k, v in e.items() if k not in ("t", "kind", "key")}
            print(f"  [{e['t'] - t0:8.3f}s] {e['kind']:<10} {e['key']}  {extra if extra else ''}")


class GuardServer:
    """Owns the AsyncLLM engine and the compiled SideRuntime of one Jac program."""

    def __init__(self, jac_path: str, model: str, *, max_model_len: int = 4096, gpu_memory_utilization: float = 0.45, enforce_eager: bool = True, type_check: bool = True, enable_probe:bool = False):
        self.side_rt = SideRuntime(jac_path, type_check=type_check)
        self.model_name = model
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=model,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            scheduling_policy="priority",  # warms must never delay a real call's prefill
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.monitor = Monitor(self)
        self.speculate = True  # Monitor's topology-driven warming; off = vanilla APC baseline
        self.prov_feed = True  # tier-3 provenance feeds; off isolates invariant-warm-only (tier-3 ablation)
        self.greedy = False  # force temperature=0 on served requests (deterministic experiments)
        self.feeds = FeedStore()
        self.prov_results: Dict[str, str] = {}  # producer key -> latest parsed-result repr (single-flow assumption)
        self._warm_inflight: set = set()
        self._warm_hash: Dict[str, str] = {}  # key -> sha1 of the last successfully warmed prefix (dedup)
        self.dump_dir = os.environ.get("GUARD_DUMP_PROMPTS") or None  # debug: write every rendered prompt to this dir
        self.last_call_key: Optional[str] = None  # current workflow position
        self.stats: Dict[str, List[Dict[str, Any]]] = {"warms": [], "calls": [], "feeds": [], "reorders": [], "probes": []}
        # Speculation admission: real-traffic tracking, idle-window execution, calibrated busy budget. All cap/state logic lives in the Scheduler.
        self.scheduler = Scheduler(self.engine)
        # Speculation ORDER: one pending job per call site ("prefill its longest
        # known stable prefix"); a single drainer executes them nearest-first by
        # live topology distance (ring 1 before ring 2, re-ranked between jobs).
        self.spec_queue: Dict[str, Dict[str, Any]] = {}
        self._queue_kick = asyncio.Event()
        self._drainer: Optional[asyncio.Task] = None
        self._drain_busy = False  # a popped job is executing (for warm_all's drain wait)
        # Probe speculator: for visit-by fan-out sites, let model say what to choose
        self.probe = enable_probe  
        self.speculator = SpeculateCandidate(self.engine, self)
        self.probe_favored: set = set()
        self.route_prompts: Dict[str, str] = {}  # visit key -> last served route prompt (probe input)
    # ------------------------------------------------------------------ engine
    @staticmethod
    def _flatten_content(content: Any) -> Any:
        """byllm user messages carry OpenAI parts lists ([{type: text, ...}]);
        chat templates want plain strings. Joining text parts reproduces the same
        bytes the invariant warm rendered directly."""
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            return "".join(parts)
        return content

    def _render(self, messages: List[Dict[str, Any]]) -> str:
        msgs = [{**m, "content": self._flatten_content(m.get("content"))} if isinstance(m, dict) else m for m in messages]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    async def _generate(self, prompt: str, sp: SamplingParams, request_id: str, priority: int = 0, on_first_token: Optional[Any] = None) -> Tuple[Any, Optional[float]]:
        """Run one request; returns (final RequestOutput, wall-clock TTFT incl. queueing)."""
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        final: Any = None
        async for out in self.engine.generate(prompt, sp, request_id, priority=priority):
            if ttft is None:
                ttft = time.perf_counter() - t0
                if on_first_token is not None:
                    on_first_token()
            final = out
        return final, ttft

    def _dump(self, kind: str, key: str, prompt: str) -> None:
        """Debug (GUARD_DUMP_PROMPTS=dir): write every rendered prompt to disk so
        warm/feed prefixes can be byte-diffed against the served prompt."""
        if not self.dump_dir:
            return
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.dump_dir) / f"{time.perf_counter():012.3f}-{kind}-{key.replace('/', '_')}.txt").write_text(prompt)

    # ------------------------------------------------------------------- warm
    def _best_session(self, key: str) -> Optional[Any]:
        """The session whose recorded bindings extend this call site's warmable
        prefix the furthest (single-flow: usually the one 'prov' session)."""
        sessions = [s for s in self.feeds.sessions_for(key) if s.fed]
        return max(sessions, key=lambda s: len(s.fed)) if sessions else None

    def _warm_user_prefix(self, key: str, fn: Any) -> Tuple[str, Optional[Any]]:
        """The longest currently-known stable user prefix for a call site — the
        invariant, extended by recorded bindings (func sites) or by the fed/learned
        graph ctx (visit sites) — plus the decoding-constraint schema whose grammar
        the warm should precompile. One prefill covers everything known so far."""
        so = getattr(fn.sampler, "structured_outputs", None) if fn.sampler is not None else None
        if fn.decl.kind != "visit":
            session = self._best_session(key)
            if session is not None:
                return partial_user_prompt(fn, session), so
            return fn.invariant_user_prefix, so
        ctx = self.feeds.get_visit_ctx(key)
        if not ctx:
            return fn.invariant_user_prefix, so
        schema = ctx.get("schema")
        inner = schema.get("json_schema", {}).get("schema") if isinstance(schema, dict) else None
        if inner:
            so = StructuredOutputsParams(json=inner)
        return fn.visit_stable_prefix(ctx.get("here", ""), ctx.get("candidates", "")), so

    def enqueue_spec(self, key: str, reason: str = "spec") -> None:
        """Queue (or refresh) the one speculation job a call site can have: prefill
        its longest known stable prefix. New bindings don't add jobs — they extend
        what the existing job will prefill when the drainer reaches it."""
        self.spec_queue[key] = {"reason": reason}
        self._queue_kick.set()

    def favor_candidate(self, visit_key: str, type_name: str, reason: str = "probe") -> List[str]:
        """Mark the byllm calls owned by the predicted candidate node type as
        drainer-favored (warmed before same-distance siblings) and (re-)enqueue
        them. Purely an ORDERING hint: unfavored candidates stay enqueued behind
        — a wrong probe restores blind order, it never drops coverage."""
        favored = [k for k in self.side_rt.callsites_topo.get(visit_key, []) if k.split(".")[0] == type_name]
        if favored:
            self.probe_favored = set(favored)
            for k in favored:
                self.enqueue_spec(k, reason=reason)
        return favored

    def start_drainer(self) -> None:
        self._drainer = asyncio.create_task(self._drain_spec())

    async def _drain_spec(self) -> None:
        """Single sequential consumer of spec_queue. Each iteration re-ranks the queue by BFS distance from the LIVE position and executes the nearest job:
        1. Jobs whose site is no longer reachable from the current position are behind the workflow
        and dropped — unless they carry recorded bindings."""
        topo = self.side_rt.callsites_topo
        while True:
            await self._queue_kick.wait()
            self._queue_kick.clear()
            while self.spec_queue:
                pos = self.last_call_key 
                dist = bfs_distances(topo, pos) if pos else entry_distances(topo)
                far = 10 ** 6
                # Probe-favored sites (the predicted winner of an in-flight selection) sort before same-distance siblings.
                key = min(self.spec_queue, key=lambda k: (dist.get(k, far), 0 if k in self.probe_favored else 1))
                self._drain_busy = True
                try:
                    job = self.spec_queue.pop(key)
                    if pos and key not in dist and self._best_session(key) is None:
                        self.stats["warms"].append({"key": key, "reason": job.get("reason", ""), "skipped": "unreachable"})
                        continue
                    await self.warm(key, reason=job.get("reason", "spec"))
                finally:
                    self._drain_busy = False

    async def warm(self, key: str, reason: str = "deploy") -> None:
        """Prefill this call site's stable prefix: the compile-time invariant, extended by the graph-derived visit context when one has been fed/learned. """
        fn = self.side_rt.byllm_callsites.get(key)
        if fn is None or key in self._warm_inflight:
            return
        # Content dedup: if the freshest stable prefix is byte-identical to what a
        # previous warm already prefilled, skip without taking an engine slot
        # (every served call re-enqueues its whole reachable set).
        prefix_now, _ = self._warm_user_prefix(key, fn)
        h = hashlib.sha1(prefix_now.encode()).hexdigest()
        if self._warm_hash.get(key) == h:
            self.stats["warms"].append({"key": key, "reason": reason, "skipped": "dup"})
            return
        self._warm_inflight.add(key)
        t0 = time.perf_counter()
        try:
            for attempt in range(4):
                # Admission-gated: parks while real traffic exceeds the calibrated
                # budget; executes in idle windows (tool exec, call gaps).
                async with self.scheduler.spec_slot() as preempted:
                    t_exec = time.perf_counter()
                    user_prefix, so = self._warm_user_prefix(key, fn)  # freshest ctx/bindings at execution time
                    sess = self._best_session(key)
                    n_fed = len(sess.fed) if sess is not None and fn.decl.kind != "visit" else 0
                    messages = [{"role": "system", "content": fn.invariant_system}, {"role": "user", "content": user_prefix}]
                    prompt = self._render(messages)
                    self._dump("warm", key, prompt)
                    sp = SamplingParams(max_tokens=1, temperature=0.0, structured_outputs=so)
                    gen = asyncio.create_task(self._generate(prompt, sp, f"warm-{uuid.uuid4().hex[:8]}", priority=1))
                    pre = asyncio.create_task(preempted.wait())
                    done, _pending = await asyncio.wait({gen, pre}, return_when=asyncio.FIRST_COMPLETED)
                    if gen in done:
                        pre.cancel()
                        now = time.perf_counter()
                        self._warm_hash[key] = hashlib.sha1(user_prefix.encode()).hexdigest()
                        self.stats["warms"].append({"key": key, "duration": now - t_exec, "parked": t_exec - t0, "attempt": attempt + 1, "fed": n_fed, "reason": reason, "prompt_head": prompt[:200]})
                        self.monitor.on_warm(key, now - t_exec, reason)
                        return
                    # A real request arrived: yield the engine. Cancelling the
                    # generator aborts the vLLM request; already-computed full
                    # blocks stay in the prefix cache, so the retry resumes cheap.
                    gen.cancel()
                    await asyncio.gather(gen, return_exceptions=True)
                    self.monitor._log("warm_preempted", key, reason=reason)
                # loop: re-park for the next idle window
        finally:
            self._warm_inflight.discard(key)

    def record_param(self, key: str, sid: str, name: str, value_repr: str) -> bool:
        """Record one binding — engine-free session bookkeeping. The prefill itself
        rides the call site's unified queue job: recording just (re-)enqueues it,
        and the job covers invariant + every binding known by the time the drainer
        admits it. One submission per site instead of one per arriving binding."""
        fn = self.side_rt.byllm_callsites.get(key)
        if fn is None:
            return False
        _session, changed = self.feeds.feed_param(key, sid, name, value_repr)
        if not changed:
            return False
        self.stats["feeds"].append({"key": key, "sid": sid, "param": name, "recorded": True})
        self.enqueue_spec(key, reason=f"feed:{name}")
        return True

    async def feed_param(self, key: str, sid: str, name: str, value_repr: str) -> None:
        """HTTP wire compat: /feed records and lets the drainer prefill."""
        self.record_param(key, sid, name, value_repr)

    async def feed_visit(self, key: str, here: str, candidates: str, schema: Optional[Any] = None) -> None:
        """Quasi-static feed: the graph-derived here/candidates zones of a visit site
        (and its choice schema, so the warm also precompiles the grammar)."""
        if self.feeds.set_visit_ctx(key, here, candidates, schema):
            self.enqueue_spec(key, reason="feed:visit")

    def record_provenance(self, succ: str, walker_fields: Optional[Dict[str, str]] = None, results: Optional[Dict[str, str]] = None) -> None:
        """Record every parameter of `succ` whose provenance spec is evaluable from
        the observed state. Params land in declaration order (so when everything
        fires at once the serve-time reorder is an identity). All provenance
        recordings for one consumer share ONE session (sid="prov") — a consumer
        fed from multiple events (h.a at f's end, h.b at g's end) accumulates
        arrival-ordered lines in a single replayable prefix instead of fragmenting
        across sessions. Re-recorded params: same value is a no-op, a changed
        value truncates from that line (PrefillSession.feed), so cross-request
        staleness self-heals."""
        prov = self.side_rt.provenance.get(succ)
        fn = self.side_rt.byllm_callsites.get(succ)
        if not prov or fn is None or fn.decl.kind == "visit":
            return
        for p in fn.decl.params:
            spec = prov.get(p["name"])
            if spec is None:
                continue
            val = eval_provenance(spec, walker_fields=walker_fields, results=results)
            if val is not None:
                self.record_param(succ, "prov", p["name"], val)

    async def warm_all(self) -> None:
        """Deploy-time prewarm: enqueue every call site ordered by BFS distance from
        the workflow's entry sites (the drainer keeps that order — no position is
        fixed yet), then wait for the queue to drain so 'warmed at boot' is true."""
        far = 10 ** 6
        dist = entry_distances(self.side_rt.callsites_topo)
        for key in sorted(self.side_rt.byllm_callsites, key=lambda k: dist.get(k, far)):
            self.enqueue_spec(key, reason="deploy")
        while self.spec_queue or self._drain_busy:
            await asyncio.sleep(0.05)

    async def reset(self, keep_visit_ctx: bool = False) -> Dict[str, Any]:
        """Reset the engine to cold state"""
        self.spec_queue.clear()
        deadline = time.perf_counter() + 2.0  # let in-flight warms finish; they would refill the cache
        while (self._warm_inflight or self._drain_busy) and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
        ok = await self.engine.reset_prefix_cache()
        self._warm_hash.clear()
        self.feeds.sessions.clear()
        if not keep_visit_ctx:
            self.feeds.visit_ctx.clear()
        self.prov_results.clear()
        self.route_prompts.clear()
        self.probe_favored.clear()
        self.last_call_key = None
        self.stats = {k: [] for k in self.stats}
        self.monitor.events.clear()
        for k in self.scheduler.stats:
            self.scheduler.stats[k] = 0
        return {"ok": bool(ok), "kv_reset": bool(ok), "visit_ctx_kept": keep_visit_ctx}

    # ------------------------------------------------------------------- call
    async def call_text(self, key: str, params: Dict[str, Any], sampling_params: Optional[SamplingParams] = None, sid: str = "") -> Tuple[str, Optional[float]]:
        """Serve one byllm call, returning raw generated text. The prompt is always
        rebuilt from the real params — a session's fed bindings replay in arrival
        order only while their values verify; warmed prefixes only make it faster,
        never different."""
        fn = self.side_rt.byllm_callsites[key]
        self.monitor.on_call_start(key)
        session = self.feeds.session(key, sid) if sid else None
        if session is not None:
            prompt = self._render(final_messages(fn, params, session))
            self.feeds.drop_session(key, sid)
        else:
            prompt = self._render(fn.build_full_prompt(params))
        sp = sampling_params or fn.sampler or SamplingParams(max_tokens=512)
        t0 = time.perf_counter()
        async with self.scheduler.real():  # busy mode: parked speculation stays parked
            final, ttft = await self._generate(prompt, sp, f"call-{uuid.uuid4().hex[:8]}", on_first_token=lambda: self.monitor.on_first_token(key))
        duration = time.perf_counter() - t0
        cached = getattr(final, "num_cached_tokens", None)
        self.stats["calls"].append({"key": key, "ttft": ttft, "duration": duration, "cached_tokens": cached})
        self.monitor.on_call_end(key, ttft, duration)
        text = final.outputs[0].text if final is not None and final.outputs else ""
        return text, ttft

    async def call(self, key: str, params: Dict[str, Any], sampling_params: Optional[SamplingParams] = None) -> Any:
        """call_text + server-side parse (used by in-process drivers; the interceptor
        parses on the Jac side instead, with the program's real classes)."""
        text, _ = await self.call_text(key, params, sampling_params)
        return self.side_rt.byllm_callsites[key].parse_response(text)

    def _match_visit_key(self, messages: List[Dict[str, Any]]) -> str:
        """Identify a visit-routing request by its prompt shape (route_visit builds its
        own messages, so the interceptor has no MTIR to derive a key from)."""
        if not messages or not isinstance(messages[0], dict):
            return ""
        sys_c = str(messages[0].get("content", "") or "")
        if not sys_c.startswith("You are routing a graph walker"):
            return ""
        user_c = str(messages[1].get("content", "") or "") if len(messages) > 1 else ""
        fallback = ""
        for key, fn in self.side_rt.byllm_callsites.items():
            if fn.decl.kind != "visit":
                continue
            fallback = fallback or key
            if fn.invariant_user_prefix and user_c.startswith(fn.invariant_user_prefix):
                return key
        return fallback

    async def generate_raw(self, 
                           messages: List[Dict[str, Any]], 
                           schema: Optional[Dict[str, Any]] = None, 
                           temperature: Optional[float] = None, 
                           max_tokens: Optional[int] = None, 
                           stop: Optional[List[str]] = None,
                           key: str = "") -> Tuple[str, Optional[float]]:
        """Serve one completion for a byllm-built prompt (the model_call_no_stream wire).

        byllm constructed the messages on the program side, so they are byte-identical
        to the native path — the warmed invariant prefix hits by construction. `key`
        (from the interceptor, or matched for visit routing) drives the monitor."""
        if not key:
            key = self._match_visit_key(messages)
        fn = None
        walker_fields: Optional[Dict[str, str]] = None
        if key:
            self.monitor.on_call_start(key)
            fn = self.side_rt.byllm_callsites.get(key)
            if fn is not None and len(messages) > 1:
                if fn.decl.kind == "visit":
                    # Server-side learning: a served route prompt carries the graph-derived
                    # stable zones — remember them (with the choice schema) so warms cover
                    # them from now on, and export them for prewarm after a restart.
                    ctx = extract_visit_ctx(str(self._flatten_content(messages[1].get("content")) or ""))
                    if ctx is not None:
                        self.feeds.set_visit_ctx(key, ctx[0], ctx[1], schema)
                else:
                    # reorder the byllm-rendered binding zone (first user message) into the best-matching session's arrival order
                    pending = self.feeds.sessions_for(key)
                    if pending:
                        user_text = str(self._flatten_content(messages[1].get("content")) or "")
                        new_text, matched, n = reorder_binding_zone(user_text, fn, pending)
                        if matched is not None and n > 0:
                            messages = list(messages)
                            messages[1] = {**messages[1], "content": new_text}
                            self.stats["reorders"].append({"key": key, "sid": matched.sid, "matched": n})
        prompt = self._render(messages)
        self._dump("serve", key or "generic", prompt)
        if key and fn is not None and fn.decl.kind == "visit":
            self.route_prompts[key] = prompt  # probe input: fork of this exact prompt is APC-cached
        kwargs: Dict[str, Any] = {"max_tokens": int(max_tokens or 512)}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if self.greedy:
            kwargs["temperature"] = 0.0  # experiment mode: deterministic routing/decoding
        if stop:
            kwargs["stop"] = list(stop) if isinstance(stop, (list, tuple)) else [str(stop)]
        inner = schema.get("json_schema", {}).get("schema") if isinstance(schema, dict) else None
        if inner:
            kwargs["structured_outputs"] = StructuredOutputsParams(json=inner)
        rid = f"gen-{uuid.uuid4().hex[:8]}"
        if key and fn is not None and fn.decl.kind == "visit" and len(messages) > 1:
            # First provenance checkpoint's inputs: the walker state riding this
            # route request is the successors' future bindings.
            walker_fields = extract_walker_fields(str(self._flatten_content(messages[1].get("content")) or ""))
        cb = (lambda: self.monitor.on_first_token(key, rid, walker_fields)) if key else None
        t0 = time.perf_counter()
        async with self.scheduler.real():  # busy mode: parked speculation stays parked
            final, ttft = await self._generate(prompt, SamplingParams(**kwargs), rid, on_first_token=cb)
        duration = time.perf_counter() - t0
        cached = getattr(final, "num_cached_tokens", None)
        print(f"[DEBUG] TTFT for subagent {key} = {ttft:.3f}s")
        self.stats["calls"].append({"key": key or "(generic)", "ttft": ttft, "duration": duration, "cached_tokens": cached, "prompt_head": prompt[:200]})
        text = final.outputs[0].text if final is not None and final.outputs else ""
        if key:
            # Second checkpoint's input: this call's parsed result (str returns and
            # finish_tool payloads; other shapes aren't feedable as bindings yet).
            result_repr: Optional[str] = None
            if fn is not None and fn.decl.kind != "visit":
                if fn.decl.tools:
                    found, val = extract_finish_output(text)
                    if found:
                        result_repr = repr(val)
                elif fn.decl.return_type in ("", "str"):
                    result_repr = repr(text)
            self.monitor.on_call_end(key, ttft, duration, rid, result_repr)
        return text, ttft

    async def shutdown(self) -> None:
        self.engine.shutdown()


# ---------------------------------------------------------------------- http
def _build_app(server: GuardServer):
    """FastAPI app exposing the interceptor wire protocol: /call, /generate, /health."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "callsites": list(server.side_rt.byllm_callsites), "topo": server.side_rt.callsites_topo}

    @app.get("/stats")
    async def stats() -> Dict[str, Any]:
        ev = server.monitor.events
        t0 = ev[0]["t"] if ev else 0.0
        sched = {**server.scheduler.stats, "busy_budget": server.scheduler.busy_budget, "profile": server.scheduler.profile_report}
        return {"stats": server.stats, "scheduler": sched, "events": [{**e, "t": round(e["t"] - t0, 3)} for e in ev]}

    @app.post("/call")
    async def call_ep(request: Request):
        payload = await request.json()
        key = payload.get("key", "")
        if key not in server.side_rt.byllm_callsites:
            return JSONResponse({"error": f"unknown callsite key: {key}"}, status_code=404)
        params = {k: RawRepr(v) if isinstance(v, str) else v for k, v in (payload.get("params") or {}).items()}
        text, ttft = await server.call_text(key, params, sid=payload.get("sid") or "")
        return {"text": text, "ttft": ttft}

    @app.post("/feed")
    async def feed_ep(request: Request):
        payload = await request.json()
        key = payload.get("key", "")
        if key not in server.side_rt.byllm_callsites:
            return JSONResponse({"error": f"unknown callsite key: {key}"}, status_code=404)
        await server.feed_param(key, payload.get("sid") or "", payload.get("param", ""), payload.get("value", ""))
        return {"ok": True}

    @app.post("/feed_visit")
    async def feed_visit_ep(request: Request):
        payload = await request.json()
        key = payload.get("key", "")
        if key not in server.side_rt.byllm_callsites:
            return JSONResponse({"error": f"unknown callsite key: {key}"}, status_code=404)
        await server.feed_visit(key, payload.get("here") or "", payload.get("candidates") or "", payload.get("schema"))
        return {"ok": True}

    @app.get("/visit_ctx")
    async def visit_ctx_ep() -> Dict[str, Any]:
        return {"visit_ctx": server.feeds.visit_ctx}

    @app.post("/reset")
    async def reset_ep(keep_visit_ctx: bool = False):
        """Cold-start reset between runs"""
        return await server.reset(keep_visit_ctx=keep_visit_ctx)

    @app.post("/generate")
    async def generate_ep(request: Request):
        payload = await request.json()
        text, ttft = await server.generate_raw(payload.get("messages") or [], schema=payload.get("schema"), temperature=payload.get("temperature"), max_tokens=payload.get("max_tokens"), stop=payload.get("stop"), key=payload.get("key") or "")
        return {"text": text, "ttft": ttft}

    return app

async def _prewarm_vllm(server: GuardServer):
    prompt = server._render([{"role":"user", "content":" ".join(f"{i:04x}" for i in range(512))}])
    sp = SamplingParams(max_tokens=16, temperature=0)
    async for _ in server.engine.generate(prompt, sp, f"prewarm-{uuid.uuid4().hex[:8]}"):
        pass
    assert await server.engine.reset_prefix_cache()
    print(f"[guard] vLLM prewarmed with 512-token prompt")
    
async def _serve(server: GuardServer, host: str, port: int, deploy_warm: bool = True, calibrate: bool = True) -> None:
    import uvicorn
    await _prewarm_vllm(server)
    if calibrate:
        await server.scheduler.calibrate()
    server.start_drainer()
    if deploy_warm:
        await server.warm_all()
    print(f"[guard] {len(server.side_rt.byllm_callsites)} call site(s){' warmed' if deploy_warm else ' (cold)'}; serving on {host}:{port}")
    config = uvicorn.Config(_build_app(server), host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()

def main() -> None:
    ap = argparse.ArgumentParser(description="Side-runtime guard server: invariant warm + topology monitor")
    ap.add_argument("file", help="entry .jac file")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--no-type-check", action="store_true")
    ap.add_argument("--deploy-time-prewarm", action="store_true", help="warm all call sites on startup (default: rely on the monitor's speculative warms only)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--no-speculate", action="store_true", help="disable the monitor's topology-driven warming (vanilla APC baseline)")
    ap.add_argument("--no-prov-feed", action="store_true", help="disable tier-3 provenance feeds only (invariant warms stay on; isolates tier-3's contribution)")
    ap.add_argument("--probe", action="store_true", help="enable the visit-by choice-distribution probe (fork at route first token, skip thinking, favor the predicted candidate in the drainer)")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the TBT interference sweep (busy budget stays 0: speculation runs in idle windows only)")
    ap.add_argument("--greedy", action="store_true", help="force temperature=0 on served requests (deterministic A/B experiments)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8964)
    args = ap.parse_args()
    server = GuardServer(args.file, args.model,
                         max_model_len=args.max_model_len, 
                         gpu_memory_utilization=args.gpu_mem, 
                         type_check=not args.no_type_check,
                         enable_probe=args.probe)
    server.speculate = not args.no_speculate
    server.prov_feed = not args.no_prov_feed
    server.greedy = args.greedy

    async def run() -> None:
        try:
            await _serve(server, args.host, args.port, deploy_warm=args.deploy_time_prewarm, calibrate=not args.no_calibrate)
        finally:
            await server.shutdown()
            print("[guard] monitor event log:")
            server.monitor.dump()

    asyncio.run(run())


if __name__ == "__main__":
    main()
