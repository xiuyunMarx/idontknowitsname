from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `static`/`runtime` importable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.async_llm import AsyncLLM

from runtime.incremental_feed import FeedStore, eval_provenance, extract_visit_ctx, extract_walker_fields, final_messages, partial_user_prompt, reorder_binding_zone
from static.static_parser import extract_finish_output
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
        self._tasks: set = set()
        self._pending_ret: set = set()  # consumers awaiting a decode shadow to be fed in

    def _log(self, kind: str, key: str, **extra: Any) -> None:
        self.events.append({"t": time.perf_counter(), "kind": kind, "key": key, **extra})

    def on_call_start(self, key: str) -> None:
        self._log("call_start", key)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def on_first_token(self, key: str, rid: str = "", walker_fields: Optional[Dict[str, str]] = None) -> None:
        """Speculate once the current call's first token is out: its TTFT-critical
        window is over, and the remaining decode (~95% of the call) is still ample
        overlap for warming successors without interfering with prefill. Besides the
        invariant warm, this is the first provenance checkpoint: successor params
        sourced from state visible NOW (walker fields, constants) get fed. Field
        feeds stay topology-adjacent on purpose — walker fields are MUTABLE state,
        and adjacency doubles as a freshness heuristic (ret feeds have no such
        restriction; see on_call_end)."""
        if not self.server.speculate:
            return
        for succ in self.server.side_rt.callsites_topo.get(key, []):
            self._log("speculate", succ, after=key)
            self._spawn(self.server.warm(succ, reason=f"spec:{key}"))
            if self.server.prov_feed:
                self._spawn(self.server.provenance_feed(succ, walker_fields=walker_fields))
        # Flush ret feeds deferred at earlier call ends: THIS call's decode is the
        # shadow they run in. Submitting them at call_end raced the immediately
        # following call's prefill (measured: ~34ms TTFT pollution per producer in
        # tight sequential flows); here the current call's TTFT-critical window is
        # already over. Feeds whose consumer is this very call are harmless no-ops
        # (their prefix is a subset of the already-cached served prompt).
        if self.server.prov_feed and self._pending_ret:
            pending, self._pending_ret = self._pending_ret, set()
            for consumer in pending:
                self._log("ret_feed", consumer, shadow=key)
                self._spawn(self.server.provenance_feed(consumer, results=self.server.prov_results))

    def on_call_end(self, key: str, ttft: Optional[float], duration: float, rid: str = "", result_repr: Optional[str] = None) -> None:
        """Second provenance checkpoint: the call's parsed result is now known.
        DATAFLOW firing, not topology firing: consumers of ret(this call) come from
        the inverse provenance index, however many calls downstream they sit —
        `a=f(); b=g(); h(a,b)` feeds h.a the moment f ends. Safe at any distance:
        ret values are SSA-like (the variable never mutates after binding). Results
        accumulate in a store so a consumer fed at different times (h.a at f's end,
        h.b at g's end) evaluates against everything observed so far."""
        self._log("call_end", key, ttft=ttft, duration=duration)
        if result_repr is None or not self.server.speculate or not self.server.prov_feed:
            return
        self.server.prov_results[key] = result_repr
        # Record consumers but DEFER the feeds to the next call's first token (see
        # on_first_token): a feed submitted right here lands in the same scheduling
        # window as the next real call's prefill and pollutes its TTFT.
        for ckey, _param in self.server.side_rt.ret_consumers.get(key, []):
            self._pending_ret.add(ckey)

    def on_warm(self, key: str, duration: float, reason: str) -> None:
        self._log("warm_done", key, duration=duration, reason=reason)

    def dump(self) -> None:
        t0 = self.events[0]["t"] if self.events else 0.0
        for e in self.events:
            extra = {k: v for k, v in e.items() if k not in ("t", "kind", "key")}
            print(f"  [{e['t'] - t0:8.3f}s] {e['kind']:<10} {e['key']}  {extra if extra else ''}")


class GuardServer:
    """Owns the AsyncLLM engine and the compiled SideRuntime of one Jac program."""

    def __init__(self, jac_path: str, model: str, *, max_model_len: int = 4096, gpu_memory_utilization: float = 0.45, enforce_eager: bool = True, type_check: bool = True):
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
        self.stats: Dict[str, List[Dict[str, Any]]] = {"warms": [], "calls": [], "feeds": [], "reorders": []}

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

    # ------------------------------------------------------------------- warm
    def _warm_user_prefix(self, key: str, fn: Any) -> Tuple[str, Optional[Any]]:
        """The longest currently-known stable user prefix for a call site, plus the
        decoding-constraint schema whose grammar the warm should precompile."""
        so = getattr(fn.sampler, "structured_outputs", None) if fn.sampler is not None else None
        if fn.decl.kind != "visit":
            return fn.invariant_user_prefix, so
        ctx = self.feeds.get_visit_ctx(key)
        if not ctx:
            return fn.invariant_user_prefix, so
        schema = ctx.get("schema")
        inner = schema.get("json_schema", {}).get("schema") if isinstance(schema, dict) else None
        if inner:
            so = StructuredOutputsParams(json=inner)
        return fn.visit_stable_prefix(ctx.get("here", ""), ctx.get("candidates", "")), so

    async def warm(self, key: str, reason: str = "deploy") -> None:
        """Prefill this call site's stable prefix: the compile-time invariant, extended by the graph-derived visit context when one has been fed/learned. """
        fn = self.side_rt.byllm_callsites.get(key)
        if fn is None or key in self._warm_inflight:
            return
        user_prefix, so = self._warm_user_prefix(key, fn)
        messages = [{"role": "system", "content": fn.invariant_system}, {"role": "user", "content": user_prefix}]
        prompt = self._render(messages)
        sp = SamplingParams(max_tokens=1, temperature=0.0, structured_outputs=so)
        self._warm_inflight.add(key)
        t0 = time.perf_counter()
        try:
            # Lower priority (higher value): a speculative warm must yield to any
            # real call's prefill instead of competing with it for engine steps.
            await self._generate(prompt, sp, f"warm-{uuid.uuid4().hex[:8]}", priority=1)
        finally:
            self._warm_inflight.discard(key)
        dur = time.perf_counter() - t0
        self.stats["warms"].append({"key": key, "duration": dur, "reason": reason, "prompt_head": prompt[:200]})
        self.monitor.on_warm(key, dur, reason)

    async def feed_param(self, key: str, sid: str, name: str, value_repr: str) -> None:
        """Per-request incremental feed: record one binding and warm the extended
        arrival-order prefix (invariant + fed lines) for the upcoming call."""
        fn = self.side_rt.byllm_callsites.get(key)
        if fn is None:
            return
        session, changed = self.feeds.feed_param(key, sid, name, value_repr)
        if not changed:
            return
        prompt = self._render([{"role": "system", "content": fn.invariant_system}, {"role": "user", "content": partial_user_prompt(fn, session)}])
        t0 = time.perf_counter()
        await self._generate(prompt, SamplingParams(max_tokens=1, temperature=0.0), f"feed-{uuid.uuid4().hex[:8]}", priority=1)
        dur = time.perf_counter() - t0
        self.stats["feeds"].append({"key": key, "sid": sid, "param": name, "duration": dur})

    async def feed_visit(self, key: str, here: str, candidates: str, schema: Optional[Any] = None) -> None:
        """Quasi-static feed: the graph-derived here/candidates zones of a visit site
        (and its choice schema, so the warm also precompiles the grammar)."""
        if self.feeds.set_visit_ctx(key, here, candidates, schema):
            await self.warm(key, reason="feed:visit")

    async def provenance_feed(self, succ: str, walker_fields: Optional[Dict[str, str]] = None, results: Optional[Dict[str, str]] = None) -> None:
        """Feed every parameter of `succ` whose provenance spec is evaluable from
        the observed state. Params are fed in declaration order (so when everything
        fires at once the reorder is an identity). All provenance feeds for one
        consumer share ONE session (sid="prov") — a consumer fed from multiple
        events (h.a at f's end, h.b at g's end) accumulates arrival-ordered lines
        in a single replayable prefix instead of fragmenting across sessions.
        Re-fed params: same value is a no-op, a changed value truncates from that
        line (PrefillSession.feed), so cross-request staleness self-heals."""
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
                await self.feed_param(succ, "prov", p["name"], val)

    async def warm_all(self) -> None:
        """Deploy-time warm of every call site's invariant."""
        await asyncio.gather(*(self.warm(k) for k in self.side_rt.byllm_callsites))

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
        final, ttft = await self._generate(prompt, SamplingParams(**kwargs), rid, on_first_token=cb)
        duration = time.perf_counter() - t0
        cached = getattr(final, "num_cached_tokens", None)
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
        return {"stats": server.stats, "events": [{**e, "t": round(e["t"] - t0, 3)} for e in ev]}

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

    @app.post("/generate")
    async def generate_ep(request: Request):
        payload = await request.json()
        text, ttft = await server.generate_raw(payload.get("messages") or [], schema=payload.get("schema"), temperature=payload.get("temperature"), max_tokens=payload.get("max_tokens"), stop=payload.get("stop"), key=payload.get("key") or "")
        return {"text": text, "ttft": ttft}

    return app


async def _serve(server: GuardServer, host: str, port: int, deploy_warm: bool = True) -> None:
    import uvicorn
    if deploy_warm:
        await server.warm_all()
    print(f"[guard] {len(server.side_rt.byllm_callsites)} call site(s){' warmed' if deploy_warm else ' (cold)'}; serving on {host}:{port}")
    config = uvicorn.Config(_build_app(server), host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()

def main() -> None:
    ap = argparse.ArgumentParser(description="Side-runtime guard server: invariant warm + topology monitor")
    ap.add_argument("file", help="entry .jac file")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-type-check", action="store_true")
    ap.add_argument("--deploy-time-prewarm", action="store_true", help="warm all call sites on startup (default: rely on the monitor's speculative warms only)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--no-speculate", action="store_true", help="disable the monitor's topology-driven warming (vanilla APC baseline)")
    ap.add_argument("--no-prov-feed", action="store_true", help="disable tier-3 provenance feeds only (invariant warms stay on; isolates tier-3's contribution)")
    ap.add_argument("--greedy", action="store_true", help="force temperature=0 on served requests (deterministic A/B experiments)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8964)
    args = ap.parse_args()
    server = GuardServer(args.file, args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, type_check=not args.no_type_check)
    server.speculate = not args.no_speculate
    server.prov_feed = not args.no_prov_feed
    server.greedy = args.greedy

    async def run() -> None:
        try:
            await _serve(server, args.host, args.port, deploy_warm=args.deploy_time_prewarm)
        finally:
            await server.shutdown()
            print("[guard] monitor event log:")
            server.monitor.dump()

    asyncio.run(run())


if __name__ == "__main__":
    main()
