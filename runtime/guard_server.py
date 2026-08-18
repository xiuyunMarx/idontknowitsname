"""GuardServer: the side-runtime server owning an AsyncLLM engine, plus the Monitor.

Phase 1 (this file): invariant-only proactive prefill.
  - warm(key)/warm_all(): prefill a call site's invariant prefix into the KV cache.
  - call(key, params): the entrypoint the future byllm interceptor routes real calls
    to — rebuilds the full prompt from real params (prefill is advisory; correctness
    never depends on what was warmed), generates on the shared AsyncLLM engine, and
    returns the parsed typed value.
  - Monitor: when a call STARTS, speculatively warms the invariants of its
    may-happen-next successors (callsites_topo) so they prefill on idle compute
    while the current call decodes.

No incremental parameter feeding yet — that is the next phase (PrefillSession).
"""

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

    def _log(self, kind: str, key: str, **extra: Any) -> None:
        self.events.append({"t": time.perf_counter(), "kind": kind, "key": key, **extra})

    def on_call_start(self, key: str) -> None:
        self._log("call_start", key)

    def on_first_token(self, key: str) -> None:
        """Speculate once the current call's first token is out: its TTFT-critical
        window is over, and the remaining decode (~95% of the call) is still ample
        overlap for warming successors without interfering with prefill."""
        if not self.server.speculate:
            return
        for succ in self.server.side_rt.callsites_topo.get(key, []):
            self._log("speculate", succ, after=key)
            task = asyncio.create_task(self.server.warm(succ, reason=f"spec:{key}"))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def on_call_end(self, key: str, ttft: Optional[float], duration: float) -> None:
        self._log("call_end", key, ttft=ttft, duration=duration)

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
        self._warm_inflight: set = set()
        self.stats: Dict[str, List[Dict[str, Any]]] = {"warms": [], "calls": []}

    # ------------------------------------------------------------------ engine
    def _render(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
    async def warm(self, key: str, reason: str = "deploy") -> None:
        """Prefill this call site's invariant prefix. Idempotent-cheap: a warm on an already-cached prefix costs one scheduling round + one token."""
        fn = self.side_rt.byllm_callsites.get(key)
        if fn is None or key in self._warm_inflight:
            return
        messages = [{"role": "system", "content": fn.invariant_system}, {"role": "user", "content": fn.invariant_user_prefix}]
        prompt = self._render(messages)
        # The output schema is compile-time constant too
        so = getattr(fn.sampler, "structured_outputs", None) if fn.sampler is not None else None
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
        self.stats["warms"].append({"key": key, "duration": dur, "reason": reason})
        self.monitor.on_warm(key, dur, reason)

    async def warm_all(self) -> None:
        """Deploy-time warm of every call site's invariant."""
        await asyncio.gather(*(self.warm(k) for k in self.side_rt.byllm_callsites))

    # ------------------------------------------------------------------- call
    async def call_text(self, key: str, params: Dict[str, Any], sampling_params: Optional[SamplingParams] = None) -> Tuple[str, Optional[float]]:
        """Serve one byllm call, returning raw generated text. The prompt is always
        rebuilt from the real params - warmed prefixes only make it faster, never different."""
        fn = self.side_rt.byllm_callsites[key]
        self.monitor.on_call_start(key)
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

    async def generate_raw(self, messages: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None, temperature: Optional[float] = None, max_tokens: int = 128) -> Tuple[str, Optional[float]]:
        """Generic generation for calls without a mapped call site (e.g. visit routing):
        no warm mapping, but same engine, with the typed-output schema as a constraint."""
        prompt = self._render(messages)
        kwargs: Dict[str, Any] = {"max_tokens": int(max_tokens)}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        inner = schema.get("json_schema", {}).get("schema") if isinstance(schema, dict) else None
        if inner:
            kwargs["structured_outputs"] = StructuredOutputsParams(json=inner)
        final, ttft = await self._generate(prompt, SamplingParams(**kwargs), f"gen-{uuid.uuid4().hex[:8]}")
        text = final.outputs[0].text if final is not None and final.outputs else ""
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
        text, ttft = await server.call_text(key, params)
        return {"text": text, "ttft": ttft}

    @app.post("/generate")
    async def generate_ep(request: Request):
        payload = await request.json()
        text, ttft = await server.generate_raw(payload.get("messages") or [], schema=payload.get("schema"), temperature=payload.get("temperature"), max_tokens=payload.get("max_tokens") or 128)
        return {"text": text, "ttft": ttft}

    return app


async def _serve(server: GuardServer, host: str, port: int, deploy_warm: bool = True) -> None:
    import uvicorn
    if deploy_warm:
        await server.warm_all()
    print(f"[guard] {len(server.side_rt.byllm_callsites)} call site(s){' warmed' if deploy_warm else ' (cold)'}; serving on {host}:{port}")
    config = uvicorn.Config(_build_app(server), host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()


# ---------------------------------------------------------------------- demo
def _sample_params(fn) -> Dict[str, Any]:
    samples = {"text": "My GPU catches fire every time I run the training job!!", "message": "How do walkers traverse a graph in Jac?", "query": "walker syntax"}
    out: Dict[str, Any] = {}
    for p in fn.decl.params:
        base = p["type"].split("[", 1)[0]
        if p["name"] in samples:
            out[p["name"]] = samples[p["name"]]
        elif base == "str":
            out[p["name"]] = "the quick brown fox jumps over the lazy dog"
        elif base == "int":
            out[p["name"]] = 3
        elif base == "float":
            out[p["name"]] = 1.0
        elif base == "bool":
            out[p["name"]] = True
        elif base in ("list", "set", "tuple"):
            out[p["name"]] = []
        elif base == "dict":
            out[p["name"]] = {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-runtime guard server: invariant warm + topology monitor")
    ap.add_argument("file", help="entry .jac file")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no-type-check", action="store_true")
    ap.add_argument("--no-deploy-warm", action="store_true", help="skip warm_all; rely on the monitor's speculative warms only")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--no-speculate", action="store_true", help="disable the monitor's topology-driven warming (vanilla APC baseline)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8964)
    args = ap.parse_args()
    server = GuardServer(args.file, args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, type_check=not args.no_type_check)
    server.speculate = not args.no_speculate

    async def run() -> None:
        try:
            await _serve(server, args.host, args.port, deploy_warm=not args.no_deploy_warm)
        finally:
            await server.shutdown()
            print("[guard] monitor event log:")
            server.monitor.dump()

    asyncio.run(run())


if __name__ == "__main__":
    main()
