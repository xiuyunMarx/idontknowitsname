import asyncio
import json
import os
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from instance_engine.model import Engine
from serve.planner import fifo_order
from static_pass import link
from static_pass.instance import ProgramInstance
from static_pass.primitives import ByLLMFunc, Program, RequestHandle


def _bare_schema(schema: dict) -> dict:
    """The JSON schema itself, out of an OpenAI-style response_format wrapper if given one."""
    if schema.get("type") == "json_schema" and isinstance(schema.get("json_schema"), dict):
        inner = schema["json_schema"]
        return inner.get("schema", inner)
    return schema


class Controller:
    """Multi-tenant FaaS layer: clients queue RequestHandles per model; the planner
    orders the models to put on the GPU (FIFO over each instance's predicted chain),
    loads the next one with the waiting prompts prefilled under the weight stream,
    and releases the bucket. `gpu_slots` engines may be resident at once;
    `host_slots` may keep a pinned host copy, the rest fall back to SSD."""

    def __init__(self, gpu_slots: int = 1, host_slots: int = 8, speculate: bool = True):
        self.engine_pool: Dict[str, Engine] = {}
        self.engine_kwargs: Dict[str, dict] = {}     # model -> Engine.create kwargs
        self.program_template: Dict[str, Program] = {}
        self.instances: Dict[Tuple[str, int], ProgramInstance] = {}
        self.pending: Dict[str, List[RequestHandle]] = {}
        self.gpu_slots, self.host_slots, self.speculate = gpu_slots, host_slots, speculate
        self.last_used: Dict[str, float] = {}
        self.active: Dict[str, int] = {}             # model -> dispatched requests not yet finished
        self.running: "set[RequestHandle]" = set()   # handles inside engine.generate right now
        self.stats: list = []   # (callsite_key, kind, engine.last) per real request
        self.loads: list = []   # (model, from_tier, seconds, prefilled) per load
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # ---- setup -----------------------------------------------------------------

    async def add_engine(self, model_name: str, **kwargs) -> Engine:
        """Create (resident) and immediately park on host so GPU slots stay free."""
        if model_name not in self.engine_pool:
            self.engine_kwargs[model_name] = kwargs
            eng = await Engine.create(model_name=model_name, **{"max_num_batched_tokens": 8192, "max_model_len": 8192, **kwargs})
            await eng.warmup()
            # await eng.offload()
            await eng.kill() 
            self.engine_pool[model_name] = eng
        return self.engine_pool[model_name]

    def register_program(self, path: str) -> Program:
        name = os.path.splitext(os.path.basename(path))[0]  # InterceptorLLM's program_name
        self.program_template[name] = Program(name=name).build_program(path)
        return self.program_template[name]

    async def listen(self, host: str, port: int) -> asyncio.AbstractServer:
        """One port for every program; connections of one process share a ProgramInstance."""
        self._task = self._task or asyncio.create_task(self.run())

        async def on_client(conn: link.Connection, hello: dict) -> None:
            name = str(hello.get("program_name") or "")
            program = self.program_template.get(name) or (
                next(iter(self.program_template.values())) if len(self.program_template) == 1 else None)
            if program is None:
                conn.send({"type": "error", "error": f"unknown program {name!r}"})
                return
            key = (program.name, int(hello.get("pid", -1)))
            inst = self.instances.get(key) or self.instances.setdefault(key, ProgramInstance(program, self, key[1]))
            try:
                await inst.serve(conn, hello)
            finally:
                if inst.connections == 0:
                    self.instances.pop(key, None)
                    self._wake.set()
        return await link.listen(host, port, on_client)

    async def control_listen(self, host: str, port: int) -> asyncio.AbstractServer:
        """Runtime control: one JSON line in, one JSON line out.
        {"set": {"speculate": bool, "gpu_slots": n, "host_slots": n}}
        {"reset": true}   park every engine on host, clear prefix caches and counters
        {"stats": true}   loads / requests since the last reset
        {"dump": true}"""
        async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                req = json.loads(await reader.readline())
                writer.write((json.dumps(await self.control(req)) + "\n").encode())
                await writer.drain()
            except Exception as e:
                writer.write((json.dumps({"error": repr(e)}) + "\n").encode())
            finally:
                writer.close()
        return await asyncio.start_server(serve, host, port)

    async def control(self, req: dict) -> dict:
        if "set" in req:
            for k, v in req["set"].items():
                if k not in ("speculate", "gpu_slots", "host_slots"):
                    raise ValueError(f"unknown setting {k!r}")
                setattr(self, k, v)
            self._wake.set()
        if req.get("reset"):
            if self.pending or any(e.busy or self.active.get(e.name) for e in self.engine_pool.values()):
                raise RuntimeError("reset while requests are in flight")
            for e in self.engine_pool.values():
                if e.resident:
                    await e.offload()
                await e.engine.reset_prefix_cache()
            self.loads.clear(); self.stats.clear(); self.last_used.clear()
        out = {"speculate": self.speculate, "gpu_slots": self.gpu_slots, "host_slots": self.host_slots}
        if req.get("stats"):
            ttft = [st["ttft_ms"] for _, _, st in self.stats]
            out["stats"] = {"loads": len(self.loads), "load_s": round(sum(l[2] for l in self.loads), 2),
                            "load_order": [(m.split("/")[-1], t) for m, t, _, _ in self.loads],
                            "requests": len(ttft), "ttft_mean_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
                            "cached_frac": round(sum(st["cached_tokens"] for _, _, st in self.stats)
                                                 / max(1, sum(st["prompt_tokens"] for _, _, st in self.stats)), 3)}
        if req.get("dump"):
            out["dump"] = self.dump()
        return out

    # ---- requests ----------------------------------------------------------------

    async def submit(self, handle: RequestHandle) -> None:
        self.pending.setdefault(handle.model_name, []).append(handle)
        self._dispatch()  # a resident model serves at once, whatever the planner is awaiting
        self._wake.set()

    async def _serve(self, handle: RequestHandle) -> None:
        engine = self.engine_pool[handle.model_name]
        try:
            p = handle.call_params
            sp = SamplingParams(
                temperature=p.get("temperature", 0.7), max_tokens=p.get("max_tokens"), stop=p.get("stop"),
                structured_outputs=StructuredOutputsParams(json=_bare_schema(handle.schema)) if handle.schema else None)

            def progress(first_token_at: float, out_tokens: int) -> None:
                handle.first_token_at, handle.out_tokens = first_token_at, out_tokens
            self.running.add(handle)
            handle.text = await engine.generate(engine.render(handle.messages),
                                                f"{handle.kind}-{uuid.uuid4().hex}", sp, handle.cache_salt, progress)
            self.stats.append((handle.callsite_key, handle.kind, {**engine.last, "output_tokens": handle.out_tokens}))
        except Exception as e:
            handle.error = repr(e)
            traceback.print_exc()
        finally:
            self.running.discard(handle)
            self.active[handle.model_name] -= 1
            self.last_used[handle.model_name] = time.perf_counter()
            handle.done.set()
            self._wake.set()
        if self.speculate:
            asyncio.create_task(self._speculate(handle))

    def _dispatch(self) -> None:
        """Release every pending handle whose model is resident."""
        for model, handles in list(self.pending.items()):
            eng = self.engine_pool.get(model)
            if eng is not None and eng.resident and not eng.transitioning:
                del self.pending[model]
                self.active[model] = self.active.get(model, 0) + len(handles)  # busy before the tasks start
                for h in handles:
                    asyncio.create_task(self._serve(h))

    # ---- planning ----------------------------------------------------------------

    def sequences(self) -> List[List[str]]:
        """Per live instance: the model it waits for or runs on, then the models of
        its predicted path to the end of the program, one per callsite."""
        out = []
        for inst in self.instances.values():
            seq: List[str] = []
            waiting = [h for hs in self.pending.values() for h in hs if h.instance is inst]
            running = [h for h in self.running if h.instance is inst]
            open_calls = [c for c in inst.calls.values() if not c.finished]
            if waiting:
                seq.append(waiting[0].model_name)
            elif running:
                seq.append(running[0].model_name)
            elif open_calls:  # a tool loop between turns keeps its model
                seq.append(open_calls[0].model_name)
            seq.extend(m for _, m in inst.predicted)
            if seq:
                out.append(seq)
        return out

    def plan(self) -> List[str]:
        return fifo_order(self.sequences())

    def _pinned(self) -> "set[str]":
        """Models some instance is in the middle of a call on (a tool loop between
        turns): its next turn is imminent, so evicting them only forces a reload."""
        return {c.model_name for inst in self.instances.values() for c in inst.calls.values() if not c.finished}

    def _victim(self, order: List[str]) -> Optional[Engine]:
        """An idle, unpinned resident engine to evict: the one the plan needs latest (LRU on ties)."""
        pinned = self._pinned()
        idle = [e for e in self.engine_pool.values()
                if e.resident and not e.serving and not self.active.get(e.name) and e.name not in pinned]
        rank = {m: i for i, m in enumerate(order)}
        idle.sort(key=lambda e: (-rank.get(e.name, len(order)), self.last_used.get(e.name, 0.0)))
        return idle[0] if idle else None

    async def run(self) -> None:
        """Planner loop: dispatch what can run, then load the next model the plan
        asks for whenever a GPU slot is free or an idle engine can make one."""
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                await self._plan_step()
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(1.0)
                self._wake.set()  # retry rather than wait for an event that may never come

    async def _plan_step(self) -> None:
        self._dispatch()
        while True:
            order = self.plan()
            target = next((m for m in order if not (m in self.engine_pool and self.engine_pool[m].resident)), None)
            if target is None:
                break
            if sum(e.resident for e in self.engine_pool.values()) >= self.gpu_slots:
                victim = self._victim(order)
                if victim is None:
                    break  # everything resident is busy; a completion will wake us
                await victim.offload()
                await self._trim_host()
            await self._load(target)
            self._dispatch()

    async def _load(self, model: str) -> None:
        """Bring `model` to the GPU, prefilling under the weight stream the waiting
        prompts and the known prefixes of every predicted callsite on this model."""
        eng = self.engine_pool.get(model)
        if eng is None:
            eng = await self.add_engine(model)
        prompts = [eng._with_salt(eng.render(h.messages), h.cache_salt) for h in self.pending.get(model, [])]
        for inst in self.instances.values():
            if any(h.instance is inst for hs in self.pending.values() for h in hs):
                continue
            for key, m in inst.predicted:
                if m != model:
                    continue
                full = self._spec_prompt(inst, key, eng)
                if full is not None:
                    prompts.append(eng._with_salt(full, inst.salt()))
        tier = eng.tier
        t = time.perf_counter()
        await eng.load(prefill=prompts)
        secs = time.perf_counter() - t
        self.loads.append((model, tier, secs, len(prompts)))
        print(f"[load] {model} from={tier} {secs * 1000:.0f}ms prefilled={len(prompts)}", flush=True)

    async def _trim_host(self) -> None:
        """Keep at most `host_slots` pinned copies; the least recently used go to SSD."""
        hosted = [e for e in self.engine_pool.values() if e.tier == "host"]
        hosted.sort(key=lambda e: self.last_used.get(e.name, 0.0))
        for e in hosted[:max(0, len(hosted) - self.host_slots)]:
            await e.kill()

    # ---- speculation --------------------------------------------------------------

    def _spec_prompt(self, inst: ProgramInstance, key: str, eng: Engine) -> Optional[str]:
        """The prefix of callsite `key` known now: full prompt if every param is bound,
        else the static part plus bound params (a partial user turn)."""
        site = inst.sites.get(key)
        if not isinstance(site, ByLLMFunc):
            return None
        ready = inst.program.ready_params(key, inst.done)
        user = site.render_full({})
        full = eng.render([{"role": "system", "content": site.render_system()}, {"role": "user", "content": user}])
        if not all(r.ready and not r.needs_state for r in ready.values()):
            full = full[:full.rindex(user) + len(user)]
        return full

    async def _speculate(self, handle: RequestHandle) -> None:
        """Warm each successor callsite whose model is resident (loading is the planner's job)."""
        inst = handle.instance
        try:
            for key in inst.program.successor_readiness(handle.callsite_key, inst.done):
                model = inst.program.model_of(inst.sites[key], handle.model_name)
                eng = self.engine_pool.get(model)
                if eng is None or not eng.speculable:
                    continue
                full = self._spec_prompt(inst, key, eng)
                if full is not None and eng.spec_allowance() >= len(eng.tokenize(full)):
                    await eng.prefill(full, f"spec-{uuid.uuid4().hex}", cache_salt=handle.cache_salt)
        except Exception:
            traceback.print_exc()
        finally:
            self._wake.set()  # the engine may have become idle (evictable)

    def dump(self) -> str:
        """One-line-per-item state for debugging (SIGUSR1 in start_server)."""
        lines = [f"pending: {{ {', '.join(f'{m.split('/')[-1]}:{len(h)}' for m, h in self.pending.items())} }}",
                 f"active: {self.active}",
                 "instances: " + "; ".join(f"{k[0]}#{k[1]} calls={sum(not c.finished for c in i.calls.values())} predicted={[m.split('/')[-1] for _, m in i.predicted]}"
                                           for k, i in self.instances.items())]
        for m, e in self.engine_pool.items():
            lines.append(f"engine {m.split('/')[-1]:<22} tier={e.tier:<4} busy={e.busy} prefill={e._inflight_prefill} "
                         f"decode={e._inflight_decode} spec={len(e._spec_tasks)} "
                         f"unfinished={e.engine.output_processor.has_unfinished_requests()}")
        for t in asyncio.all_tasks():
            frames = t.get_stack()[-3:]
            lines.append("task: " + " <- ".join(f"{f.f_code.co_name}:{f.f_lineno}" for f in frames))
        return "\n".join(lines)
