import asyncio
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from instance_engine.model import Engine
from serve.planner import fifo_order, scs_order
from static_pass import link
from static_pass.instance import ProgramInstance
from static_pass.primitives import ByLLMFunc, Program, RequestHandle

DEFAULT_LOAD_COST = {"gpu": 0.0, "host": 1.0, "ssd": 3.0}  # relative, until measured


class Controller:
    """Multi-tenant FaaS layer: clients queue RequestHandles per model; the planner
    decides which model to put on the GPU next (SCS over predicted sequences), loads
    it with the waiting prompts prefilled under the weight stream, and releases the
    bucket. `gpu_slots` engines may be resident at once; `host_slots` may keep a
    pinned host copy, the rest fall back to SSD."""

    def __init__(self, gpu_slots: int = 1, host_slots: int = 8, planner: str = "scs", speculate: bool = True):
        self.engine_pool: Dict[str, Engine] = {}
        self.engine_kwargs: Dict[str, dict] = {}     # model -> Engine.create kwargs
        self.program_template: Dict[str, Program] = {}
        self.instances: Dict[Tuple[str, int], ProgramInstance] = {}
        self.pending: Dict[str, List[RequestHandle]] = {}
        self.gpu_slots, self.host_slots, self.planner, self.speculate = gpu_slots, host_slots, planner, speculate
        self.load_cost: Dict[str, Dict[str, float]] = {}  # model -> tier -> measured seconds
        self.last_used: Dict[str, float] = {}
        self.active: Dict[str, int] = {}             # model -> dispatched requests not yet finished
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
            await eng.offload()
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

    # ---- requests ----------------------------------------------------------------

    async def submit(self, handle: RequestHandle) -> None:
        self.pending.setdefault(handle.model_name, []).append(handle)
        self._wake.set()

    async def _serve(self, handle: RequestHandle) -> None:
        engine = self.engine_pool[handle.model_name]
        try:
            p = handle.call_params
            sp = SamplingParams(
                temperature=p.get("temperature", 0.7), max_tokens=p.get("max_tokens"), stop=p.get("stop"),
                structured_outputs=StructuredOutputsParams(json=handle.schema) if handle.schema else None)
            handle.text = await engine.generate(engine.render(handle.messages),
                                                f"{handle.kind}-{uuid.uuid4().hex}", sp, handle.cache_salt)
            self.stats.append((handle.callsite_key, handle.kind, dict(engine.last)))
        except Exception as e:
            handle.error = str(e)
        finally:
            self.active[handle.model_name] -= 1
            self.last_used[handle.model_name] = time.perf_counter()
            handle.done.set()
            self._wake.set()
        if self.speculate:
            asyncio.create_task(self._speculate(handle))

    def _dispatch(self) -> None:
        """Release every pending handle whose model is resident."""
        for model, handles in list(self.pending.items()):
            if self.engine_pool.get(model) is not None and self.engine_pool[model].resident:
                del self.pending[model]
                self.active[model] = self.active.get(model, 0) + len(handles)  # busy before the tasks start
                for h in handles:
                    asyncio.create_task(self._serve(h))

    # ---- planning ----------------------------------------------------------------

    def sequences(self) -> List[List[str]]:
        """Per live instance: the model it waits for or runs on, then the predicted rest."""
        out = []
        for inst in self.instances.values():
            seq: List[str] = []
            waiting = [h.model_name for hs in self.pending.values() for h in hs if h.instance is inst]
            if waiting:
                seq.append(waiting[0])
            elif inst.calls:  # a tool loop in flight keeps its model
                seq.append(next(iter(inst.calls.values())).model_name)
            for m in inst.expected:
                if not seq or seq[-1] != m:
                    seq.append(m)
            if seq:
                out.append(seq)
        return out

    def cost(self, model: str) -> float:
        eng = self.engine_pool.get(model)
        tier = eng.tier if eng is not None else "ssd"
        return self.load_cost.get(model, {}).get(tier, DEFAULT_LOAD_COST[tier])

    def plan(self) -> List[str]:
        seqs = self.sequences()
        if self.planner == "reactive":
            return fifo_order([[h.model_name] for hs in self.pending.values() for h in hs])
        resident = {m for m, e in self.engine_pool.items() if e.resident}
        return scs_order(seqs, self.cost, resident)

    def _victim(self, order: List[str]) -> Optional[Engine]:
        """An idle resident engine to evict: the one the plan needs latest (LRU on ties)."""
        idle = [e for e in self.engine_pool.values() if e.resident and not e.busy and not self.active.get(e.name)]
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
            except Exception as e:  # keep planning; the failed load surfaces on the next wake
                print(f"[planner] error: {e!r}", flush=True)

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
        """Bring `model` to the GPU, prefilling the waiting prompts and the predicted
        static prefixes of instances heading to it under the weight stream."""
        eng = self.engine_pool.get(model)
        if eng is None:
            eng = await self.add_engine(model)
        prompts = [eng._with_salt(eng.render(h.messages), h.cache_salt) for h in self.pending.get(model, [])]
        for inst in self.instances.values():
            if inst.expected[:1] == [model] and not any(h.instance is inst for hs in self.pending.values() for h in hs):
                full = self._spec_prompt(inst, inst.expected_path[0], eng)
                if full is not None:
                    prompts.append(eng._with_salt(full, inst.salt()))
        tier = eng.tier
        t = time.perf_counter()
        await eng.load(prefill=prompts)
        secs = time.perf_counter() - t
        self.load_cost.setdefault(model, {})[tier] = secs
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
        for key in inst.program.successor_readiness(handle.callsite_key, inst.done):
            model = inst.program.model_of(inst.sites[key], handle.model_name)
            eng = self.engine_pool.get(model)
            if eng is None or not eng.resident:
                continue
            full = self._spec_prompt(inst, key, eng)
            if full is not None and eng.spec_allowance() >= len(eng.tokenize(full)):
                await eng.prefill(full, f"spec-{uuid.uuid4().hex}", cache_salt=handle.cache_salt)
