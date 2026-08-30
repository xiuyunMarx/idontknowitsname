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
from serve.cost import CostModel
from serve.planner import (INF, Chain, PlanInput, Step, choose, dp_order, fifo_order, greedy_order, keep_value,
                           next_use)
from serve.predict import predicted_chain
from static_pass import link
from static_pass.instance import ProgramInstance
from static_pass.primitives import ByLLMFunc, Program, RequestHandle

PLANNERS = ("fifo", "greedy", "dp")
SETTINGS = {"speculate": bool, "gpu_slots": int, "host_slots": int, "planner": str, "horizon": int,
            "hysteresis": float, "tau": float, "gamma": float, "learn_costs": bool}
MAX_CHAIN = 12  # predicted steps the cost-aware planners look at per instance


def _bare_schema(schema: dict) -> dict:
    """The JSON schema itself, out of an OpenAI-style response_format wrapper if given one."""
    if schema.get("type") == "json_schema" and isinstance(schema.get("json_schema"), dict):
        inner = schema["json_schema"]
        return inner.get("schema", inner)
    return schema


def _ms(xs: List[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": len(xs), "mean": round(1000 * sum(xs) / len(xs), 1),
            "p50": round(1000 * xs[len(xs) // 2], 1), "p95": round(1000 * xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1)}


class Controller:
    """Multi-tenant FaaS layer: clients queue RequestHandles per model; the planner
    orders the models to put on the GPU (FIFO over each instance's predicted chain,
    or the cost-aware greedy / DP of serve.planner), loads the next one with the
    waiting prompts prefilled under the weight stream, and releases the bucket.
    `gpu_slots` engines may be resident at once; `host_slots` may keep a pinned host
    copy, the rest fall back to SSD."""

    def __init__(self, gpu_slots: int = 1, host_slots: int = 8, speculate: bool = True, planner: str = "fifo"):
        self.engine_pool: Dict[str, Engine] = {}
        self.engine_kwargs: Dict[str, dict] = {}     # model -> Engine.create kwargs
        self.program_template: Dict[str, Program] = {}
        self.instances: Dict[Tuple[str, int], ProgramInstance] = {}
        self.pending: Dict[str, List[RequestHandle]] = {}
        self.gpu_slots, self.host_slots, self.speculate = gpu_slots, host_slots, speculate
        self.planner, self.horizon, self.hysteresis, self.tau, self.gamma = planner, 4, 0.10, 10.0, 1.0
        self.cost = CostModel()                      # knowledge, not a counter: survives reset
        self.last_used: Dict[str, float] = {}
        self.active: Dict[str, int] = {}             # model -> dispatched requests not yet finished
        self.running: "set[RequestHandle]" = set()   # handles inside engine.generate right now
        self.stats: list = []   # (callsite_key, kind, engine.last) per real request
        self.loads: list = []   # (model, from_tier, seconds, prefilled) per load
        self.requests: list = []   # per finished request: lifecycle timestamps (queue wait, ttft, e2e)
        self.plan_stats = {"plans": 0, "deviations": 0, "fallbacks": 0, "plan_ms": 0.0}
        self._plan_cache: Tuple[Optional[tuple], List[str]] = (None, [])
        self._last_pi: Optional[PlanInput] = None
        self._branches: Optional[Dict[str, dict]] = None   # program -> branch_freq snapshot restored on reset
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
            self.cost.register(model_name)
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
        {"set": {"speculate": bool, "gpu_slots": n, "host_slots": n, "planner": "fifo|greedy|dp",
                 "horizon": n, "hysteresis": f, "tau": f, "gamma": f, "learn_costs": bool}}
        {"reset": true[, "cold": true]}   park every engine on host (cold: on SSD), clear caches and counters
        {"calibrate": true}   load every engine once from SSD and once from host to fill the cost table
        {"forget": true}      drop the cost table back to its priors
        {"freeze_branches": true}   snapshot every program's branch predictor; each reset restores it
        {"stats": true[, "detail": true]}   loads / requests since the last reset (detail: per request)
        {"cost": true}        the cost table
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

    def _idle(self) -> bool:
        return not self.pending and not any(e.busy or self.active.get(e.name) for e in self.engine_pool.values())

    async def control(self, req: dict) -> dict:
        if "set" in req:
            for k, v in req["set"].items():
                if k not in SETTINGS:
                    raise ValueError(f"unknown setting {k!r}")
                if k == "planner":
                    if v not in PLANNERS:
                        raise ValueError(f"planner must be one of {PLANNERS}")
                    self.planner = v
                elif k == "learn_costs":
                    self.cost.learn = bool(v)
                else:
                    setattr(self, k, SETTINGS[k](v))
            self._plan_cache = (None, [])
            self._wake.set()
        if req.get("reset"):
            if not self._idle():
                raise RuntimeError("reset while requests are in flight")
            for e in self.engine_pool.values():
                if req.get("cold"):
                    if e.tier != "ssd":
                        await e.kill()
                elif e.resident:
                    await e.offload()
                await e.engine.reset_prefix_cache()
            self.loads.clear(); self.stats.clear(); self.last_used.clear(); self.requests.clear()
            self.plan_stats = {"plans": 0, "deviations": 0, "fallbacks": 0, "plan_ms": 0.0}
            self._plan_cache = (None, [])
            if self._branches is not None:
                for name, freq in self._branches.items():
                    self.program_template[name].branch_freq = {k: dict(v) for k, v in freq.items()}
        if req.get("freeze_branches"):
            self._branches = {name: {k: dict(v) for k, v in p.branch_freq.items()} for name, p in self.program_template.items()}
        if req.get("forget"):
            self.cost = CostModel(learn=self.cost.learn)
            for m in self.engine_pool:
                self.cost.register(m)
        if req.get("calibrate"):
            await self._calibrate()
        out = {"speculate": self.speculate, "gpu_slots": self.gpu_slots, "host_slots": self.host_slots,
               "planner": self.planner, "horizon": self.horizon, "hysteresis": self.hysteresis, "tau": self.tau,
               "gamma": self.gamma, "learn_costs": self.cost.learn, "frozen_branches": self._branches is not None}
        if req.get("stats"):
            ttft = [st["ttft_ms"] for _, _, st in self.stats]
            waits = [r["dispatched_at"] - r["created_at"] for r in self.requests if r["dispatched_at"]]
            ttft_submit = [r["first_token_at"] - r["created_at"] for r in self.requests if r["first_token_at"]]
            e2e = [r["done_at"] - r["created_at"] for r in self.requests]
            plans = self.plan_stats["plans"]
            out["stats"] = {"loads": len(self.loads), "load_s": round(sum(l[2] for l in self.loads), 2),
                            "load_order": [(m.split("/")[-1], t) for m, t, _, _ in self.loads],
                            "requests": len(ttft), "ttft_mean_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
                            "cached_frac": round(sum(st["cached_tokens"] for _, _, st in self.stats)
                                                 / max(1, sum(st["prompt_tokens"] for _, _, st in self.stats)), 3),
                            "queue_wait_ms": _ms(waits), "ttft_submit_ms": _ms(ttft_submit), "e2e_ms": _ms(e2e),
                            "sum_wait_s": round(sum(waits), 2), "failed": sum(r["error"] is not None for r in self.requests),
                            "planner": self.planner, "plans": plans, "deviations": self.plan_stats["deviations"],
                            "fallbacks": self.plan_stats["fallbacks"],
                            "plan_ms_mean": round(self.plan_stats["plan_ms"] / plans, 1) if plans else None}
            if req.get("detail"):
                out["requests"] = list(self.requests)
        if req.get("cost"):
            out["cost"] = self.cost.export(now=time.perf_counter())
        if req.get("dump"):
            out["dump"] = self.dump()
        return out

    async def _calibrate(self) -> None:
        """Measure every engine's SSD->GPU and host->GPU load once so the cost table starts
        from observations rather than size priors; leaves every engine on SSD."""
        if not self._idle():
            raise RuntimeError("calibrate while requests are in flight")
        learn, self.cost.learn = self.cost.learn, True
        try:
            for e in self.engine_pool.values():
                if e.resident:
                    await e.offload()
            for e in self.engine_pool.values():
                if e.tier != "ssd":
                    await e.kill()
                for _ in range(2):  # ssd -> gpu, then host -> gpu
                    tier, t = e.tier, time.perf_counter()
                    await e.load()
                    secs = time.perf_counter() - t
                    self.cost.observe_load(e.name, tier, secs)
                    print(f"[calibrate] {e.name} from={tier} {secs * 1000:.0f}ms", flush=True)
                    await e.offload()
                await e.kill()
        finally:
            self.cost.learn = learn

    # ---- requests ----------------------------------------------------------------

    async def submit(self, handle: RequestHandle) -> None:
        now = time.perf_counter()
        handle.created_at = handle.created_at or now
        self.cost.observe_arrival(handle.model_name, now)
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
            t0 = time.perf_counter()
            handle.text = await engine.generate(engine.render(handle.messages),
                                                f"{handle.kind}-{uuid.uuid4().hex}", sp, handle.cache_salt, progress)
            self.cost.observe_exec(handle.callsite_key, handle.model_name, time.perf_counter() - t0,
                                   first_turn=handle.kind in ("call", "generate"))
            self.stats.append((handle.callsite_key, handle.kind, {**engine.last, "output_tokens": handle.out_tokens}))
        except Exception as e:
            handle.error = repr(e)
            traceback.print_exc()
        finally:
            handle.done_at = time.perf_counter()
            self.requests.append({"callsite": handle.callsite_key, "kind": handle.kind, "model": handle.model_name,
                                  "created_at": handle.created_at, "dispatched_at": handle.dispatched_at,
                                  "first_token_at": handle.first_token_at, "done_at": handle.done_at,
                                  "output_tokens": handle.out_tokens, "error": handle.error})
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
                now = time.perf_counter()
                for h in handles:
                    h.dispatched_at = now
                    asyncio.create_task(self._serve(h))

    # ---- planning ----------------------------------------------------------------

    def _head(self, inst: ProgramInstance, now: float):
        """(model, kind, exec seconds, callsite_key, waited seconds) for the step `inst` is on
        now: the request it waits for, the one running for it, or the call it is mid-way
        through on the client (a tool loop between turns); None between calls."""
        waiting = [h for hs in self.pending.values() for h in hs if h.instance is inst]
        if waiting:
            h = waiting[0]
            return h.model_name, "waiting", self.cost.exec_s(h.callsite_key, h.model_name), h.callsite_key, now - h.created_at
        running = [h for h in self.running if h.instance is inst]
        if running:
            h = running[0]
            left = self.cost.exec_s(h.callsite_key, h.model_name) - (now - h.dispatched_at)
            return h.model_name, "running", max(0.0, left), h.callsite_key, 0.0
        open_calls = [c for c in inst.calls.values() if not c.finished]
        if open_calls:
            c = open_calls[0]
            return c.model_name, "open", self.cost.exec_s(c.site.callsite_key, c.model_name), c.site.callsite_key, 0.0
        return None

    def sequences(self) -> List[List[str]]:
        """Per live instance: the model it waits for or runs on, then the models of
        its predicted path to the end of the program, one per callsite."""
        out = []
        now = time.perf_counter()
        for inst in self.instances.values():
            head = self._head(inst, now)
            seq: List[str] = [head[0]] if head else []
            seq.extend(m for _, m in inst.predicted)
            if seq:
                out.append(seq)
        return out

    def plan_input(self) -> PlanInput:
        """What the cost-aware planners see: per instance its current step (weighted by how
        long it has waited) and up to MAX_CHAIN predicted steps (discounted by gamma**j),
        predicted loop-aware from the current callsite (serve.predict) rather than by
        `inst.predicted`, which stops at the first lap of a loop."""
        now = time.perf_counter()
        chains = []
        for inst in self.instances.values():
            steps: List[Step] = []
            head = self._head(inst, now)
            kind = "predicted"
            if head is not None:
                model, kind, exec_s, key, waited = head
                w = 1.0 + waited / self.tau if kind == "waiting" else 1.0
                steps.append(Step(model, exec_s, w, False, key))
                predicted = predicted_chain(inst.program, key, model, MAX_CHAIN)
            else:
                predicted = list(inst.predicted)
            for j, (key, m) in enumerate(predicted, 1):
                steps.append(Step(m, self.cost.exec_s(key, m), self.gamma ** j, True, key))
            if steps:
                chains.append(Chain(inst, tuple(steps[:MAX_CHAIN]), kind))
        resident = frozenset(m for m, e in self.engine_pool.items() if e.resident and not e.transitioning)
        busy = frozenset(m for m, e in self.engine_pool.items() if e.resident and (e.serving or self.active.get(m)))
        tier = {m: e.tier for m, e in self.engine_pool.items()}
        rate = {m: self.cost.rate(m, now) for m in self.engine_pool}
        return PlanInput(chains, resident, tier, self.cost.load_s, rate, self.gpu_slots, busy, frozenset(self._pinned()))

    def _fingerprint(self, pi: PlanInput) -> tuple:
        return (tuple(id(h) for hs in self.pending.values() for h in hs), tuple(sorted(id(h) for h in self.running)),
                tuple(sorted(pi.resident)), tuple(sorted(pi.tier.items())), tuple(id(i) for i in self.instances.values()),
                self.planner, self.horizon, self.hysteresis, self.gpu_slots)

    def plan(self) -> List[str]:
        fifo = fifo_order(self.sequences())
        if self.planner == "fifo":
            return fifo
        pi = self.plan_input()
        self._last_pi = pi
        key = self._fingerprint(pi)
        if key == self._plan_cache[0]:
            return self._plan_cache[1]
        t, info = time.perf_counter(), {}
        cand = greedy_order(pi) if self.planner == "greedy" else dp_order(pi, self.horizon, info=info)
        order, deviated = choose(pi, fifo, cand, self.hysteresis)
        self.plan_stats["plans"] += 1
        self.plan_stats["deviations"] += deviated
        self.plan_stats["fallbacks"] += bool(info.get("fallback"))
        self.plan_stats["plan_ms"] += (time.perf_counter() - t) * 1000
        self._plan_cache = (key, order)
        return order

    def _pinned(self) -> "set[str]":
        """Models some instance is in the middle of a call on (a tool loop between
        turns): its next turn is imminent, so evicting them only forces a reload."""
        return {c.model_name for inst in self.instances.values() for c in inst.calls.values() if not c.finished}

    def _keep(self, model: str, reload: Optional[float] = None) -> float:
        """keep_value of a resident/hosted model under the last plan input (0 = nobody wants it)."""
        pi = self._last_pi if self._last_pi is not None else self.plan_input()
        return keep_value(pi, model, next_use(pi), reload)

    def _victim(self, order: List[str]) -> Optional[Engine]:
        """An idle, unpinned resident engine to evict. FIFO: the one the plan needs latest (LRU on
        ties); cost-aware planners: the lowest keep_value (reload cost over time to next use)."""
        pinned = self._pinned()
        idle = [e for e in self.engine_pool.values()
                if e.resident and not e.transitioning and not e.serving and not self.active.get(e.name) and e.name not in pinned]
        if self.planner == "fifo":
            rank = {m: i for i, m in enumerate(order)}
            idle.sort(key=lambda e: (-rank.get(e.name, len(order)), self.last_used.get(e.name, 0.0)))
        else:
            idle.sort(key=lambda e: (self._keep(e.name), self.last_used.get(e.name, 0.0)))
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
            target = next((m for m in order if not (m in self.engine_pool and
                                                    (self.engine_pool[m].resident or self.engine_pool[m].transitioning))), None)
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
        self.cost.observe_load(model, tier, secs)
        print(f"[load] {model} from={tier} {secs * 1000:.0f}ms prefilled={len(prompts)}", flush=True)

    async def _trim_host(self) -> None:
        """Keep at most `host_slots` pinned copies. FIFO: the least recently used go to SSD;
        cost-aware planners: those whose extra SSD reload cost is least likely to be paid soon."""
        hosted = [e for e in self.engine_pool.values() if e.tier == "host"]
        if self.planner == "fifo":
            hosted.sort(key=lambda e: self.last_used.get(e.name, 0.0))
        else:
            hosted.sort(key=lambda e: (self._keep(e.name, self.cost.load_s(e.name, "ssd") - self.cost.load_s(e.name, "host")),
                                       self.last_used.get(e.name, 0.0)))
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
                 f"planner: {self.planner} {self.plan_stats}",
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
