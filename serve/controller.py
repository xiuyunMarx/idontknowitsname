import asyncio
import json
import os
import re
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from instance_engine.model import Engine
from instance_engine.route_speculate import RouteSpeculate
from serve.planner import fifo_order, scs_order
from static_pass import link
from static_pass.instance import ProgramInstance
from static_pass.primitives import ByLLMFunc, Program, RequestHandle, VisitByLLM

DEFAULT_LOAD_COST = {"gpu": 0.0, "host": 1.0, "ssd": 3.0}  # relative, until measured


def _bare_schema(schema: dict) -> dict:
    """The JSON schema itself, out of an OpenAI-style response_format wrapper if given one."""
    if schema.get("type") == "json_schema" and isinstance(schema.get("json_schema"), dict):
        inner = schema["json_schema"]
        return inner.get("schema", inner)
    return schema


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
        self.running: "set[RequestHandle]" = set()   # handles inside engine.generate right now
        self.out_len: Dict[str, Tuple[float, float]] = {}  # callsite -> EMA (mean, mean of squares) of output tokens
        self.out_len_alpha = 0.3
        self.router = RouteSpeculate()
        self.route_probe = True      # probe a routing call's answer without thinking, ahead of the real one
        self.route_probes: list = []  # (callsite_key, probe_text, narrowed_to, seconds) per probe
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
        {"set": {"planner": "scs"|"reactive", "speculate": bool, "gpu_slots": n, "host_slots": n}}
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
                if k not in ("planner", "speculate", "gpu_slots", "host_slots", "route_probe"):
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
            self.loads.clear(); self.stats.clear(); self.last_used.clear(); self.load_cost.clear(); self.out_len.clear()
            self.route_probes.clear()
        out = {"planner": self.planner, "speculate": self.speculate, "gpu_slots": self.gpu_slots,
               "host_slots": self.host_slots, "route_probe": self.route_probe}
        if req.get("stats"):
            ttft = [st["ttft_ms"] for _, _, st in self.stats]
            out["stats"] = {"loads": len(self.loads), "load_s": round(sum(l[2] for l in self.loads), 2),
                            "route_probes": len(self.route_probes),
                            "route_probe_hits": sum(1 for _, _, n, _ in self.route_probes if n),
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
            probe_armed = self.speculate and self.route_probe and handle.kind == "generate" \
                and isinstance(handle.instance.sites.get(handle.callsite_key), VisitByLLM)

            def progress(first_token_at: float, out_tokens: int) -> None:
                nonlocal probe_armed
                handle.first_token_at, handle.out_tokens = first_token_at, out_tokens
                if probe_armed:  # the real prefill is done: its KV is shared, the probe costs only its suffix
                    probe_armed = False
                    asyncio.create_task(self._route_probe(handle, engine))
            self.running.add(handle)
            handle.text = await engine.generate(engine.render(handle.messages),
                                                f"{handle.kind}-{uuid.uuid4().hex}", sp, handle.cache_salt, progress)
            self.stats.append((handle.callsite_key, handle.kind, {**engine.last, "output_tokens": handle.out_tokens}))
            m, q = self.out_len.get(handle.callsite_key, (float(handle.out_tokens), float(handle.out_tokens) ** 2))
            a = self.out_len_alpha
            self.out_len[handle.callsite_key] = ((1 - a) * m + a * handle.out_tokens,
                                                 (1 - a) * q + a * handle.out_tokens ** 2)
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

    DEFAULT_TBT_MS = 30.0        # until a model is profiled
    DEFAULT_MAX_TOKENS = 1024    # a callsite without a literal max_tokens

    def _tbt(self, model: str) -> float:
        eng = self.engine_pool.get(model)
        return (eng.tbt_ms if eng is not None and eng.tbt_ms else self.DEFAULT_TBT_MS) / 1000.0

    def _tokens_range(self, call_params: dict, callsite_key: Optional[str] = None) -> Tuple[float, float]:
        """Output tokens one call may produce: [0, max_tokens] statically; with history
        for the callsite, the EMA mean +/- 2 sigma clipped into that."""
        cap = float(call_params.get("max_tokens") or self.DEFAULT_MAX_TOKENS)
        hist = self.out_len.get(callsite_key or "")
        if hist is None:
            return 0.0, cap
        m, q = hist
        sd = max(0.0, q - m * m) ** 0.5
        return max(0.0, min(cap, m - 2 * sd)), max(0.0, min(cap, m + 2 * sd))

    def _decode_range(self, model: str, call_params: dict, callsite_key: Optional[str] = None) -> Tuple[float, float]:
        """[shortest, longest] seconds one call of `model` at `callsite_key` may take."""
        lo, hi = self._tokens_range(call_params, callsite_key)
        tbt = self._tbt(model)
        return lo * tbt, hi * tbt

    def _decode_bound(self, model: str, call_params: dict, callsite_key: Optional[str] = None) -> float:
        return self._decode_range(model, call_params, callsite_key)[1]

    def _remaining_bound(self, handle: RequestHandle) -> float:
        """The static bound of a running request, tightened by what it has produced:
        the tokens left to its max_tokens at the step time observed on this very
        request (the current batch's real pace) once it has a few, else profiled."""
        return self._remaining_range(handle)[1]

    def _remaining_range(self, handle: RequestHandle) -> Tuple[float, float]:
        """[shortest, longest] seconds a running request may still take: its token
        range (static, or the callsite's history) minus what it has produced, at the
        step time observed on this very request (the current batch's real pace)
        once it has a few tokens, else profiled."""
        lo_t, hi_t = self._tokens_range(handle.call_params, handle.callsite_key)
        tbt = self._tbt(handle.model_name)
        if handle.first_token_at <= 0.0:  # still prefilling
            return lo_t * tbt, hi_t * tbt
        if handle.out_tokens >= 4:
            tbt = max(tbt, (time.perf_counter() - handle.first_token_at) / handle.out_tokens)
        if handle.out_tokens >= hi_t:  # history is already wrong for this one: back to the static cap
            hi_t = float(handle.call_params.get("max_tokens") or self.DEFAULT_MAX_TOKENS)
        return max(0.0, lo_t - handle.out_tokens) * tbt, max(0.0, hi_t - handle.out_tokens) * tbt

    def sequences(self) -> List[List[Tuple[str, float, float]]]:
        """Per live instance: (model, earliest, latest) for the model it waits for or
        runs on, then the predicted rest; the interval is when that need arrives, in
        seconds from now, from the static decode bounds of the calls before it (a
        call may finish at once, or run to its max_tokens). Consecutive callsites
        on one model merge into the first's arrival."""
        out = []
        for inst in self.instances.values():
            seq: List[Tuple[str, float, float]] = []
            lo = hi = 0.0
            waiting = [h for hs in self.pending.values() for h in hs if h.instance is inst]
            running = [h for h in self.running if h.instance is inst]
            if waiting:
                h = waiting[0]
                seq.append((h.model_name, 0.0, 0.0))
                d = self._decode_range(h.model_name, h.call_params, h.callsite_key)
            elif running:
                seq.append((running[0].model_name, 0.0, 0.0))
                d = self._remaining_range(running[0])
            elif inst.calls:  # a tool loop between turns keeps its model; the next turn has no progress yet
                call = next(iter(inst.calls.values()))
                seq.append((call.model_name, 0.0, 0.0))
                d = self._decode_range(call.model_name, call.call_params, call.site.callsite_key)
            else:
                d = (0.0, 0.0)
            lo, hi = lo + d[0], hi + d[1]
            for key, m in zip(inst.expected_path, (self.program_model(inst, k) for k in inst.expected_path)):
                if not seq or seq[-1][0] != m:
                    seq.append((m, lo, hi))
                d = self._decode_range(m, inst.sites[key].call_params, key)
                lo, hi = lo + d[0], hi + d[1]
            if seq:
                out.append(seq)
        return out

    def program_model(self, inst: ProgramInstance, key: str) -> str:
        fallback = inst.expected[0] if inst.expected else ""
        return inst.program.model_of(inst.sites[key], fallback)

    def guesses(self) -> List[Tuple[str, float]]:
        """Models that may be needed after some instance's divergence, by summed
        probability; only instances with no certain demand left contribute."""
        acc: Dict[str, float] = {}
        for inst in self.instances.values():
            if inst.expected or any(not c.finished for c in inst.calls.values()) \
                    or any(h.instance is inst for hs in self.pending.values() for h in hs):
                continue
            for _, m, p in inst.guesses:
                acc[m] = acc.get(m, 0.0) + p
        return sorted(acc.items(), key=lambda t: -t[1])

    def tier(self, model: str) -> str:
        eng = self.engine_pool.get(model)
        return eng.tier if eng is not None else "ssd"

    def cost(self, model: str, tier: Optional[str] = None) -> float:
        tier = tier or self.tier(model)
        return self.load_cost.get(model, {}).get(tier, DEFAULT_LOAD_COST[tier])

    def plan(self) -> List[str]:
        seqs = self.sequences()
        if self.planner == "reactive":
            return fifo_order([[h.model_name] for hs in self.pending.values() for h in hs])
        resident = {m for m, e in self.engine_pool.items() if e.resident}
        models = {m for s in seqs for m, _, _ in s} | resident
        hold = float("inf") if len(models) <= self.gpu_slots else 0.0  # no eviction pressure: timing is moot
        return scs_order(seqs, self.cost, self.tier, resident, slots=self.gpu_slots, hold=hold, depth=8)  # TODO: how to set depth

    def _victim(self, order: List[str]) -> Optional[Engine]:
        """An idle resident engine to evict: the one the plan needs latest (LRU on ties)."""
        idle = [e for e in self.engine_pool.values() if e.resident and not e.serving and not self.active.get(e.name)]
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
            # No certain demand left unserved: use a free slot (never an eviction) for the
            # likeliest post-divergence model, prefilling the prefixes that lead to it.
            if sum(e.resident for e in self.engine_pool.values()) < self.gpu_slots:
                guess = next((m for m, _ in self.guesses() if not (m in self.engine_pool and self.engine_pool[m].resident)), None)
                if guess is not None:
                    await self._load(guess, speculative=True)

    async def _load(self, model: str, speculative: bool = False) -> None:
        """Bring `model` to the GPU, prefilling under the weight stream the waiting
        prompts and the static prefixes of the callsites heading to it (certain next
        steps, or the guessed ones when the load itself is speculative)."""
        eng = self.engine_pool.get(model)
        if eng is None:
            eng = await self.add_engine(model)
        prompts = [eng._with_salt(eng.render(h.messages), h.cache_salt) for h in self.pending.get(model, [])]
        for inst in self.instances.values():
            if any(h.instance is inst for hs in self.pending.values() for h in hs):
                continue
            if inst.expected[:1] == [model]:
                keys = inst.expected_path[:1]
            elif speculative and not inst.expected:
                keys = [k for k, m, _ in inst.guesses if m == model]
            else:
                continue
            for key in keys:
                full = self._spec_prompt(inst, key, eng)
                if full is not None:
                    prompts.append(eng._with_salt(full, inst.salt()))
        tier = eng.tier
        t = time.perf_counter()
        await eng.load(prefill=prompts)
        secs = time.perf_counter() - t
        self.load_cost.setdefault(model, {})[tier] = secs
        self.loads.append((model, tier, secs, len(prompts)))
        print(f"[load] {model} from={tier} {secs * 1000:.0f}ms prefilled={len(prompts)}"
              + (" speculative=1" if speculative else ""), flush=True)

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

    async def _route_probe(self, handle: RequestHandle, eng: Engine) -> None:
        """A routing call (`visit ... by llm`) is decoding, thinking first; ask the same
        prompt with thinking suppressed for its first handle, and if that names one
        candidate archetype, make its callsites the instance's expected path so the
        planner loads and prefills for them before the real answer arrives."""
        inst = handle.instance
        t = time.perf_counter()
        if not eng.speculable:
            return
        try:
            text = await self.router.probe(eng, eng.render(handle.messages), 16, handle.cache_salt, prompt_cached=True)
        except Exception as e:
            print(f"[route-probe] {handle.callsite_key} error {e!r}", flush=True)
            return
        narrowed = self._route_narrow(inst, handle.callsite_key, text) if text and not handle.done.is_set() else []
        self.route_probes.append((handle.callsite_key, text, narrowed, time.perf_counter() - t))
        print(f"[route-probe] {handle.callsite_key} -> {text!r} narrowed={narrowed} "
              f"{(time.perf_counter() - t) * 1000:.0f}ms", flush=True)
        if narrowed:
            self._wake.set()

    @staticmethod
    def _slug(name: str) -> str:
        return re.sub(r"\W+", "_", name).strip("_")

    def _route_narrow(self, inst: ProgramInstance, visit_key: str, handle_text: str) -> List[str]:
        """Successor callsites of `visit_key` hosted by the archetype the probed handle
        names (handles are the slugified archetype, suffixed when several nodes share
        it); set as the instance's expected path. Empty when nothing matches."""
        succ = inst.program.next_call.get(visit_key, [])
        hosts = {inst.program.callsites[k].host for k in succ if isinstance(inst.program.callsites.get(k), ByLLMFunc)}
        match = max((h for h in hosts if handle_text == self._slug(h) or handle_text.startswith(self._slug(h) + "_")),
                    key=len, default=None)
        if match is None:
            return []
        keys = [k for k in succ if isinstance(inst.program.callsites.get(k), ByLLMFunc) and inst.program.callsites[k].host == match]
        if not keys:
            return []
        path = keys[:1] + inst.program.certain_chain(keys[0])
        inst.expected_path = path
        inst.expected = []
        for key in path:
            m = inst.program.model_of(inst.sites[key], inst.program.model_of(inst.sites[visit_key], ""))
            if not inst.expected or inst.expected[-1] != m:
                inst.expected.append(m)
        inst.guesses = []
        return path

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
                 "instances: " + "; ".join(f"{k[0]}#{k[1]} calls={sum(not c.finished for c in i.calls.values())} expected={[m.split('/')[-1] for m in i.expected]}"
                                           for k, i in self.instances.items())]
        for m, e in self.engine_pool.items():
            lines.append(f"engine {m.split('/')[-1]:<22} tier={e.tier:<4} busy={e.busy} prefill={e._inflight_prefill} "
                         f"decode={e._inflight_decode} spec={len(e._spec_tasks)} "
                         f"unfinished={e.engine.output_processor.has_unfinished_requests()}")
        for t in asyncio.all_tasks():
            frames = t.get_stack()[-3:]
            lines.append("task: " + " <- ".join(f"{f.f_code.co_name}:{f.f_lineno}" for f in frames))
        return "\n".join(lines)
