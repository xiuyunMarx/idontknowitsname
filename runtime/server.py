import asyncio
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vllm.sampling_params import SamplingParams

from runtime.engine import ModelEngine
from runtime.routing_speculate import RoutingSpeculate
from utils.jac_static_parser import ProgramTopology, ByLLMCallsite
from utils.interceptor_receiver import InterceptorLLMBackend, ClientSession, ByLLMRequest
from console_helper.debug_output import console_debug, console_log, console_warn, console_error

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

Bindings = Dict[str, str]  # param name -> repr, insertion order = warm prefix order
WarmKey = Tuple[str, Tuple[Tuple[str, str], ...], Optional[str]]  # (callsite uuid, bindings, cache salt) = the warmed bytes


@dataclass
class _CallState:
    session: ClientSession
    request: ByLLMRequest
    site: ByLLMCallsite
    messages: list[dict[str, str]]
    done: set[str]
    prefilled_sites: set[str] = field(default_factory=set)
    pending_tool_text: Optional[str] = None  # assistant turn awaiting a tool_result
    turn_warm: Optional[Tuple[str, int]] = None  # (tool-turn tag, tokens warmed so far)
    route_plan: Optional[List[ByLLMCallsite]] = None  # visit successors, ranked once per call
    cache_salt: Optional[str] = None  # this instance's prefix-cache salt (None = shared cache)
    inbox: asyncio.Queue[Optional[ByLLMRequest]] = field(default_factory=asyncio.Queue)


@dataclass
class _SessionState:
    """Everything the server remembers about one running program instance.

    A callsite object is a compile-time template shared by every instance of
    the program; the values that flow into its parameters at run time belong
    here, keyed by callsite uuid. Two instances of the same program therefore
    never see each other's arguments."""
    done: set[str] = field(default_factory=set)  # callsite keys served in this instance
    bindings: Dict[str, Bindings] = field(default_factory=dict)  # callsite uuid -> ready params
    last_call: Optional[_CallState] = None  # final sent, next frame not yet seen
    cache_salt: Optional[str] = None  # per-instance prefix-cache salt under cache isolation


class GuardServer:
    SPEC_FEATURES = frozenset({"const", "ret", "toolturn", "probe"})

    def __init__(
        self, engine: ModelEngine, num_workers: int = 4, proactive_prefill: bool = True,
        spec_policy: str = "global", spec_chunk: int = 128,
        spec_features: Optional[set] = None, spec_order: str = "static",
        cache_isolation: str = "none",
    ):
        # Ablation switches. Features are the compile-time facts speculation may
        # use beyond the may-run-next topology itself: const/ret parameter
        # provenance, the in-flight call's own tool turn, and the route probe.
        # With none of them, only each successor's invariant (anchor) is warmed.
        self._spec_features = self.SPEC_FEATURES if spec_features is None else set(spec_features)
        unknown = self._spec_features - self.SPEC_FEATURES
        if unknown:
            raise ValueError(f"unknown spec features {sorted(unknown)}")
        if spec_order not in ("static", "random"):
            raise ValueError("spec_order must be static or random")
        self._spec_order = spec_order  # random: shuffle successor order (predictor ablation)
        self._num_workers = num_workers
        self._engine = engine
        self._programs: Dict[str, ProgramTopology] = {}
        self._llm_backend: Dict[str, InterceptorLLMBackend] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._sessions: Dict[int, _SessionState] = {}  # session id -> instance state
        self._calls: Dict[Tuple[int, int], _CallState] = {}  # (session id, call id) -> in flight
        # Warm state is a property of the bytes, not of who asked for them: two
        # instances warming the same successor with the same bindings share one
        # entry, so the second does not re-issue what the first already prefilled.
        self._warm_tokens: Dict[WarmKey, int] = {}
        self._generation_slots = asyncio.Semaphore(num_workers)
        self._speculate = RoutingSpeculate(engine)
        self._proactive_prefill = proactive_prefill
        self._spec_policy = spec_policy  # "global" | "newest" (ablation: pre-multi-tenant behavior)
        self._spec_chunk = spec_chunk  # max uncached tokens per speculative prefill request
        self._engine_idle_interval = 0.01
        if cache_isolation not in ("none", "instance"):
            raise ValueError("cache_isolation must be none or instance")
        # instance: every workflow instance (client connection) gets its own
        # prefix-cache salt, so no instance ever hits blocks another one computed —
        # the FaaS setting where a task's KV cache dies with the task. Speculative
        # prefills carry the salt of the instance they are issued for.
        self._cache_isolation = cache_isolation

    def add_program(self, program_name: str, src_path: str, port: int) -> None:
        self._programs[program_name] = ProgramTopology(program_name, src_path)
        self._programs[program_name].parse_dependency()
        self._llm_backend[program_name] = InterceptorLLMBackend(
            program_name, queue=self._task_queue, comm_port=port
        )

    async def serve(self) -> None:
        if not self._llm_backend:
            raise RuntimeError("no programs registered")
        await self._warmup()

        tasks = [
            asyncio.create_task(self.monitor_task()),
            *(
                asyncio.create_task(backend.listen())
                for backend in self._llm_backend.values()
            ),
        ]
        if self._proactive_prefill:
            tasks.append(asyncio.create_task(self.monitor_idle()))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _warmup(self) -> None:
        """Reach the already-serving state before listeners open: compile the chat
        template, tokenize every callsite's invariant prefix (what a speculation
        tick renders), and run the engine's warmup. `listening on` is printed by
        the listeners, so a harness waiting for it also waits for this."""
        started = time.perf_counter()
        sites = 0
        for program in self._programs.values():
            for site in program.callsites:
                self._engine.tokenize(self._engine.render(site.get_ready_prompt({})))
                sites += 1
        await self._engine.warmup()
        console_log(f"[guard] warmup done: {sites} callsites rendered, {(time.perf_counter() - started) * 1000:.0f} ms")

    # ---- per-instance state -------------------------------------------------

    def _new_session_state(self, session: ClientSession) -> _SessionState:
        salt = f"{session.program_name}#{session.session_id}" if self._cache_isolation == "instance" else None
        return _SessionState(cache_salt=salt)

    def _session(self, session: ClientSession) -> _SessionState:
        sess = self._sessions.get(session.session_id)
        if sess is None:
            sess = self._sessions[session.session_id] = self._new_session_state(session)
        return sess

    @staticmethod
    def _warm_key(site: ByLLMCallsite, bindings: Bindings, salt: Optional[str] = None) -> WarmKey:
        return site.callsite_uuid, tuple(bindings.items()), salt

    def _forget_site(self, sess: _SessionState, site: ByLLMCallsite) -> None:
        """Drop an instance's bindings for a site once it has been served (or the
        instance is gone) and the warm entry those bytes had."""
        bindings = sess.bindings.pop(site.callsite_uuid, None)
        if bindings is not None:
            self._warm_tokens.pop(self._warm_key(site, bindings, sess.cache_salt), None)

    async def _drop_session(self, session: ClientSession) -> None:
        """The instance's connection closed: release every call waiting on it and
        every warm entry its bindings named."""
        sess = self._sessions.pop(session.session_id, None)
        for key in [k for k in self._calls if k[0] == session.session_id]:
            state = self._calls.pop(key)
            await state.inbox.put(None)
        if sess is None:
            return
        program = self._programs.get(session.program_name)
        for site_uuid, bindings in sess.bindings.items():
            self._warm_tokens.pop((site_uuid, tuple(bindings.items()), sess.cache_salt), None)
        console_debug(f"[guard] dropped {session.tag}"
                      + (f" ({len(sess.done)} sites served)" if program else ""))

    # ---- speculation ---------------------------------------------------------

    async def monitor_idle(self) -> None:
        while True:
            await asyncio.sleep(self._engine_idle_interval)
            if not self._calls or self._engine.spec_allowance() <= 0:
                continue

            # Speculation is best-effort: a failed tick must never kill serving.
            try:
                states = list(self._calls.values())
                if self._spec_policy == "newest":
                    states = states[-1:]

                # Every in-flight call's own next turn comes first: those bytes are certain.
                for state in reversed(states):
                    if not await self._warm_tool_turn(state):
                        break
                else:
                    await self._warm_successors(states)
            except Exception as exc:
                console_error(f"[guard] speculation tick failed: {exc!r}")

    async def _warm_tool_turn(self, state: _CallState) -> bool:
        """Prefill the in-flight call's pending tool-calling turn in chunks of at
        most spec_chunk uncached tokens, resuming across ticks. False only when the token budget is exhausted for this tick."""
        text = state.pending_tool_text
        if text is None or "toolturn" not in self._spec_features:
            return True
        tag = f"toolturn-{state.site.callsite_uuid}-{len(state.messages)}"
        if tag in state.prefilled_sites:
            return True
        ids = self._engine.tokenize(self._engine.render(
            state.messages
            + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "<tool_result>"},
            ]
        ))
        # The transcript is resident from the call's own turns; chunks cover only
        # the tail (the assistant text over-counts a little — its tokens are in
        # cache from decode — which only errs safe).
        if state.turn_warm and state.turn_warm[0] == tag:
            warmed = state.turn_warm[1]
        else:
            warmed = len(self._engine.tokenize(self._engine.render(state.messages)))
        while warmed < len(ids):
            # The step's slack, not a fixed chunk, sizes each request under load.
            cost = min(self._spec_chunk, self._engine.spec_allowance(), len(ids) - warmed)
            if cost <= 0:
                state.turn_warm = (tag, warmed)
                return False
            if not await self._engine.prefill(ids[: warmed + cost], f"prefill-{tag}-{uuid.uuid4().hex}", cost,
                                              cache_salt=state.cache_salt):
                state.turn_warm = (tag, warmed)  # killed by a real admission: resume here next tick
                return False
            warmed += cost
        state.turn_warm = None
        state.prefilled_sites.add(tag)
        return True

    async def _warm_successors(self, states: List[_CallState]) -> None:
        queues = []
        for state in reversed(states):
            program = self._programs[state.request.program_name]  # type: ignore[index]
            if state.site.is_visit:
                # Probed once per call: the ranked plan is cached on the state so
                # subsequent idle ticks reuse it instead of re-probing the router.
                if state.route_plan is None:
                    # The probe is a speculative request like any other: it needs
                    # a step it may use, and None means a real admission killed it
                    # (or owns the step), so the plan stays unranked until a
                    # later tick probes again.
                    if self._engine.spec_allowance() <= 0:
                        continue
                    state.route_plan = await self._speculate.sort_candidate_calls(
                        state, program, rank="probe" in self._spec_features
                    )
                    if state.route_plan is None:
                        continue
                sites = state.route_plan
            else:
                sites = [
                    site
                    for key in program.next_calls(state.site)
                    for site in program.sites_of(key)
                ]
            if self._spec_order == "random":
                sites = random.sample(sites, len(sites))
            pending = [s for s in sites if s.callsite_uuid not in state.prefilled_sites]
            if pending:
                queues.append((state, program, pending))

        # One chunk per call per round, newest call first, so no tenant's long
        # successor list starves the others; capping each speculative request at
        # spec_chunk uncached tokens bounds how long an arriving real prefill
        # can wait behind speculation.
        while queues:
            for entry in list(queues):
                state, program, pending = entry
                site = pending[0]
                sess = self._session(state.session)
                bindings = sess.bindings.setdefault(site.callsite_uuid, {})
                if "const" in self._spec_features:
                    constants = {
                        name: repr(info["spec"]["value"])
                        for name, info in program.ready_params(
                            site.key, state.done, via=state.site.key
                        ).items()
                        if info["ready"] and info["spec"]["kind"] == "const"
                    }
                    site.incremental_prompt(bindings, constants)
                ids = self._engine.tokenize(self._engine.render(site.get_ready_prompt(bindings)))
                key = self._warm_key(site, bindings, sess.cache_salt)
                warmed = self._warm_tokens.get(key, 0)
                if warmed < len(ids):
                    cost = min(self._spec_chunk, self._engine.spec_allowance(), len(ids) - warmed)
                    if cost <= 0:
                        return
                    if not await self._engine.prefill(
                        ids[: warmed + cost], f"prefill-{site.callsite_uuid}-{uuid.uuid4().hex}", cost,
                        cache_salt=sess.cache_salt,
                    ):
                        return  # killed by a real admission; progress so far stays recorded
                    warmed += cost
                    self._warm_tokens[key] = warmed
                if warmed >= len(ids):
                    pending.pop(0)
                    state.prefilled_sites.add(site.callsite_uuid)
                    if not pending:
                        queues.remove(entry)

    # ---- transport events ----------------------------------------------------

    async def monitor_task(self) -> None:
        """Keep draining transport events; long-running calls get their own task."""
        while True:
            session, request = await self._task_queue.get()
            try:
                if request.type == "register":
                    if request.program_name not in self._programs:
                        raise ValueError(f"unknown program {request.program_name!r}")
                    self._sessions[session.session_id] = self._new_session_state(session)
                    continue

                if request.type == "disconnect":
                    await self._drop_session(session)
                    continue

                if request.type == "call":
                    # A new call is the protocol's implicit acknowledgement of
                    # the preceding final response on this connection.
                    await self._retire(session)
                    asyncio.create_task(self._run_call(session, request))
                    continue

                if request.type in ("tool_result", "reject"):
                    if request.call is None:
                        raise ValueError(f"{request.type} has no call id")
                    state = self._calls.get((session.session_id, int(request.call)))
                    if state is None:
                        raise ValueError(f"unknown call id {request.call}")
                    await state.inbox.put(request)
                    continue

                if request.type == "generate":
                    await self._retire(session)
                    asyncio.create_task(self._process_generate(session, request))
                    continue

                raise ValueError(f"unsupported request type {request.type!r}")
            except Exception as exc:
                await session.send(
                    {"type": "error", "id": request.id, "error": str(exc)}
                )

    def _sampling_params(
        self, request: ByLLMRequest, site: Optional[ByLLMCallsite] = None
    ) -> SamplingParams:
        params = dict(site.decl.call_params if site is not None else {})
        params.update(request.call_params or {})
        requested_temperature:float = request.temperature if request.temperature is not None else 0.7
        requested_max_tokens:int = request.max_tokens if request.max_tokens is not None else 512
        max_tokens = params.get("max_tokens")
        temperature = params.get("temperature")
        stop = request.stop
        include_stop = False
        if site is not None and site.decl.tools:
            stop = ["</tool_call>"]
            include_stop = True
        return SamplingParams(
            max_tokens=int(requested_max_tokens if max_tokens is None else max_tokens),
            temperature=float( requested_temperature if temperature is None else temperature),
            stop=stop,
            include_stop_str_in_output=include_stop,
        )

    async def _complete(
        self, messages: list[dict[str, str]], sampling: SamplingParams, tag: str,
        cache_salt: Optional[str] = None,
    ) -> str:
        prompt = self._engine.render(messages)
        async with self._generation_slots:
            return await self._engine.generate(prompt, f"{tag}-{uuid.uuid4().hex}", sampling, cache_salt=cache_salt)

    @staticmethod
    def _parse_tool_call(text: str) -> dict:
        match = _TOOL_CALL.search(text)
        payload = match.group(1) if match else text.strip()
        try:
            call, _ = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            if not payload:
                raise ValueError("model returned an empty tool response")
            call = {"name": "finish_tool", "arguments": {"final_output": payload}}
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("tool call must contain a string name")
        if not isinstance(call.get("arguments", {}), dict):
            raise ValueError("tool call arguments must be an object")
        return call

    async def _run_call(
        self, session: ClientSession, request: ByLLMRequest
    ) -> None:
        try:
            await self._process_request(session, request)
        except Exception as exc:
            await session.send({"type": "error", "id": request.id, "error": str(exc)})

    async def _process_request(
        self, session: ClientSession, request: ByLLMRequest
    ) -> None:
        if request.program_name is None or request.key is None:
            raise ValueError("call request is missing program_name or key")
        program = self._programs.get(request.program_name)
        if program is None:
            raise ValueError(f"unknown program {request.program_name!r}")

        sess = self._session(session)
        site = program.site_of(request.key, request.site)
        # This instance's warmed bindings render first so the served prompt
        # extends the prefix that was prefilled for it.
        messages = site.assemble_prompt(
            request.args or {}, request.nest_scope_desc, sess.bindings.get(site.callsite_uuid)
        )
        call_id:int = request.id  #type: ignore
        state = _CallState(session, request, site, messages, sess.done, cache_salt=sess.cache_salt)
        state_key = (session.session_id, call_id)
        if state_key in self._calls:
            raise ValueError(f"duplicate call id {call_id}")
        self._calls[state_key] = state

        call_params = request.call_params or {}
        max_iterations = int(call_params.get("max_react_iterations") or 8)
        retries_left = int(call_params.get("max_output_retries") or 2)
        sampling = self._sampling_params(request, site)
        tag = f"{request.program_name}:{site.key}:s{session.session_id}c{call_id}"

        try:
            iterations = 0
            while True:
                text = await self._complete(state.messages, sampling, tag, sess.cache_salt)

                if site.decl.tools:
                    tool_call = self._parse_tool_call(text)
                    name = tool_call["name"]
                    arguments = tool_call.get("arguments", {})
                    if name != "finish_tool":
                        iterations += 1
                        if iterations > max_iterations:
                            raise RuntimeError("maximum ReAct iterations exceeded")
                        state.pending_tool_text = text
                        await session.send(
                            {
                                "type": "tool_call",
                                "call": call_id,
                                "name": name,
                                "arguments": arguments,
                                "text": text,
                            }
                        )
                        event = await state.inbox.get()
                        state.pending_tool_text = None
                        if event is None:
                            return
                        if event.type != "tool_result":
                            raise ValueError("expected tool_result after tool_call")
                        state.messages.extend(
                            [
                                {"role": "assistant", "content": text},
                                {
                                    "role": "user",
                                    "content": f"<tool_result>{event.content}</tool_result>",
                                },
                            ]
                        )
                        continue
                    output = arguments.get("final_output")
                else:
                    output = text.strip()

                state.done.add(site.key)
                ok, value = site.decl.parse_response(output)
                console_debug(f"[final] {tag} typed_ok={ok} output={str(output)[:160]!r}")
                if ok and "ret" in self._spec_features:
                    for consumer, parameter in program.consumers_of(site.key):
                        item = program.provenance[consumer][parameter]["kind"] == "ret_item"
                        if item and not (isinstance(value, (list, tuple)) and value):
                            continue
                        # A ret_item consumer serves one element per iteration, and
                        # self-edges are pruned from the topology, so only the first
                        # iteration is ever warmable — bind its element.
                        view = repr(value[0]) if item else repr(value)
                        for consumer_site in program.sites_of(consumer):
                            # The bytes change, so the old warm entry no longer
                            # describes anything; the new bindings get a fresh key.
                            self._forget_site(sess, consumer_site)
                            consumer_site.incremental_prompt(
                                sess.bindings.setdefault(consumer_site.callsite_uuid, {}),
                                {parameter: view},
                            )
                            state.prefilled_sites.discard(consumer_site.callsite_uuid)

                await session.send(
                    {"type": "final", "call": call_id, "output": output, "text": text}
                )
                sess.last_call = state

                event = await state.inbox.get()
                if event is None:
                    return
                if event.type != "reject":
                    raise ValueError("only reject is valid after final")
                console_debug(f"[reject] {tag} feedback={str(event.feedback)[:160]!r}")
                state.done.discard(site.key)
                if retries_left <= 0:
                    raise RuntimeError("maximum output retries exceeded")
                retries_left -= 1
                state.messages.extend(
                    [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": f"The output was rejected: {event.feedback}",
                        },
                    ]
                )
        except Exception as exc:
            await session.send({"type": "error", "id": call_id, "error": str(exc)})
        finally:
            self._calls.pop(state_key, None)
            if sess.last_call is state:
                sess.last_call = None
            self._forget_site(sess, site)

    async def _retire(self, session: ClientSession) -> None:
        """Close the window the previous frame left open on this connection.

        A `final` (or a routing `result`) is only acknowledged by the client's next
        frame, and the gap between them — client executing local code, engine idle —
        is exactly when monitor_idle speculates. So the state stays registered until
        the next frame arrives, and this is what drops it."""
        sess = self._sessions.get(session.session_id)
        if sess is None or sess.last_call is None:
            return
        previous, sess.last_call = sess.last_call, None
        if previous.request.type == "generate":
            self._calls.pop((session.session_id, previous.request.id), None)  # type: ignore[arg-type]
            self._forget_site(sess, previous.site)
        else:
            await previous.inbox.put(None)

    def _route_site(self, request: ByLLMRequest) -> Optional[ByLLMCallsite]:
        """The routing callsite where a `generate` frame came from.

        `visit ... by llm()` has no callable, so the client keys it by source
        location (`__visit@file.jac:line`) and the static pass synthesizes the
        matching pseudo-decl from the VisitStmt. """
        if not request.key or not request.program_name:
            return None
        program = self._programs.get(request.program_name)
        if program is None:
            return None
        try:
            return program.site_of(request.key, request.site)
        except KeyError:
            console_warn(f"[guard] warning: no routing callsite for {request.key!r}; serving untracked")
            return None

    async def _process_generate(
        self, session: ClientSession, request: ByLLMRequest
    ) -> None:
        try:
            if request.id is None or request.messages is None:
                raise ValueError("generate request is missing id or messages")
            site = self._route_site(request)
            if site is None:
                await session.send({
                    "type": "result",
                    "id": request.id,
                    "text": await self._complete(
                        request.messages,
                        self._sampling_params(request),
                        f"{request.program_name}:generate:s{session.session_id}c{request.id}",
                        self._session(session).cache_salt,
                    ),
                })
                return
            sess = self._session(session)
            state = _CallState(session, request, site, list(request.messages), sess.done, cache_salt=sess.cache_salt)
            # Registered before generation so that the idle window *after* the
            # router answers finds it as the current call and speculates on the
            # node abilities the chosen candidate may run.
            self._calls[(session.session_id, request.id)] = state
            try:
                text = await self._complete(
                    request.messages,
                    self._sampling_params(request, site),
                    f"{request.program_name}:{site.key}:s{session.session_id}c{request.id}",
                    sess.cache_salt,
                )
            except Exception:
                self._calls.pop((session.session_id, request.id), None)
                raise
            state.done.add(site.key)
            await session.send({"type": "result", "id": request.id, "text": text})
            sess.last_call = state
        except Exception as exc:
            await session.send({"type": "error", "id": request.id, "error": str(exc)})

    def show_info(self, program_name: str):
        if program_name not in self._programs:
            raise ValueError(f"Program {program_name} not found.")
        p = self._programs[program_name]
        print("decls:", list(p.decls))
        print("\ncallsites (uuid | consumers):")
        for s in p.callsites:
            print(f"  {s.callsite_uuid} | {s.consumers}")
        print("\ntopology (may-run-next):")
        for k, succ in p.topology.items():
            print(f"  {k} -> {succ}")
        print("\nprovenance (param -> value source):")
        for k, params in p.provenance.items():
            for pname, spec in params.items():
                print(f"  {k}.{pname} <- {spec}")
        if p.callsites:
            s0 = p.callsites[-1]
            demo = {prm["name"]: repr(f"<{prm['name']} value>") for prm in s0.decl.params}
            partial = dict(list(demo.items())[:1])
            print(f"\n--- invariant system of {s0.callsite_uuid} ---\n{s0.invariant_system}")
            print(f"\n--- partial assembly ({list(partial)}) ---\n{s0.assemble_prompt(partial)[1]['content']}")
            print(f"\n--- full assembly ---\n{s0.assemble_prompt(demo)[1]['content']}")
