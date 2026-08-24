import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vllm.sampling_params import SamplingParams

from runtime.engine import ModelEngine
from runtime.routing_speculate import RoutingSpeculate
from utils.jac_static_parser import ProgramTopology, ByLLMCallsite
from utils.interceptor_receiver import InterceptorLLMBackend, ByLLMRequest
from console_helper.debug_output import console_debug, console_log, console_warn, console_error

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class _CallState:
    backend: InterceptorLLMBackend
    request: ByLLMRequest
    site: ByLLMCallsite
    messages: list[dict[str, str]]
    done: set[str]
    prefilled_sites: set[str] = field(default_factory=set)
    pending_tool_text: Optional[str] = None  # assistant turn awaiting a tool_result
    turn_warm: Optional[Tuple[str, int]] = None  # (tool-turn tag, tokens warmed so far)
    route_plan: Optional[List[ByLLMCallsite]] = None  # visit successors, ranked once per call
    inbox: asyncio.Queue[Optional[ByLLMRequest]] = field(default_factory=asyncio.Queue)


class GuardServer:
    def __init__(
        self, engine: ModelEngine, num_workers: int = 4, proactive_prefill: bool = True,
        spec_policy: str = "global", spec_chunk: int = 128,
    ):
        self._num_workers = num_workers
        self._engine = engine
        self._programs: Dict[str, ProgramTopology] = {}
        self._llm_backend: Dict[str, InterceptorLLMBackend] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._calls: Dict[Tuple[int, int], _CallState] = {}
        self._last_call: Dict[int, _CallState] = {}
        self._done: Dict[int, set[str]] = {}
        self._warm_tokens: Dict[str, int] = {}  # callsite_uuid -> prompt tokens last prefilled
        self._generation_slots = asyncio.Semaphore(num_workers)
        self._speculate = RoutingSpeculate(engine)
        self._proactive_prefill = proactive_prefill
        self._spec_policy = spec_policy  # "global" | "newest" (ablation: pre-multi-tenant behavior)
        self._spec_chunk = spec_chunk  # max uncached tokens per speculative prefill request
        self._engine_idle_interval = 0.01

    def add_program(self, program_name: str, src_path: str, port: int) -> None:
        self._programs[program_name] = ProgramTopology(program_name, src_path)
        self._programs[program_name].parse_dependency()
        self._llm_backend[program_name] = InterceptorLLMBackend(
            program_name, queue=self._task_queue, comm_port=port
        )

    async def serve(self) -> None:
        if not self._llm_backend:
            raise RuntimeError("no programs registered")

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

    async def monitor_idle(self) -> None:
        while True:
            await asyncio.sleep(self._engine_idle_interval)
            if not self._calls or not self._engine.has_prefill_room(1):
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
        most spec_chunk uncached tokens, resuming across ticks.
        False only when the token budget is exhausted for this tick."""
        text = state.pending_tool_text
        if text is None:
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
            warmed = self._engine.count_tokens(self._engine.render(state.messages))
        while warmed < len(ids):
            cost = min(self._spec_chunk, len(ids) - warmed)
            if not self._engine.has_prefill_room(cost):
                state.turn_warm = (tag, warmed)
                return False
            await self._engine.prefill(ids[: warmed + cost], f"prefill-{tag}-{uuid.uuid4().hex}", cost)
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
                    state.route_plan = await self._speculate.sort_candidate_calls(state, program)
                sites = state.route_plan
            else:
                sites = [
                    site
                    for key in program.next_calls(state.site)
                    for site in program.sites_of(key)
                ]
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
                constants = {
                    name: repr(info["spec"]["value"])
                    for name, info in program.ready_params(
                        site.key, state.done, via=state.site.key
                    ).items()
                    if info["ready"] and info["spec"]["kind"] == "const"
                }
                site.incremental_prompt(constants)
                ids = self._engine.tokenize(self._engine.render(site.get_ready_prompt()))
                warmed = self._warm_tokens.get(site.callsite_uuid, 0)
                if warmed < len(ids):
                    cost = min(self._spec_chunk, len(ids) - warmed)
                    if not self._engine.has_prefill_room(cost):
                        return
                    await self._engine.prefill(
                        ids[: warmed + cost], f"prefill-{site.callsite_uuid}-{uuid.uuid4().hex}", cost
                    )
                    warmed += cost
                    self._warm_tokens[site.callsite_uuid] = warmed
                if warmed >= len(ids):
                    pending.pop(0)
                    state.prefilled_sites.add(site.callsite_uuid)
                    if not pending:
                        queues.remove(entry)

    async def monitor_task(self) -> None:
        """Keep draining transport events; long-running calls get their own task."""
        while True:
            backend, request = await self._task_queue.get()
            try:
                if request.type == "register":
                    if request.program_name not in self._programs:
                        raise ValueError(f"unknown program {request.program_name!r}")
                    self._llm_backend[request.program_name] = backend
                    self._done[id(backend)] = set()
                    continue

                if request.type == "call":
                    # A new call is the protocol's implicit acknowledgement of
                    # the preceding final response on this connection.
                    await self._retire(backend)
                    asyncio.create_task(self._run_call(backend, request))
                    continue

                if request.type in ("tool_result", "reject"):
                    if request.call is None:
                        raise ValueError(f"{request.type} has no call id")
                    state = self._calls.get((id(backend), int(request.call)))
                    if state is None:
                        raise ValueError(f"unknown call id {request.call}")
                    await state.inbox.put(request)
                    continue

                if request.type == "generate":
                    await self._retire(backend)
                    asyncio.create_task(self._process_generate(backend, request))
                    continue

                raise ValueError(f"unsupported request type {request.type!r}")
            except Exception as exc:
                await backend.send(
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
        self, messages: list[dict[str, str]], sampling: SamplingParams, tag: str
    ) -> str:
        prompt = self._engine.render(messages)
        async with self._generation_slots:
            return await self._engine.generate(prompt, f"{tag}-{uuid.uuid4().hex}", sampling)

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
        self, backend: InterceptorLLMBackend, request: ByLLMRequest
    ) -> None:
        try:
            await self._process_request(backend, request)
        except Exception as exc:
            await backend.send({"type": "error", "id": request.id, "error": str(exc)})

    async def _process_request(
        self, backend: InterceptorLLMBackend, request: ByLLMRequest
    ) -> None:
        if request.program_name is None or request.key is None:
            raise ValueError("call request is missing program_name or key")
        program = self._programs.get(request.program_name)
        if program is None:
            raise ValueError(f"unknown program {request.program_name!r}")

        site = program.site_of(request.key, request.site)
        messages = site.assemble_prompt(request.args or {}, request.nest_scope_desc)
        call_id:int = request.id  #type: ignore
        state = _CallState(
            backend,
            request,
            site,
            messages,
            self._done.setdefault(id(backend), set()),
        )
        state_key = (id(backend), call_id)
        if state_key in self._calls:
            raise ValueError(f"duplicate call id {call_id}")
        self._calls[state_key] = state

        call_params = request.call_params or {}
        max_iterations = int(call_params.get("max_react_iterations") or 8)
        retries_left = int(call_params.get("max_output_retries") or 2)
        sampling = self._sampling_params(request, site)

        try:
            iterations = 0
            while True:
                text = await self._complete(
                    state.messages,
                    sampling,
                    f"{request.program_name}:{site.key}:{call_id}",
                )

                if site.decl.tools:
                    tool_call = self._parse_tool_call(text)
                    name = tool_call["name"]
                    arguments = tool_call.get("arguments", {})
                    if name != "finish_tool":
                        iterations += 1
                        if iterations > max_iterations:
                            raise RuntimeError("maximum ReAct iterations exceeded")
                        state.pending_tool_text = text
                        await backend.send(
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
                if ok:
                    for consumer, parameter in program.consumers_of(site.key):
                        item = program.provenance[consumer][parameter]["kind"] == "ret_item"
                        if item and not (isinstance(value, (list, tuple)) and value):
                            continue
                        # A ret_item consumer serves one element per iteration, and
                        # self-edges are pruned from the topology, so only the first
                        # iteration is ever warmable — bind its element.
                        view = repr(value[0]) if item else repr(value)
                        for consumer_site in program.sites_of(consumer):
                            consumer_site.incremental_prompt({parameter: view})
                            state.prefilled_sites.discard(consumer_site.callsite_uuid)
                            self._warm_tokens.pop(consumer_site.callsite_uuid, None)

                await backend.send(
                    {"type": "final", "call": call_id, "output": output, "text": text}
                )
                self._last_call[id(backend)] = state

                event = await state.inbox.get()
                if event is None:
                    return
                if event.type != "reject":
                    raise ValueError("only reject is valid after final")
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
            await backend.send({"type": "error", "id": call_id, "error": str(exc)})
        finally:
            self._calls.pop(state_key, None)
            if self._last_call.get(id(backend)) is state:
                self._last_call.pop(id(backend), None)
            site.clear_bindings()
            self._warm_tokens.pop(site.callsite_uuid, None)

    async def _retire(self, backend: InterceptorLLMBackend) -> None:
        """Close the window the previous frame left open on this connection.

        A `final` (or a routing `result`) is only acknowledged by the client's next
        frame, and the gap between them — client executing local code, engine idle —
        is exactly when monitor_idle speculates. So the state stays registered until
        the next frame arrives, and this is what drops it."""
        previous = self._last_call.pop(id(backend), None)
        if previous is None:
            return
        if previous.request.type == "generate":
            self._calls.pop((id(backend), previous.request.id), None)  # type: ignore[arg-type]
            previous.site.clear_bindings()
            self._warm_tokens.pop(previous.site.callsite_uuid, None)
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
        self, backend: InterceptorLLMBackend, request: ByLLMRequest
    ) -> None:
        try:
            if request.id is None or request.messages is None:
                raise ValueError("generate request is missing id or messages")
            site = self._route_site(request)
            if site is None:
                await backend.send({
                    "type": "result",
                    "id": request.id,
                    "text": await self._complete(
                        request.messages,
                        self._sampling_params(request),
                        f"{request.program_name}:generate:{request.id}",
                    ),
                })
                return
            state = _CallState(
                backend,
                request,
                site,
                list(request.messages),
                self._done.setdefault(id(backend), set()),
            )
            # Registered before generation so that the idle window *after* the
            # router answers finds it as the current call and speculates on the
            # node abilities the chosen candidate may run.
            self._calls[(id(backend), request.id)] = state
            try:
                text = await self._complete(
                    request.messages,
                    self._sampling_params(request, site),
                    f"{request.program_name}:{site.key}:{request.id}",
                )
            except Exception:
                self._calls.pop((id(backend), request.id), None)
                raise
            state.done.add(site.key)
            await backend.send({"type": "result", "id": request.id, "text": text})
            self._last_call[id(backend)] = state
        except Exception as exc:
            await backend.send({"type": "error", "id": request.id, "error": str(exc)})

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
