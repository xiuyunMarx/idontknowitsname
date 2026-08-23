import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from vllm.sampling_params import SamplingParams

from runtime.engine import ModelEngine
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
    inbox: asyncio.Queue[Optional[ByLLMRequest]] = field(default_factory=asyncio.Queue)


class GuardServer:
    def __init__(
        self, engine: ModelEngine, num_workers: int = 4, proactive_prefill: bool = True
    ):
        self._num_workers = num_workers
        self._engine = engine
        self._programs: Dict[str, ProgramTopology] = {}
        self._llm_backend: Dict[str, InterceptorLLMBackend] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._calls: Dict[Tuple[int, int], _CallState] = {}
        self._last_call: Dict[int, _CallState] = {}
        self._done: Dict[int, set[str]] = {}
        self._generation_slots = asyncio.Semaphore(num_workers)
        self._proactive_prefill = proactive_prefill
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
            if not await self._engine.is_engine_idle() or not self._calls:
                continue

            state = next(reversed(self._calls.values()))
            program = self._programs[state.request.program_name]  # type: ignore[index]
            warmed = False
            for key in program.next_calls(state.site.key): # type: ignore[union-attr]
                for site in program.sites_of(key):
                    if site.callsite_uuid in state.prefilled_sites:
                        continue
                    constants = {
                        name: repr(info["spec"]["value"])
                        for name, info in program.ready_params(
                            key, state.done, via=state.site.key
                        ).items()
                        if info["ready"] and info["spec"]["kind"] == "const"
                    }
                    site.incremental_prompt(constants)
                    prompt = self._engine.render(site.get_ready_prompt())
                    await self._engine.prefill(
                        prompt, f"prefill-{site.callsite_uuid}-{uuid.uuid4().hex}"
                    )
                    state.prefilled_sites.add(site.callsite_uuid)
                    warmed = True
                    break
                if warmed:
                    break

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
                for consumer, parameter in program.consumers_of(site.key):
                    for consumer_site in program.sites_of(consumer):
                        consumer_site.incremental_prompt({parameter: repr(output)})
                        state.prefilled_sites.discard(consumer_site.callsite_uuid)

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
                        f"generate-{request.id}",
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
                    f"generate-{request.id}",
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
