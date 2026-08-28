"""One running process of a Program: its connections (one per InterceptorLLM
object), observed state, bindings, in-flight calls and the predicted model
sequence ahead. Sites are deep-copied from the template so bound values never
leak between instances."""
import asyncio
import copy
import json
import re
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from static_pass.link import Connection
from static_pass.primitives import ByLLMFunc, Program, RequestHandle, VisitByLLM
from static_pass.state import Binding, InstanceState, bind_params, watch_set

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class _Call:
    """One in-flight `call` of the client: the server-driven ReAct sub-conversation."""
    id: int
    conn: Connection
    site: ByLLMFunc
    model_name: str
    messages: List[Dict[str, str]]
    schema: Optional[Dict[str, Any]]
    call_params: Dict[str, Any]
    turns: int = 0
    rejects: int = 0


class ProgramInstance:
    def __init__(self, program: Program, server: Any, pid: int):
        self.program, self.server, self.pid = program, server, pid  # server has `async submit(RequestHandle)`
        self.sites: Dict[str, Union[ByLLMFunc, VisitByLLM]] = copy.deepcopy(program.callsites)
        self.state = InstanceState()              # observed walker/node fields (state / enter frames)
        self.done: "set[str]" = set()             # callsites that completed at least once
        self.produced: Dict[str, str] = {}        # callsite_key -> repr of its last output
        self.calls: Dict[int, _Call] = {}         # in-flight calls by wire id
        self.expected_path: List[str] = []        # callsites sure to follow the current call (to the first divergence)
        self.expected: List[str] = []             # ... their models, consecutive duplicates merged (planner input)
        self.guesses: List[Tuple[str, str, float]] = []  # (callsite, model, prob) possible after the divergence
        self.connections = 0

    async def serve(self, conn: Connection, hello: dict) -> None:
        """One InterceptorLLM connection of this process, until it drops."""
        self.connections += 1
        conn.send({"type": "registered", "route_layout": "cache", "watch": watch_set(self.program)})
        try:
            while True:
                self.on_frame(await conn.recv(), conn)
        finally:
            self.connections -= 1
            if self.connections == 0:
                self.calls.clear()

    def on_frame(self, frame: dict, conn: Connection) -> None:
        if self.state.apply(frame):
            self.refresh_bindings()
            return
        kind = frame.get("type")
        if kind == "call":
            self._begin_call(frame, conn)
        elif kind == "tool_result":
            self._continue_call(int(frame["call"]), {"role": "user", "content": str(frame.get("content", ""))}, "tool_turn", conn)
        elif kind == "reject":
            self._reject_call(int(frame["call"]), str(frame.get("feedback", "")), conn)
        elif kind == "generate":
            self._spawn(self._generate(frame, conn), conn, {"id": frame.get("id")})
        else:
            conn.send({"type": "error", "id": frame.get("id"), "error": f"unknown frame type {kind!r}"})

    def _spawn(self, coro, conn: Connection, ref: dict) -> None:
        """Run a call step; a server-side bug becomes an error frame, never a silent hang."""
        async def guarded():
            try:
                await coro
            except Exception as e:
                traceback.print_exc()
                if "call" in ref:
                    self.calls.pop(ref["call"], None)
                conn.send({"type": "error", **ref, "error": f"server bug: {e!r}"})
        asyncio.create_task(guarded())

    def site_of(self, key: str, site: Optional[str]) -> Optional[Union[ByLLMFunc, VisitByLLM]]:
        tpl = self.program.site_of(key, site)
        return self.sites[tpl.callsite_key] if tpl is not None else None

    # ------------------------------------------------------------ call loop

    def _begin_call(self, frame: dict, conn: Connection) -> None:
        site = self.site_of(str(frame.get("key", "")), frame.get("site"))
        if not isinstance(site, ByLLMFunc):
            conn.send({"type": "error", "id": frame.get("id"), "error": f"unknown callsite {frame.get('key')!r}"})
            return
        params = dict(frame.get("args") or {})
        if frame.get("self") is not None:
            params["self"] = frame["self"]
        messages = [{"role": "system", "content": site.render_system()},
                    {"role": "user", "content": site.render_full(params)}]
        model = self.program.model_of(site, str(frame.get("model_name") or ""))
        call = _Call(int(frame["id"]), conn, site, model, messages, frame.get("schema"),
                     {**site.call_params, **(frame.get("call_params") or {})})  # literal llm(...) kwargs + wire
        self.calls[call.id] = call
        self._predict(site.callsite_key, model)
        self._spawn(self._step(call, "call"), conn, {"call": call.id})

    def _continue_call(self, call_id: int, message: Dict[str, str], kind: str, conn: Connection) -> None:
        call = self.calls.get(call_id)
        if call is None:
            conn.send({"type": "error", "call": call_id, "error": "no such call"})
            return
        call.messages.append(message)
        self._spawn(self._step(call, kind), conn, {"call": call.id})

    def _reject_call(self, call_id: int, feedback: str, conn: Connection) -> None:
        call = self.calls.get(call_id)
        if call is None:
            return
        call.rejects += 1
        self._continue_call(call_id, {"role": "user", "content": feedback}, "reject", conn)

    async def _request(self, handle: RequestHandle) -> str:
        """Hand `handle` to the server and wait for its text; raises on server error."""
        await self.server.submit(handle)
        await handle.done.wait()
        if handle.error is not None:
            raise RuntimeError(handle.error)
        return handle.text

    def salt(self) -> str:
        return f"{self.program.name}:{self.pid}"

    def _predict(self, callsite_key: str, model: str) -> None:
        self.expected_path = self.program.certain_chain(callsite_key)
        self.expected = []
        for key in self.expected_path:
            m = self.program.model_of(self.sites[key], model)
            if not self.expected or self.expected[-1] != m:
                self.expected.append(m)
        self.guesses = [(key, self.program.model_of(self.sites[key], model), p)
                        for key, p in self.program.branch_candidates(callsite_key)]

    async def _step(self, call: _Call, kind: str) -> None:
        """One engine turn: a tool call goes back to the client, anything else is final."""
        try:
            text = await self._request(RequestHandle(
                self, call.site.callsite_key, call.model_name, list(call.messages), kind,
                call.schema, call.call_params, self.salt()))
        except Exception as e:  # the server's failure is the client's error
            call.conn.send({"type": "error", "call": call.id, "error": str(e)})
            self.calls.pop(call.id, None)
            return
        call.messages.append({"role": "assistant", "content": text})
        call.turns += 1
        max_iter = int(call.call_params.get("max_react_iterations") or self.program.max_react_iterations)
        tool = self._parse_tool_call(text) if call.site.tools else None
        if tool is not None and tool["name"] != "finish_tool" and call.turns < max_iter:
            call.conn.send({"type": "tool_call", "call": call.id, "name": tool["name"],
                           "arguments": tool.get("arguments", {}), "text": text})
            return
        output = (tool.get("arguments", {}).get("final_output", text) if tool is not None else text)
        self._finish(call, output, text)

    def _finish(self, call: _Call, output: Any, text: str) -> None:
        site = call.site
        call.conn.send({"type": "final", "call": call.id, "output": output, "text": text})
        self.calls.pop(call.id, None)
        site.return_value = output
        self.produced[site.callsite_key] = repr(output)  # the client's typed value reprs the same way
        self.done.add(site.callsite_key)
        self.refresh_bindings(via=site.callsite_key)

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
        m = _TOOL_CALL_RE.search(text)
        payload = m.group(1) if m else text.strip()
        try:
            call, _ = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return None
        if not isinstance(call.get("arguments", {}), dict):
            return None
        return call

    async def _generate(self, frame: dict, conn: Connection) -> None:
        """Single-turn `generate` (visit routing): the client ships full messages."""
        site = self.site_of(str(frame.get("key", "")), frame.get("site"))
        if not isinstance(site, VisitByLLM):
            conn.send({"type": "error", "id": frame.get("id"),
                       "error": f"unknown routing callsite {frame.get('key')!r} at {frame.get('site')!r}"})
            return
        params = {k: frame[k] for k in ("temperature", "max_tokens", "stop") if frame.get(k) is not None}
        model = self.program.model_of(site, str(frame.get("model_name") or ""))
        self._predict(site.callsite_key, model)
        try:
            text = await self._request(RequestHandle(
                self, site.callsite_key, model, list(frame.get("messages") or []),
                "generate", frame.get("schema"), params, self.salt()))
        except Exception as e:
            conn.send({"type": "error", "id": frame.get("id"), "error": str(e)})
            return
        conn.send({"type": "result", "id": frame.get("id"), "text": text})
        self.done.add(site.callsite_key)
        self.refresh_bindings(via=site.callsite_key)

    # --------------------------------------------------------------- binding

    def bind_ready(self, consumer_key: str, via: Optional[str] = None) -> Dict[str, Binding]:
        """Bytes of every param of `consumer_key` bindable now (static readiness +
        produced outputs + observed fields)."""
        return bind_params(self.program, consumer_key, self.done, self.state, self.pid, via, self.produced)

    def refresh_bindings(self, via: Optional[str] = None) -> None:
        """Push the current bindable bytes into every ByLLMFunc's `params_value` (in
        bind order). Called after each observation and each completed call."""
        for site in self.sites.values():
            if not isinstance(site, ByLLMFunc):
                continue
            for name, b in self.bind_ready(site.callsite_key, via).items():
                if site.params_value.get(name) != b.text:
                    site.bind(name, b.text)
