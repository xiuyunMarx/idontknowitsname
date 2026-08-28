import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from static_pass.corpus import SYSTEM_PERSONA, TOOL_INSTRUCTION, ROUTE_ZONE_LABEL, format_tools_for_prompt, route_system_prompt
from static_pass.schema_render import TypeRegistry, schema_entry
from static_pass.state import Binding, InstanceState, bind_params, watch_set


class SourceKind(Enum):
    """Where a byllm argument's value comes from (static provenance)."""
    CONST = "const"        # a literal at the callsite: known at compile time
    FIELD = "field"        # `self.x` / `here.x` / `visitor.x` no byllm call writes: read from live state
    RET = "ret"            # the return of exactly one byllm call (directly, or through a field it writes)
    RET_ITEM = "ret_item"  # one element of a byllm call's returned list (loop variable)
    RET_ANY = "ret_any"    # the return of whichever of several byllm calls ran last (loop / branches)
    CALL = "call"          # a plain function or tool result: only observable after it runs
    UNKNOWN = "unknown"


@dataclass
class ParamSource:
    """Provenance of one argument at one callsite."""
    kind: SourceKind
    value: Any = None                      # CONST: the literal
    producers: List[str] = field(default_factory=list)  # RET / RET_ITEM / RET_ANY: producer callsite keys (decl key if it has no static site)
    scope: str = ""                        # FIELD / RET via a field: "self" | "here" | "visitor"
    attr: str = ""                         # ... the field name
    arch: str = ""                         # ... archetype owning the field
    binding: str = ""                      # ... "walker" | "node" | "obj": which lifetime the field rides on
    slice: Optional[str] = None            # ... `[...]` subscript text, if any
    default: Any = None                    # ... the field's `has` default literal (what it holds before any producer ran)
    has_default: bool = False
    expr: str = ""                         # CALL / UNKNOWN: the source expression text

    @property
    def via_field(self) -> str:
        return f"{self.arch}.{self.attr}" if self.arch and self.attr else ""

    def ready(self, done: "set[str]") -> bool:
        """Derivable now, given the completed callsite keys `done`: constants and
        live fields always, returns once a producer finished."""
        if self.kind in (SourceKind.CONST, SourceKind.FIELD):
            return True
        if self.kind in (SourceKind.RET, SourceKind.RET_ITEM, SourceKind.RET_ANY):
            return any(k in done for k in self.producers)
        return False


@dataclass
class ParamReady:
    """Readiness of one parameter of a consumer callsite, given what has completed."""
    name: str
    ready: bool
    source: ParamSource
    value: Any = None          # the bytes-known value when `known`: a CONST literal, or the
    known: bool = False        # field's `has` default while no producer has run yet

    @property
    def needs_state(self) -> bool:
        """Ready, but the value must be read from live walker/node state (FIELD)."""
        return self.ready and self.source.kind is SourceKind.FIELD and not self.known


@dataclass(frozen=True)
class Consumer:
    """A (callsite, param) this callsite's return feeds."""
    callsite_key: str
    param: str


class GuardKind(Enum):
    CERTAIN = "certain"          # always executes once the enclosing scope runs
    CONDITIONAL = "conditional"  # under if / try / match
    LOOP = "loop"                # inside a loop body


class ByLLMFunc:
    """def f(...) -> T by llm(...)

    One instance per callsite. Holds what the compiler knows about the site,
    the values bound at runtime, and renders the prompt from them."""

    def __init__(self, func_name: str, callsite_key: str, model_name: str):
        # --- declaration ---
        self.func_name: str = func_name
        self.callsite_key: str = callsite_key             # module:line:col, wire identity
        self.scope: Optional[str] = None                  # enclosing ability/archetype; None for module-level
        self.host: str = ""                               # archetype whose ability contains this callsite ("" at module level)
        self.params_type: Dict[str, Any] = {}             # name -> type
        self.return_type: Any = None
        self.response_format: Optional[Dict[str, Any]] = None  # json schema for structured return
        self.tools: List[Dict[str, Any]] = []             # tool schemas, if any

        # --- byLLM semantics ---
        self.func_sem: str = ""
        self.param_sem: Dict[str, str] = {}               # name -> sem
        self.param_schema: Dict[str, str] = {}            # name -> full schema-zone entry (param row + obj/enum member rows with their sems); built by parsing
        self.owner_sem: str = ""                          # sem of the owning archetype
        self.model_name: str = model_name
        self.call_params: Dict[str, Any] = {}             # literal kwargs of llm(...)
        self.extra_system_prompt: str = ""

        # --- dependency ---
        self.param_source: Dict[str, ParamSource] = {}    # param name -> where its value comes from
        self.consumers: List[Consumer] = []               # (consumer callsite_key, param) fed by our return

        # --- guard ---
        self.guard: GuardKind = GuardKind.CERTAIN
        self.guard_pred: Optional[str] = None             # predicate source text, if any

        # --- runtime ---
        self.params_value: Dict[str, Any] = {}            # name -> value bound so far
        self.return_value: Any = None

        self._invariant: Optional[str] = None
        self._system: str = ""
        self._user: str = ""

    def bind(self, name: str, value: Any) -> None:
        self.params_value[name] = value

    def reset(self) -> None:
        self.params_value = {}
        self.return_value = None

    def render_invariant(self) -> str:
        """Compile-time invariant prefix, same bytes as byllm's factory:
        [system] persona + extra system_prompt + tool instruction + tool block
        [user]   qualified signature (--- func_sem) + described-param schema rows
                 (+ response schema, when the return is structured)."""
        if self._invariant is not None:
            return self._invariant
        system = SYSTEM_PERSONA
        if self.extra_system_prompt:
            system += "\n\n" + self.extra_system_prompt
        if self.tools:
            system += TOOL_INSTRUCTION + "\n\n" + format_tools_for_prompt(self.tools)

        args = ", ".join(f"{n}: {t}" if t else n for n, t in self.params_type.items())
        header = f"{self.func_name}({args})"
        if self.return_type:
            header += f" -> {self.return_type}"
        if self.scope:
            header = f"{self.scope}.{header}"
        if self.func_sem:
            header += f" --- {self.func_sem}"
        lines = [header]
        for name, ty in self.params_type.items():
            row = self.param_schema.get(name) or schema_entry(name, ty, self.param_sem.get(name, ""), TypeRegistry())
            for ln in row.split("\n") if row else []:
                lines.append("      " + ln)  # byllm indents every line of the entry
        # response_format is not prompt text: byllm passes it as the API's
        # response_format (the server turns it into a decoding grammar).

        self._system, self._user = system, "\n".join(lines)
        self._invariant = self._system + "\n\n" + self._user
        return self._invariant

    def render_system(self) -> str:
        """[system] zone alone (it has blank lines of its own when tools are present)."""
        self.render_invariant()
        return self._system

    def render_user_invariant(self) -> str:
        """[user] compile-time prefix alone."""
        self.render_invariant()
        return self._user

    def render_full(self, params: Dict[str, Any]) -> str:
        """Full prompt as an extension of the invariant: params already bound keep
        their position (bind order) — a conflicting request value wins in place —
        then the request's remaining params in declaration order, then `self`."""
        lines = [self.render_user_invariant()]
        for name, bound in self.params_value.items():
            lines.append(f"{name} = {params.get(name, bound)}")
        for name in self.params_type:
            if name in params and name not in self.params_value:
                lines.append(f"{name} = {params[name]}")
        zones = ["\n".join(lines)]
        self_view = params.get("self")
        if self_view is not None and self.owner_sem:
            zones.append(f"self = {self_view} ---- {self.owner_sem}")
        return "\n\n".join(zones)


class VisitByLLM:
    """visit [-->] by llm(intent=..., select=...)."""

    def __init__(self, callsite_key: str, model_name: str):
        # --- declaration ---
        self.callsite_key: str = callsite_key             # module:line, wire identity
        self.scope: Optional[str] = None                  # enclosing ability
        self.host: str = ""                               # archetype whose ability contains the visit
        self.walker: str = ""                             # walker archetype running the visit
        self.owner_sem: str = ""                          # sem of the walker

        # --- routing semantics ---
        self.intent: str = ""                             # static intent= text
        self.select: Any = "all"                          # 1 | int | (lo, hi) | "all"
        self.candidates: List[str] = []                   # node archetypes the router may pick
        self.zones: List[str] = []                        # runtime zone order (self / here / edges ...)
        self.model_name: str = model_name
        self.call_params: Dict[str, Any] = {}             # literal kwargs of llm(...)

        # --- guard ---
        self.guard: GuardKind = GuardKind.CERTAIN
        self.guard_pred: Optional[str] = None

        # --- runtime ---
        self.zone_value: Dict[str, str] = {}              # zone name -> rendered text bound so far
        self.chosen: List[str] = []                       # node handles the LLM picked

        self._invariant: Optional[str] = None

    def render_system(self) -> str:
        """route_visit's routing instruction, fixed by `select=` at compile time."""
        return route_system_prompt(self.select)

    def render_invariant(self) -> str:
        """Compile-time user prefix of a routing call: the static `Goal:` line."""
        if self._invariant is None:
            self._invariant = f"Goal: {self.intent}" if self.intent else ""
        return self._invariant

    def render_full(self, zone_values: Dict[str, str]) -> str:
        """Routing prompt as route_visit emits it: invariant, then the bound zones in
        layout order, joined by blank lines."""
        parts = [self.render_invariant()]
        for zone in self.zones:
            if zone in zone_values:
                parts.append(f"{ROUTE_ZONE_LABEL[zone]}\n{zone_values[zone]}")
        return "\n\n".join(p for p in parts if p)

    def full_prompt(self) -> List[Dict[str, str]]:
        """The complete routing call, return chat tamplate Raises ValueError while any zone is still unbound"""
        missing = [zone for zone in self.zones if zone not in self.zone_value]
        if missing:
            raise ValueError(f"{self.callsite_key}: zones {missing} not bound yet")
        return [{"role": "system", "content": self.render_system()},
                {"role": "user", "content": self.render_full(self.zone_value)}]
    
# Generation engine the Program drives: (model_name, messages, response_format | None,
# call_params) -> assistant text. Plugged in by the server layer (InstanceEngine).
Engine = Callable[[str, List[Dict[str, str]], Optional[Dict[str, Any]], Dict[str, Any]], str]

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class _Call:
    """One in-flight `call` of the client: the server-driven ReAct sub-conversation."""
    id: int
    site: ByLLMFunc
    model_name: str
    messages: List[Dict[str, str]]
    schema: Optional[Dict[str, Any]]
    call_params: Dict[str, Any]
    turns: int = 0
    rejects: int = 0


class Program:
    """The static picture of one .jac program (callsites, topology, provenance,
    schemas) wired to one running instance of it: the InterceptorLLM connects here,
    pushes field observations, and hands every `by llm()` call over with its
    model name; the Program renders the prompt, drives the engine, runs the tool
    loop, and keeps every consumer's bindings current."""

    def __init__(self, name: str):
        self.name: str = name
        self.callsites: Dict[str, Union[ByLLMFunc, VisitByLLM]] = {}
        self.next_call: Dict[str, List[str]] = {}  # callsite_key -> list of next callsite_keys
        self.types: Any = None  # schema_render.TypeRegistry: the module's obj/enum declarations

        # --- the bound runtime instance ---
        self.instance_address: Tuple[str, int] = ("localhost", 8964)
        self.link: Any = None                       # link.InstanceLink once wired
        self.engine: Optional[Engine] = None        # generation backend, keyed by model name per call
        self.state = InstanceState()                # observed walker/node fields (state / enter frames)
        self.done: "set[str]" = set()               # callsites that completed at least once
        self.produced: Dict[str, str] = {}          # callsite_key -> repr of its last output
        self.calls: Dict[int, _Call] = {}           # in-flight calls by wire id
        self.max_output_retries: int = 2
        self.max_react_iterations: int = 8

    def build_program(self, path: str) -> "Program":
        from static_pass.parsing import build  # parsing imports the primitives; keep this lazy
        return build(self, path)

    # ------------------------------------------------------------------ wiring

    def wire_instance(self, host: str, port: int, engine: Optional[Engine] = None) -> "Program":
        """Listen for the one InterceptorLLM instance this Program serves. `engine`
        generates text for (model_name, messages, schema, call_params)."""
        from static_pass.link import InstanceLink
        if engine is not None:
            self.engine = engine
        self.link = InstanceLink(self, host, port).listen()
        self.instance_address = (host, self.link.port)
        return self

    def on_register(self, hello: dict) -> dict:
        """The `registered` ack: the observation watch set and the routing layout."""
        self.state = InstanceState()
        self.done, self.produced, self.calls = set(), {}, {}
        for site in self.callsites.values():
            site.reset() #type: ignore
        return {"type": "registered", "route_layout": "cache", "watch": watch_set(self)}

    def on_disconnect(self) -> None:
        self.calls.clear()

    def on_frame(self, frame: dict) -> None:
        """Every frame after `register`, in wire order."""
        if self.state.apply(frame):
            self.refresh_bindings()
            return
        kind = frame.get("type")
        if kind == "call":
            self._begin_call(frame)
        elif kind == "tool_result":
            self._continue_call(int(frame["call"]), {"role": "user", "content": str(frame.get("content", ""))})
        elif kind == "reject":
            self._reject_call(int(frame["call"]), str(frame.get("feedback", "")))
        elif kind == "generate":
            self._generate(frame)
        else:
            self._send({"type": "error", "id": frame.get("id"), "error": f"unknown frame type {kind!r}"})

    def _send(self, frame: dict) -> None:
        if self.link is not None:
            self.link.send(frame)

    # ------------------------------------------------------------ call loop

    def site_of(self, key: str, site: Optional[str]) -> Optional[Union[ByLLMFunc, VisitByLLM]]:
        """The callsite a wire (key, site) names: `Owner.name` + `file.jac:line`."""
        if key in self.callsites:
            return self.callsites[key]
        cands = [s for k, s in self.callsites.items() if k.rsplit("@", 1)[0] == key]
        if site and len(cands) > 1:
            line = site.rpartition(":")[2]
            for s in cands:
                if s.callsite_key.rsplit("@", 1)[1].split(":")[0] == line:
                    return s
        return cands[0] if cands else None

    def _begin_call(self, frame: dict) -> None:
        site = self.site_of(str(frame.get("key", "")), frame.get("site"))
        if not isinstance(site, ByLLMFunc):
            self._send({"type": "error", "id": frame.get("id"), "error": f"unknown callsite {frame.get('key')!r}"})
            return
        args = dict(frame.get("args") or {})
        self_view = frame.get("self")
        params = dict(args)
        if self_view is not None:
            params["self"] = self_view
        messages = [{"role": "system", "content": site.render_system()},
                    {"role": "user", "content": site.render_full(params)}]
        call = _Call(int(frame["id"]), site, str(frame.get("model_name") or (self.link.model_name if self.link else "")),
                     messages, frame.get("schema"), dict(frame.get("call_params") or {}))
        self.calls[call.id] = call
        self._step(call)

    def _continue_call(self, call_id: int, message: Dict[str, str]) -> None:
        call = self.calls.get(call_id)
        if call is None:
            self._send({"type": "error", "call": call_id, "error": "no such call"})
            return
        call.messages.append(message)
        self._step(call)

    def _reject_call(self, call_id: int, feedback: str) -> None:
        call = self.calls.get(call_id)
        if call is None:
            return
        call.rejects += 1
        call.messages.append({"role": "user", "content": feedback})
        self._step(call)

    def _step(self, call: _Call) -> None:
        """One engine turn: a tool call goes back to the client, anything else is final."""
        if self.engine is None:
            self._send({"type": "error", "call": call.id, "error": "program has no engine"})
            return
        try:
            text = self.engine(call.model_name, list(call.messages), call.schema, call.call_params)
        except Exception as e:  # the engine's failure is the client's error
            self._send({"type": "error", "call": call.id, "error": str(e)})
            self.calls.pop(call.id, None)
            return
        call.messages.append({"role": "assistant", "content": text})
        call.turns += 1
        site = call.site
        max_iter = int(call.call_params.get("max_react_iterations") or self.max_react_iterations)
        tool = self._parse_tool_call(text) if site.tools else None
        if tool is not None and tool["name"] != "finish_tool" and call.turns < max_iter:
            self._send({"type": "tool_call", "call": call.id, "name": tool["name"],
                        "arguments": tool.get("arguments", {}), "text": text})
            return
        output = (tool.get("arguments", {}).get("final_output", text) if tool is not None else text)
        self._finish(call, output, text)

    def _finish(self, call: _Call, output: Any, text: str) -> None:
        site = call.site
        self._send({"type": "final", "call": call.id, "output": output, "text": text})
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

    def _generate(self, frame: dict) -> None:
        """Single-turn `generate` (visit routing): the client ships full messages."""
        if self.engine is None:
            self._send({"type": "error", "id": frame.get("id"), "error": "program has no engine"})
            return
        site = self.site_of(str(frame.get("key", "")), frame.get("site"))
        if not isinstance(site, VisitByLLM):
            self._send({"type": "error", "id": frame.get("id"),
                        "error": f"unknown routing callsite {frame.get('key')!r} at {frame.get('site')!r}"})
            return
        params = {k: frame[k] for k in ("temperature", "max_tokens", "stop") if frame.get(k) is not None}
        try:
            text = self.engine(str(frame.get("model_name") or ""), list(frame.get("messages") or []),
                               frame.get("schema"), params)
        except Exception as e:
            self._send({"type": "error", "id": frame.get("id"), "error": str(e)})
            return
        self._send({"type": "result", "id": frame.get("id"), "text": text})
        self.done.add(site.callsite_key)
        self.refresh_bindings(via=site.callsite_key)

    # --------------------------------------------------------------- binding

    def bind_ready(self, consumer_key: str, via: Optional[str] = None) -> Dict[str, Binding]:
        """Bytes of every param of `consumer_key` bindable now (static readiness +
        produced outputs + observed fields)."""
        pid = self.link.pid if self.link is not None else -1
        return bind_params(self, consumer_key, self.done, self.state, pid, via, self.produced)

    def refresh_bindings(self, via: Optional[str] = None) -> None:
        """Push the current bindable bytes into every ByLLMFunc's `params_value` (in
        bind order). Called after each observation and each completed call."""
        for site in self.byLLMs:
            for name, b in self.bind_ready(site.callsite_key, via).items():
                if site.params_value.get(name) != b.text:
                    site.bind(name, b.text)

    # ---------------------------------------------------------------- queries

    def successors(self, callsite_key: str) -> List[Any]:
        """The callsites that may run next after `callsite_key` (topology edge)."""
        return [self.callsites[k] for k in self.next_call.get(callsite_key, []) if k in self.callsites]

    def ready_params(self, consumer_key: str, done: "set[str]",
                     via: Optional[str] = None) -> Dict[str, ParamReady]:
        consumer = self.callsites.get(consumer_key)
        if not isinstance(consumer, ByLLMFunc):
            return {}
        crossing = isinstance(self.callsites.get(via), VisitByLLM) if via else False
        out: Dict[str, ParamReady] = {}
        for name, src in consumer.param_source.items():
            ready = src.ready(done)
            known, value = False, None
            if src.kind is SourceKind.CONST:
                known, value = True, src.value
            elif src.kind in (SourceKind.RET, SourceKind.RET_ANY) and src.has_default \
                    and not any(k in done for k in src.producers):
                known, value, ready = True, src.default, True
            if crossing and src.binding == "node":
                ready, known = False, False
            out[name] = ParamReady(name, ready, src, value, known)
        return out

    def successor_readiness(self, callsite_key: str, done: "set[str]") -> Dict[str, Dict[str, ParamReady]]:
        """For each byllm successor of `callsite_key`: its params' readiness once
        `callsite_key` itself has completed (it is added to `done`)."""
        done = set(done) | {callsite_key}
        return {s.callsite_key: self.ready_params(s.callsite_key, done, via=callsite_key)
                for s in self.successors(callsite_key) if isinstance(s, ByLLMFunc)}

    @property
    def byLLMs(self) -> List["ByLLMFunc"]:
        return [s for s in self.callsites.values() if isinstance(s, ByLLMFunc)]

    @property
    def visitBys(self) -> List["VisitByLLM"]:
        return [s for s in self.callsites.values() if isinstance(s, VisitByLLM)]


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    prog = Program("test").build_program(path)
    breakpoint()