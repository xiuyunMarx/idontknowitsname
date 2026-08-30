import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from static_pass.corpus import SYSTEM_PERSONA, TOOL_INSTRUCTION, ROUTE_ZONE_LABEL, format_tools_for_prompt, route_system_prompt
from static_pass.schema_render import TypeRegistry, schema_entry


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
                lines.append("      " + ln) 

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
    
@dataclass(eq=False)  # identity: handles live in sets
class RequestHandle:
    """One generation turn, shared between the Program that needs the text and the
    server that produces it. The Program awaits `done`; the server fills `text`
    or `error` and sets it. `program` + `callsite_key` give the server the
    compile-time picture (successors, bound params) for speculation."""
    instance: Any                            # instance.ProgramInstance
    callsite_key: str
    model_name: str
    messages: List[Dict[str, str]]
    kind: str = "call"                       # call | tool_turn | reject | generate
    schema: Optional[Dict[str, Any]] = None  # structured-output grammar
    call_params: Dict[str, Any] = field(default_factory=dict)
    cache_salt: str = ""                     # prefix-cache tenant; speculation must use the same
    text: str = ""
    error: Optional[str] = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    # live progress, written by the engine while the request runs (planner input)
    first_token_at: float = 0.0              # perf_counter of the first token; 0 while prefilling
    out_tokens: int = 0
    # lifecycle, written by the controller (queue wait = dispatched - created)
    created_at: float = 0.0                  # perf_counter at submit
    dispatched_at: float = 0.0               # perf_counter when its model was resident and it was released
    done_at: float = 0.0


class Program:
    """The static picture of one .jac program (callsites, topology, provenance,
    schemas). A template: every connected process gets an instance.ProgramInstance
    that binds values, drives the tool loop and tracks progress on its own copy."""

    def __init__(self, name: str):
        self.name: str = name
        self.callsites: Dict[str, Union[ByLLMFunc, VisitByLLM]] = {}
        self.next_call: Dict[str, List[str]] = {}  # callsite_key -> list of next callsite_keys
        self.types: Any = None  # schema_render.TypeRegistry: the module's obj/enum declarations
        self.unresolved_models: "set[str]" = set()  # llm variables whose model_name is not a literal
        self.max_react_iterations: int = 8
        
        # branch prediction: callsite_key -> {successor callsite_key: observed frequency},
        # keyed by the site whose successor is being chosen (prior 1 per topology edge)
        self.branch_freq: Dict[str, Dict[str, int]] = {}

    def build_program(self, path: str) -> "Program":
        from static_pass.parsing import build  # parsing imports the primitives; keep this lazy
        program = build(self, path)
        program.branch_freq = {k: {s: 1 for s in succ} for k, succ in program.next_call.items() if succ}
        return program


    # ---------------------------------------------------------------- lookup

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

    def model_of(self, site: Union[ByLLMFunc, VisitByLLM], fallback: str) -> str:
        """The served model of `site`: its literal model_name, else what the client reported."""
        return fallback if site.model_name in self.unresolved_models else site.model_name

    def update_branch_probs(self, callsite_key: str, selected: str) -> None:
        """Count one observed transition `callsite_key` -> `selected`."""
        if callsite_key not in self.branch_freq:
            raise ValueError(f"Callsite {callsite_key} has no successors in the static topology.")
        if selected not in self.branch_freq[callsite_key]:
            raise ValueError(f"Selected callsite {selected} is not a successor of {callsite_key}.")
        self.branch_freq[callsite_key][selected] += 1

    def branch_candidates(self, callsite_key: str) -> List[Tuple[str, float]]:
        """Successors of `callsite_key` by observed frequency, most frequent first."""
        freq = self.branch_freq.get(callsite_key, {})
        total = sum(freq.values())
        return sorted(((k, n / total) for k, n in freq.items()), key=lambda t: -t[1]) if total else []

    def predict_path(self, callsite_key: str) -> List[str]:
        """Most frequent successor at every step from `callsite_key`, until no successor
        or a site already on the path."""
        path: List[str] = []
        seen = {callsite_key}
        key = callsite_key
        while True:
            cands = self.branch_candidates(key)
            if not cands or cands[0][0] in seen:
                return path
            key = cands[0][0]
            seen.add(key)
            path.append(key)

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