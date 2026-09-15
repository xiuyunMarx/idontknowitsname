"""Primitives of the compiler-server co-design.

Everything the compiler knows about a `by llm` call is fixed text plus a few
typed bindings; the server only ever sees values. The split is:

  PromptTemplate  one callsite: the constant bytes and the binding list, with
                  each binding's heterogeneity and lifetime from static analysis.
  Program         the callsites as a directed graph. Edges are the compiler's
                  visit graph; the server marks them with frequencies and times.
  CallInstance    one request: a template plus the binding values it carried.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from statistics import median
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


def _quantile(xs: List[float], q: float) -> float:
    """Empirical quantile of raw samples; 0.0 when empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


# ------------------------------------------------------------------ timing

@dataclass
class CallStats:
    '''Timing Stats for a single call to a LLM functions'''
    engine_time: List[List[float]] = field(default_factory=list)
    gap: List[List[float]] = field(default_factory=list)         # Time between requests, seconds.
    num_turns: List[int] = field(default_factory=list)         # HTTP requests per call.

    def add_record(self, engine_time: List[float], num_turns: int = 1, gap: List[float] = []) -> None:
        if len(engine_time) != num_turns or len(gap) != num_turns - 1:
            raise ValueError("Inconsistent recordings")
        self.engine_time.append(engine_time)
        self.num_turns.append(num_turns)
        self.gap.append(gap)

    def duration(self) -> float:
        '''e2e duration of the call, including gaps between requests'''
        e2e_times: List[float] = [sum(engine_time) + sum(gap) for engine_time, gap in zip(self.engine_time, self.gap)]
        return median(e2e_times) if e2e_times else 0.0

    def engine_duration(self) -> float:
        '''Sum of engine times of one call, excluding gaps between requests'''
        engine_time = [sum(engine_time) for engine_time in self.engine_time]
        return median(engine_time) if engine_time else 0.0

    def quantile_duration(self, quantile: float = 0.5) -> float:
        '''Quantile of e2e duration of the call, including gaps between requests'''
        e2e_times: List[float] = [sum(engine_time) + sum(gap) for engine_time, gap in zip(self.engine_time, self.gap)]
        return _quantile(e2e_times, quantile) if e2e_times else 0.0


# ------------------------------------------------------------------ call site

@dataclass(frozen=True)
class CallSiteID:
    """One call expression of a `by llm` function: what static analysis emits
    and what InterceptorLLM puts on every request as `callsite`."""
    signature: str          # qualified function name, e.g. "Planner.plan"
    lineno: int             # line of the call expression, not of the def
    file: str = ""          # source file as the compiler saw it

    def __repr__(self) -> str:
        return f"{self.signature}:{self.lineno}"

    @classmethod
    def from_request(cls, body: Dict[str, Any]) -> Optional["CallSiteID"]:
        """The `callsite` field of an InterceptorLLM request, or None."""
        cs = body.get("callsite")
        if not isinstance(cs, dict) or "signature" not in cs or "lineno" not in cs:
            return None
        return cls(str(cs["signature"]), int(cs["lineno"]), str(cs.get("file", "")))


# ------------------------------------------------------------------ bindings

class Heterogeneity(Enum):
    """Where a binding's value comes from, decided by the compiler."""
    CONST = auto()     # a literal of the program: the same bytes in every session
    COPY = auto()      # the same bytes as an earlier value of the session (own site or another)
    EXTEND = auto()    # an append-only container: the earlier value minus its closing delimiter is a prefix
    RESP = auto()      # a JSON path into an earlier site's reply
    TAKE = auto()      # another site's binding under a different name
    VOLATILE = auto()  # fresh every call, or from a channel the compiler cannot see through


LAYOUT_RANK = {Heterogeneity.CONST: 0, Heterogeneity.COPY: 1, Heterogeneity.TAKE: 1,
               Heterogeneity.RESP: 2, Heterogeneity.EXTEND: 3, Heterogeneity.VOLATILE: 4}


class BindingKind(Enum):
    PARAM = auto()       # `name = repr(value)` of a by-llm parameter
    SELF = auto()        # `self = repr(owner) ---- sem`, followed by the constant member rows


@dataclass
class Binding:
    '''Binding information of a callsite'''
    name: str
    heterogeneity: Heterogeneity = Heterogeneity.VOLATILE
    kind: BindingKind = BindingKind.PARAM
    source: Optional[Tuple[CallSiteID, str]] = None   # COPY/TAKE: (site, binding name); RESP: (site, json path)
    literal: Optional[str] = None              # CONST: the bytes
    field: str = ""                            # walker field the value carries ("f[]": one element of it; "" none)
    scope: str = ""                            # the CFG scope whose exit invalidates the value ("" = session)
    label: str = ""                            # bytes written before the value, e.g. "spec = "
    tail: str = ""                             # constant bytes written after the value, e.g. self's member rows

    @property
    def shared(self) -> bool:
        """The value is carried by another callsite of the same session."""
        return self.source is not None and self.heterogeneity in (
            Heterogeneity.COPY, Heterogeneity.TAKE, Heterogeneity.EXTEND)


# ------------------------------------------------------------------ template

@dataclass
class PromptTemplate:
    """
    One byLLM function's prompt template, callsites of same byLLM functions share the same template but own different instances.
    
    Built once by the compiler. The server decides `order` and `header_last`
    (the re-layout) from the bindings' heterogeneity, then renders and
    speculates with the template;
    """
    key: CallSiteID                            # the call expression this template belongs to
    ability: str = ""                          # the walker ability the call sits in, e.g. "Planner.start" (binding scopes name these)
    resp_spec: Optional[Dict[str, Any]] = None # structure of the return type, to render a reply's repr (see type_spec)
    model: str = ""
    call_params: Dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, ...
    system_prompt: str = ""                    # bytes before the user message, never re-laid-out
    tool_schema: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    header: str = ""                           # signature line + schema rows
    bindings: List[Binding] = field(default_factory=list)   # native order
    hint: str = ""                             # "Schema requirements:" block; "" for str returns and ReAct sites
    hint_as_part: bool = False                 # appended as a separate text part (no "\n\n" before it)
    # ---- re-layout policy, decided by the server once from the bindings ----
    order: Optional[List[str]] = None          # param names in the served order; None = native
    header_last: bool = False                  # header block behind the values
    no_header_last: bool = False               # program override: header must lead
    # ---- runtime marks ----
    stats: CallStats = field(default_factory=CallStats)

    # Lookup
    def binding(self, name: str) -> Binding:
        for b in self.bindings:
            if b.name == name:
                return b
        raise KeyError(name)

    @property
    def params(self) -> List[Binding]:
        return [b for b in self.bindings if b.kind is BindingKind.PARAM]

    @property
    def is_react(self) -> bool:
        return bool(self.tool_schema)

    # Layout
    def decide_layout(self) -> None:
        """Order the params so the bytes most likely already cached lead"""
        names = [b.name for b in self.params]
        ranked = sorted(names, key=lambda n: (0 if self.binding(n).shared else 1,
                                              0 if self.binding(n).scope == "" else 1,
                                              LAYOUT_RANK[self.binding(n).heterogeneity]))
        self.order = ranked
        self.header_last = bool(ranked) and self.binding(ranked[0]).shared and not self.no_header_last

    def served_order(self) -> List[str]:
        return list(self.order) if self.order is not None else [b.name for b in self.params]

    # Rendering
    def render(self, values: Dict[str, str]) -> str:
        """The user message byllm sends for `values`, in the served layout. The hint
        is the dispatcher's, appended by the caller with `hint_join`."""
        lines = [f"{self.binding(n).label}{values[n]}" for n in self.served_order()]
        body = "\n".join(lines)
        for b in self.bindings:
            if b.kind is BindingKind.SELF and b.name in values:
                body += "\n\n" + f"{b.label}{values[b.name]}" + b.tail
        if not self.header:
            return body
        # header_last: the whole header block moves behind the values, 
        return body + "\n" + self.header if self.header_last else self.header + "\n" + body

    def hint_join(self, user_text: str) -> str:
        """The user message as the engine sees it: byllm sends the hint as a second
        text part, which the chat renderer joins with one newline; a string body
        gets it after a blank line."""
        if not self.hint:
            return user_text
        return user_text + ("\n" if self.hint_as_part else "\n\n") + self.hint

    def strip_hint(self, user_text: str) -> str:
        """The user message without the dispatcher's schema hint."""
        if not self.hint:
            return user_text
        if user_text.endswith(self.hint):
            return user_text[:-len(self.hint)].rstrip("\n")
        i = user_text.rfind(self.hint)
        return user_text[:i].rstrip("\n") if i >= 0 else user_text

    def split(self, user_text: str) -> Optional[Dict[str, str]]:
        """Binding values of a request's user message, in whatever layout it arrived
        (native or already re-laid-out): the constant header and self rows are
        located, the rest is cut at the binding labels. None when the message does
        not fit the template."""
        body = self.strip_hint(user_text)
        # header: at the front (native) or at the back (header_last)
        if self.header:
            if body.startswith(self.header + "\n"):
                body = body[len(self.header) + 1:]
            elif body.endswith("\n" + self.header):
                body = body[:-len(self.header) - 1]
            elif body == self.header:
                body = ""
            else:
                return None
        values: Dict[str, str] = {}
        selfb = next((b for b in self.bindings if b.kind is BindingKind.SELF), None)
        if selfb is not None:
            i = body.rfind("\n\n" + selfb.label)
            if i < 0:
                return None
            block = body[i + 2 + len(selfb.label):]
            if selfb.tail and not block.endswith(selfb.tail):
                return None
            values[selfb.name] = block[:len(block) - len(selfb.tail)] if selfb.tail else block
            body = body[:i]
        params = self.params
        if not params:
            return values if not body else None
        # the binding blocks, in the order they appear
        starts: List[Tuple[int, Binding]] = []
        for b in params:
            if body.startswith(b.label):
                starts.append((0, b))
                continue
            j = body.find("\n" + b.label)
            if j < 0:
                return None
            starts.append((j + 1, b))
        starts.sort(key=lambda x: x[0])
        for n, (pos, b) in enumerate(starts):
            end = starts[n + 1][0] - 1 if n + 1 < len(starts) else len(body)
            values[b.name] = body[pos + len(b.label):end]
        return values

    def fixed_head(self) -> str:
        """Bytes of the user message known before any value: the header unless it
        was moved behind the values, then only the leading CONST params."""
        if self.header_last:
            head = []
            for n in self.served_order():
                b = self.binding(n)
                if b.heterogeneity is not Heterogeneity.CONST:
                    break
                head.append(f"{b.label}{b.literal}")
            return "\n".join(head)
        return self.header


# ------------------------------------------------------------------ instance

@dataclass
class CallInstance:
    """One request of a callsite: the binding values it carried and what came back."""
    template: PromptTemplate
    values: Dict[str, str]                     # binding name -> bytes, cut by the request's segment offsets
    t_arrive: float = 0.0
    t_done: float = 0.0
    response: str = ""
    served_ids: List[int] = field(default_factory=list)      # token ids of the served prompt (first turn)
    engine_time: List[float] = field(default_factory=list)   # per HTTP turn
    gap: List[float] = field(default_factory=list)           # between turns

    @property
    def key(self) -> CallSiteID:
        return self.template.key


# ------------------------------------------------------------------ program

@dataclass
class Edge:
    """A visit-graph edge with the server's marks on it."""
    src: CallSiteID
    dst: CallSiteID
    writes: FrozenSet[str] = frozenset()       # walker fields the program writes between the two calls (static)
    overrides: Dict[str, Binding] = field(default_factory=dict)   # dst binding name -> how it resolves along THIS edge
    count: int = 0                             # times the transition was taken
    gap: List[float] = field(default_factory=list)   # seconds from src's reply to dst's arrival
    stats: CallStats = field(default_factory=CallStats)   # dst's timing when reached from src

    def gap_q(self, q: float) -> float:
        return _quantile(self.gap, q)

@dataclass
class Program:
    """The callsites of one agent as a directed graph.

    Nodes and edges come from the compiler (the visit graph). Frequencies and
    times on them come from the server. Nothing else is learned.
    """
    entry: CallSiteID
    sites: Dict[CallSiteID, PromptTemplate] = field(default_factory=dict)
    edges: Dict[Tuple[CallSiteID, CallSiteID], Edge] = field(default_factory=dict)
    exits: FrozenSet[CallSiteID] = frozenset()        # sites after which the session may end
    n_sessions: int = 0

    # Layout
    def decide_layouts(self, header_last: bool = True) -> None:
        """Choose every site's served order together, so sites that carry the same
        walker fields lay them out in the same relative order and their prompts
        share a byte prefix. A field counts as shared when two or more sites carry
        it (the producer included: its prompt's head is what the consumers reuse).
        Shared fields lead, ordered by a program-wide key: session-scoped before
        scope-reset ones (a value that survives this site's own repeated calls
        beats one a loop iteration replaces), constants before copies before
        append-only histories. Unshared values follow, most stable first. The
        header moves behind the values only where the leading value is shared."""
        carriers: Dict[str, set] = {}
        scoped: Dict[str, bool] = {}
        kind: Dict[str, int] = {}
        for t in self.sites.values():
            for b in t.params:
                if not b.field:
                    continue
                carriers.setdefault(b.field, set()).add(t.key)
                scoped[b.field] = scoped.get(b.field, False) or b.scope != ""
                # the field's own kind: append-only if any site sees it extend, else
                # a copy; a producer's VOLATILE or a reset's CONST view does not
                # describe the field
                k = 3 if b.heterogeneity is Heterogeneity.EXTEND else 1
                kind[b.field] = max(kind.get(b.field, 1), k)

        for t in self.sites.values():
            def key(n: str, t: PromptTemplate = t) -> Tuple[int, int, int]:
                b = t.binding(n)
                f = b.field
                if f and len(carriers.get(f, ())) >= 2:
                    return (0, 1 if scoped[f] else 0, kind[f])
                return (1, 1 if b.scope else 0, LAYOUT_RANK[b.heterogeneity])
            names = [b.name for b in t.params]
            t.order = sorted(names, key=key)
            lead = t.binding(t.order[0]) if t.order else None
            t.header_last = (header_last and lead is not None and bool(lead.field)
                             and len(carriers.get(lead.field, ())) >= 2 and not t.no_header_last)

    # Static shape
    def add_site(self, t: PromptTemplate) -> PromptTemplate:
        self.sites[t.key] = t
        return t

    def add_edge(self, src: CallSiteID, dst: CallSiteID, writes: Optional[FrozenSet[str]] = None,
                 overrides: Optional[Dict[str, Binding]] = None) -> Edge:
        e = self.edges.get((src, dst))
        if e is None:
            e = self.edges[(src, dst)] = Edge(src, dst)
        if writes:
            e.writes = e.writes | frozenset(writes)
        if overrides:
            e.overrides.update(overrides)
        return e

    def resets(self) -> Dict[str, FrozenSet[str]]:
        """ability -> walker fields whose values die when that ability runs (the
        bindings' scopes, inverted): the runtime invalidation table."""
        out: Dict[str, set] = {}
        for t in self.sites.values():
            for b in t.bindings:
                base = b.field[:-2] if b.field.endswith("[]") else b.field
                if not base or not b.scope:
                    continue
                for ab in b.scope.split("|"):
                    out.setdefault(ab, set()).add(base)
        return {k: frozenset(v) for k, v in out.items()}

    def successors(self, key: CallSiteID) -> List[Edge]:
        return [e for (s, _), e in self.edges.items() if s == key]

    # Runtime marks
    def observe(self, prev: Optional[CallInstance], cur: CallInstance) -> None:
        """Mark one transition: the edge count, the gap before `cur`, and `cur`'s
        timing both on its site and on the edge it was reached by."""
        turns = len(cur.engine_time)
        cur.template.stats.add_record(cur.engine_time, turns, cur.gap)
        if prev is None:
            return
        e = self.add_edge(prev.key, cur.key)
        e.count += 1
        e.gap.append(max(0.0, cur.t_arrive - prev.t_done))
        e.stats.add_record(cur.engine_time, turns, cur.gap)

    def gap_q(self, src: CallSiteID, dst: CallSiteID, q: float) -> float:
        """Quantile of the agent's own time between src's reply and dst's arrival."""
        e = self.edges.get((src, dst))
        return e.gap_q(q) if e is not None and e.gap else 0.0

    def duration_q(self, src: Optional[CallSiteID], dst: CallSiteID, q: float) -> float:
        """Quantile of dst's end-to-end duration: by in-edge when that edge has
        samples, else the site's own."""
        e = self.edges.get((src, dst)) if src is not None else None
        if e is not None and e.stats.engine_time:
            return e.stats.quantile_duration(q)
        t = self.sites.get(dst)
        return t.stats.quantile_duration(q) if t is not None and t.stats.engine_time else 0.0

    def predict_tree(self, cur: CallSiteID, p_min: float = 0.02, horizon_s: float = 120.0,
                     max_nodes: int = 64, top_k: int = 3, max_depth: int = 8) -> List["PredictedCall"]:
        """Fan-out prediction from the current site over the static graph: every
        step expands the top_k successors by edge frequency, arrival times add the
        edge gap and the predecessor's duration. Zero counts fall back to a uniform
        prior over the static successors, so the first session is planned too."""
        from math import sqrt
        out: Dict[CallSiteID, PredictedCall] = {}
        frontier: List[Tuple[float, CallSiteID, Optional[CallSiteID], float, float, int]] = [
            (1.0, cur, None, 0.0, 0.0, 0)]   # (path p, site, its predecessor, t50, spread^2, depth)
        expanded = 0
        while frontier and expanded < max_nodes:
            frontier.sort(key=lambda row: -row[0])
            path_p, q, q_pred, t50, var, depth = frontier.pop(0)
            expanded += 1
            for key, p_step in self.branch_probs(q)[:top_k]:
                p = path_p * p_step
                if p < p_min:
                    continue
                g50, g90 = self.gap_q(q, key, 0.5), self.gap_q(q, key, 0.9)
                a50 = t50 + g50
                if a50 > horizon_s:
                    continue
                spread2 = var + (g90 - g50) ** 2
                a90 = a50 + sqrt(spread2)
                known = out.get(key)
                if known is None:
                    out[key] = PredictedCall(key=key, p=min(1.0, p), t50=a50, t90=a90)
                else:
                    known.p = min(1.0, known.p + p)
                    if a50 < known.t50:
                        known.t50, known.t90 = a50, a90
                if depth + 1 < max_depth:
                    d50, d90 = self.duration_q(q, key, 0.5), self.duration_q(q, key, 0.9)
                    frontier.append((p, key, q, a50 + d50, spread2 + (d90 - d50) ** 2, depth + 1))
        return sorted(out.values(), key=lambda c: -c.p)

    def branch_probs(self, key: CallSiteID) -> List[Tuple[CallSiteID, float]]:
        """P(next = dst | current = key) from edge counts; uniform over the
        static successors before any count exists (the cold-start prior)."""
        succ = self.successors(key)
        if not succ:
            return []
        total = sum(e.count for e in succ)
        if total == 0:
            return [(e.dst, 1.0 / len(succ)) for e in succ]
        return sorted(((e.dst, e.count / total) for e in succ), key=lambda kv: -kv[1])


@dataclass
class PredictedCall:
    """One future call the tree search expects, with arrival-time quantiles
    (seconds past the anchor)."""
    key: CallSiteID
    p: float
    t50: float
    t90: float


# ------------------------------------------------------------------ reply repr

def type_spec(t: Any, depth: int = 0) -> Dict[str, Any]:
    """The structure of a return type, as JSON, enough to write the repr of the
    object byllm parses a reply into (`Verdict(label=<VerdictLabel.NEED_MORE: 3>,
    rationale='r', sources=['a'])`): objects with their fields in declaration
    order and default reprs, enums with their member names by value, containers,
    primitives. Computed by the analyzer from the real classes."""
    import dataclasses
    import enum as _enum
    import typing
    if depth > 6 or t is None or t is type(None):
        return {"k": "prim", "t": "none"}
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if origin in (list, set, frozenset, tuple):
        return {"k": "list", "item": type_spec(args[0], depth + 1) if args else {"k": "prim", "t": "any"}}
    if origin is dict:
        return {"k": "dict", "value": type_spec(args[1], depth + 1) if len(args) > 1 else {"k": "prim", "t": "any"}}
    if origin is not None:   # Optional[X], X | Y: the first non-None member
        inner = [a for a in args if a is not type(None)]
        return {"k": "opt", "of": type_spec(inner[0], depth + 1)} if inner else {"k": "prim", "t": "none"}
    if isinstance(t, type):
        if issubclass(t, _enum.Enum):
            return {"k": "enum", "name": t.__name__, "members": {str(m.value): m.name for m in t}}
        if dataclasses.is_dataclass(t):
            try:
                hints = typing.get_type_hints(t)
            except Exception:
                hints = {}
            fields = []
            for f in dataclasses.fields(t):
                if f.name.startswith("_"):
                    continue
                default: Optional[str] = None
                if f.default is not dataclasses.MISSING:
                    default = repr(f.default)
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    try:
                        default = repr(f.default_factory())  # type: ignore[misc]
                    except Exception:
                        default = None
                fields.append([f.name, type_spec(hints.get(f.name, f.type), depth + 1), default])
            return {"k": "obj", "name": t.__name__, "fields": fields}
        if t in (str, int, float, bool):
            return {"k": "prim", "t": t.__name__}
    return {"k": "prim", "t": "any"}


def render_repr(value: Any, spec: Optional[Dict[str, Any]]) -> Optional[str]:
    """The repr of the object byllm builds from JSON `value` under `spec`, or None
    when the value does not fit (the speculation is then simply not attempted)."""
    if spec is None:
        return None
    k = spec.get("k")
    if k == "wrapped":   # byllm's schema_object_wrapper around a non-object return type
        if isinstance(value, dict) and "schema_object_wrapper" in value:
            value = value["schema_object_wrapper"]
        return render_repr(value, spec["of"])
    if k == "prim":
        t = spec.get("t")
        if value is None:
            return "None"
        if t == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return repr(float(value))
        if t == "int" and isinstance(value, float) and value.is_integer():
            return repr(int(value))
        if t == "str" and not isinstance(value, str):
            return None
        return repr(value)
    if k == "opt":
        return "None" if value is None else render_repr(value, spec["of"])
    if k == "enum":
        name = spec["members"].get(str(value))
        if name is None and isinstance(value, str):     # the member's name instead of its value
            for v, n in spec["members"].items():
                if n == value:
                    name, value = n, (int(v) if v.lstrip("-").isdigit() else v)
                    break
        if name is None:
            return None
        return f"<{spec['name']}.{name}: {value!r}>"
    if k == "list":
        if not isinstance(value, list):
            return None
        parts = [render_repr(v, spec["item"]) for v in value]
        return None if any(p is None for p in parts) else "[" + ", ".join(parts) + "]"   # type: ignore[arg-type]
    if k == "dict":
        if not isinstance(value, dict):
            return None
        parts = []
        for kk, vv in value.items():
            r = render_repr(vv, spec["value"])
            if r is None:
                return None
            parts.append(f"{kk!r}: {r}")
        return "{" + ", ".join(parts) + "}"
    if k == "obj":
        if not isinstance(value, dict):
            return None
        parts = []
        for name, fspec, default in spec["fields"]:
            if name in value:
                r = render_repr(value[name], fspec)
                if r is None:
                    return None
            elif default is not None:
                r = default
            else:
                return None
            parts.append(f"{name}={r}")
        return f"{spec['name']}(" + ", ".join(parts) + ")"
    return None


def field_spec(spec: Optional[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """The spec of one field of an object spec."""
    if not spec or spec.get("k") != "obj":
        return None
    for fname, fspec, _ in spec["fields"]:
        if fname == name:
            return fspec
    return None


# ------------------------------------------------------------------ wire format

def site_to_dict(cs: CallSiteID) -> Dict[str, Any]:
    return {"signature": cs.signature, "lineno": cs.lineno, "file": cs.file}


def site_from_dict(d: Dict[str, Any]) -> CallSiteID:
    return CallSiteID(str(d["signature"]), int(d["lineno"]), str(d.get("file", "")))


def template_to_dict(t: PromptTemplate) -> Dict[str, Any]:
    """The static part of a template: what the analyzer emits and the server
    registers. Runtime marks (stats, order, header_last) are not on the wire."""
    return {
        "key": site_to_dict(t.key),
        "ability": t.ability,
        "resp_spec": t.resp_spec,
        "model": t.model,
        "call_params": t.call_params,
        "system_prompt": t.system_prompt,
        "tool_schema": t.tool_schema,
        "response_format": t.response_format,
        "header": t.header,
        "hint": t.hint,
        "hint_as_part": t.hint_as_part,
        "no_header_last": t.no_header_last,
        "bindings": [_binding_to_dict(b) for b in t.bindings],
    }


def _binding_to_dict(b: Binding) -> Dict[str, Any]:
    return {
        "name": b.name,
        "kind": b.kind.name,
        "heterogeneity": b.heterogeneity.name,
        "source": [site_to_dict(b.source[0]), b.source[1]] if b.source else None,
        "literal": b.literal,
        "field": b.field,
        "scope": b.scope,
        "label": b.label,
        "tail": b.tail,
    }


def _binding_from_dict(b: Dict[str, Any]) -> Binding:
    src = b.get("source")
    return Binding(
        name=b["name"], heterogeneity=Heterogeneity[b.get("heterogeneity", "VOLATILE")],
        kind=BindingKind[b.get("kind", "PARAM")],
        source=(site_from_dict(src[0]), str(src[1])) if src else None,
        literal=b.get("literal"), field=b.get("field", ""), scope=b.get("scope", ""),
        label=b.get("label", ""), tail=b.get("tail", ""))


def template_from_dict(d: Dict[str, Any]) -> PromptTemplate:
    t = PromptTemplate(
        key=site_from_dict(d["key"]), ability=d.get("ability", ""), resp_spec=d.get("resp_spec"),
        model=d.get("model", ""), call_params=dict(d.get("call_params") or {}),
        system_prompt=d.get("system_prompt", ""), tool_schema=d.get("tool_schema"),
        response_format=d.get("response_format"), header=d.get("header", ""), hint=d.get("hint", ""),
        hint_as_part=bool(d.get("hint_as_part", False)), no_header_last=bool(d.get("no_header_last", False)))
    for b in d.get("bindings", []):
        t.bindings.append(_binding_from_dict(b))
    return t


def program_to_dict(p: Program) -> Dict[str, Any]:
    return {
        "entry": site_to_dict(p.entry),
        "sites": [template_to_dict(t) for t in p.sites.values()],
        "edges": [[site_to_dict(a), site_to_dict(b), sorted(e.writes),
                   {n: _binding_to_dict(ob) for n, ob in e.overrides.items()}] for (a, b), e in p.edges.items()],
        "exits": [site_to_dict(k) for k in p.exits],
    }


def program_from_dict(d: Dict[str, Any]) -> Program:
    p = Program(entry=site_from_dict(d["entry"]))
    for t in d.get("sites", []):
        p.add_site(template_from_dict(t))
    for row in d.get("edges", []):
        p.add_edge(site_from_dict(row[0]), site_from_dict(row[1]), frozenset(row[2]) if len(row) > 2 else None,
                   {n: _binding_from_dict(ob) for n, ob in row[3].items()} if len(row) > 3 else None)
    p.exits = frozenset(site_from_dict(k) for k in d.get("exits", []))
    return p


# ------------------------------------------------------------------ visit by llm (deferred)

class VisitTemplate(PromptTemplate):
    """A `visit [...] by llm` routing callsite: "Goal: <intent>" header, walker
    and current-node reprs, the offered candidate lines. Not modelled yet."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("visit-by-llm callsites are not modelled yet")
