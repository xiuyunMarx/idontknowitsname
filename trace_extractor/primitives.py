"""Data model for history-based workflow recovery.

Nothing here parses text. `parser.py` turns raw /v1/chat/completions requests into
these objects; this module only holds them and the statistics derived from them.

Naming: a *callsite* is one `by llm()` / `visit by` in the source program. We never
see the source, so a callsite is identified by its `key`: a hash of the parts of the
request that are the same every time that line of code runs (system prompt, the
signature/description header, model). See parser.callsite_key.

The order structure is a probability-annotated prefix tree (`Program.root`), not a
Markov table: a node is a context (the last k callsites of a session), so the same
function reached along different paths keeps distinct statistics.
"""
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

END = "END"  # pseudo-successor: the session finished after this callsite


class CallsiteType(Enum):
    BYLLM = 1    # def f(...) -> T by llm(...)
    VISITBY = 2  # visit [...] by llm(intent=..., select=...)
    TOOL = 3     # def f(...) by llm(tools=[...]): a BYLLM site that runs a ReAct loop


# ----------------------------------------------------------------------------- raw input

@dataclass
class TraceRecord:
    """One line of the trace: one HTTP request to the LLM endpoint and its reply."""
    session: str            # OpenAI `user` field, "program:pid"; groups requests of one run
    t_arrive: float         # request received (seconds)
    t_done: float           # response sent (seconds)
    request: Dict[str, Any]  # the JSON body as sent: model, messages, tools, response_format, ...
    response: Any = None    # assistant message content (str) or the whole assistant message (dict)
    engine_s: Optional[float] = None  # pure engine time of this request, when the server measured it

    @property
    def program(self) -> str:
        return self.session.split(":", 1)[0] if ":" in self.session else "default"


# ----------------------------------------------------------------------------- callsites

def _lcp(a: str, b: str) -> str:
    i, n = 0, min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class LLMCallsite:
    kind: CallsiteType
    label: str                       # human name: "Agent.summarize" or "visit: <intent...>"
    model: str
    system_prompt: str
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stable_prefix: str = ""          # longest common prefix of every first-turn user text seen
    prefix_n: int = 0                # first-turn observations folded into stable_prefix

    @property
    def key(self) -> str:
        """Content-derived identity; each concrete callsite kind defines its own."""
        raise NotImplementedError


@dataclass
class ByLLMCallsite(LLMCallsite):
    context_desc: str = ""                                    # header + schema rows, fixed text
    qualname: str = ""                                        # "Agent.summarize"
    params: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, type, sem)
    return_type: str = ""
    sem: str = ""
    tools: List[str] = field(default_factory=list)            # tool names incl. finish_tool

    @property
    def key(self) -> str:
        param_desc = ",".join(f"{n}:{t}:{s}" for n, t, s in self.params)
        return f"{self.qualname}({param_desc}) by {self.model}"


@dataclass
class VisitByCallsite(LLMCallsite):
    intent: str = ""
    select: str = ""                                          # "exactly one" / "exactly 3" / "between 1 and 3" / "all"
    candidates: List[str] = field(default_factory=list)       # union of handles seen across calls
    edges: List[str] = field(default_factory=list)            # union of edge descriptions seen

    @property
    def key(self) -> str:
        return f"visit: {self.intent} by {self.model}"
# ----------------------------------------------------------------------------- per-call data

@dataclass
class Call:
    """One parsed request (one HTTP round trip)."""
    key: str
    session: str
    t_arrive: float
    t_done: float
    model: str
    turn: int = 0                                  # ReAct turn index (0 = first request of the call)
    bindings: Dict[str, str] = field(default_factory=dict)   # BYLLM: param -> repr text
    self_view: Optional[str] = None                # BYLLM: the `self = ...` repr, if present
    walker: Optional[str] = None                   # VISITBY zones
    here: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    response: Any = None
    raw: Optional[Dict[str, Any]] = None           # the request, kept for prefix-extension checks
    engine_s: Optional[float] = None               # measured engine seconds, if the trace carries them


@dataclass
class CallInstance:
    """One execution of a callsite. A ReAct call is several requests folded into one instance."""
    key: str
    session: str
    model: str
    t_arrive: float
    t_done: float
    turns: List[Tuple[float, float]] = field(default_factory=list)  # (arrive, done) per request
    bindings: Dict[str, str] = field(default_factory=dict)
    self_view: Optional[str] = None
    result: Any = None                              # final response (last turn)
    engine_s: float = 0.0                           # measured engine seconds summed over turns, if known

    @property
    def exec_s(self) -> float:
        """Engine time: the measured sum when the trace carries it, else request wall time
        summed over turns (includes queueing) excluding the tool gaps between turns."""
        return self.engine_s if self.engine_s > 0 else sum(d - a for a, d in self.turns)

    @property
    def tool_gaps(self) -> List[float]:
        return [self.turns[i + 1][0] - self.turns[i][1] for i in range(len(self.turns) - 1)]


@dataclass
class Session:
    id: str
    program: str
    calls: List[CallInstance] = field(default_factory=list)

    @property
    def sequence(self) -> List[str]:
        return [c.key for c in self.calls]


# ----------------------------------------------------------------------------- workflow

START = "^"  # context sentinel: a session's first callsite is a child of [START]


@dataclass
class TrieNode:
    """One context: a suffix of some ([START] + session sequence), rooted at the tree.
    `children[k]` are the observed continuations; `count` how often this context
    occurred; `end` how often the session ended right here (count == sum(children) + end)."""
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    count: int = 0
    end: int = 0

    @property
    def resolved(self) -> int:
        """Occurrences whose continuation is known (a live session's freshly-extended
        context has count > resolved until its next callsite arrives or it closes).
        Probabilities divide by this, so concurrent sessions do not dilute each other."""
        return sum(c.count for c in self.children.values()) + self.end

    def to_dict(self) -> Dict[str, Any]:
        return {"c": self.count, "e": self.end, "k": {k: v.to_dict() for k, v in self.children.items()}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrieNode":
        node = cls(count=d["c"], end=d["e"])
        node.children = {k: cls.from_dict(v) for k, v in d["k"].items()}
        return node


def _find_in_result(result: Any, v: str) -> Optional[list]:
    """Where repr-text `v` sits inside a call's result: [] when it is the whole text,
    the key/index path of a JSON field of it, None when absent."""
    if not isinstance(result, str):
        return None
    if repr(result) == v:
        return []
    try:
        parsed = json.loads(result)
    except Exception:
        return None
    stack: List[Tuple[Any, list]] = [(parsed, [])]
    while stack:
        x, path = stack.pop()
        if isinstance(x, dict):
            stack.extend((val, path + [k]) for k, val in x.items())
        elif isinstance(x, list):
            stack.extend((val, path + [i]) for i, val in enumerate(x))
        elif repr(x) == v:
            return path
    return None


def _value_at(result: Any, path: list) -> Optional[str]:
    """The repr text at `path` of a result (inverse of _find_in_result)."""
    if not isinstance(result, str):
        return None
    if not path:
        return repr(result)
    try:
        x: Any = json.loads(result)
        for p in path:
            x = x[p]
    except Exception:
        return None
    return repr(x)


def _provenance(earlier: List[CallInstance], name: str, v: str) -> str:
    """Which earlier value of the session explains binding `name = v`:
    copy (same name), resp (the source call's result or a JSON field of it),
    take (another name), else the literal itself — a value that only ever repeats
    across sessions accumulates support as a constant, a fresh one never does."""
    if any(e.bindings.get(name) == v for e in earlier):
        return "copy"
    for e in reversed(earlier):
        path = _find_in_result(e.result, v)
        if path is not None:
            return f"resp:{e.key}\x00{json.dumps(path)}"
    for e in reversed(earlier):
        for param, w in e.bindings.items():
            if w == v:
                return f"take:{e.key}\x00{param}"
    return f"const:{v}"


class Program:
    """Everything learned about one program from its sessions.

    Order structure: a generalized prefix tree over callsite sequences. Every suffix of
    every ([START] + sequence) is inserted from the root"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.callsites: Dict[str, LLMCallsite] = {}
        self.sessions: Dict[str, Session] = {}
        self.root = TrieNode()
        # time statistics
        self.gap: Dict[Tuple[str, str], List[float]] = defaultdict(list)   # (a, b) -> b.arrive - a.done
        self.exec: Dict[str, List[float]] = defaultdict(list)          # key -> engine seconds per instance
        self.turns: Dict[str, List[int]] = defaultdict(list)           # key -> ReAct turns per instance
        self.tool_gap: Dict[str, List[float]] = defaultdict(list)      # key -> seconds between turns
        # value flow: (key, param) -> provenance rule -> times observed (see _provenance)
        self.flow: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------ building
    def add_callsite(self, site: LLMCallsite) -> LLMCallsite:
        """Register a callsite; seen again, it folds the new first-turn user text into
        stable_prefix (the byte-stable head speculative prefill may warm) and a VISITBY
        site merges its candidate/edge sets."""
        known = self.callsites.get(site.key)
        if known is None:
            self.callsites[site.key] = site
            return site
        if site.prefix_n:
            if known.prefix_n:
                known.stable_prefix = _lcp(known.stable_prefix, site.stable_prefix)
            else:
                known.stable_prefix = site.stable_prefix
            known.prefix_n += 1
        if isinstance(known, VisitByCallsite) and isinstance(site, VisitByCallsite):
            for h in site.candidates:
                if h not in known.candidates:
                    known.candidates.append(h)
            for e in site.edges:
                if e not in known.edges:
                    known.edges.append(e)
        return known

    def begin_session(self) -> List[TrieNode]:
        """Active context nodes of a fresh session (one per open suffix); feed to `extend`."""
        return self.extend([self.root], START)

    def extend(self, active: List[TrieNode], key: str) -> List[TrieNode]:
        """One callsite arrived: extend every open suffix (and open a new one at the root)."""
        out = [self.root]
        for node in active:
            child = node.children.get(key)
            if child is None:
                child = node.children[key] = TrieNode()
            child.count += 1
            out.append(child)
        return out

    def finish_session(self, active: List[TrieNode], session: Session) -> None:
        """The session is over: every open context ends here."""
        for node in active:
            if node is not self.root:
                node.end += 1
        self.sessions[session.id] = session
        self.observe_flow(session)

    def observe_flow(self, session: Session) -> None:
        """Learn, per (callsite, param), where the parameter's value came from within
        the session — so a future session's values can be prefilled before the call."""
        for j, inst in enumerate(session.calls):
            site = self.callsites.get(inst.key)
            if not isinstance(site, ByLLMCallsite) or j == 0:
                continue
            for name, _, _ in site.params:
                v = inst.bindings.get(name)
                if v is not None:
                    self.flow[(inst.key, name)][_provenance(session.calls[:j], name, v)] += 1

    def observe_times(self, inst: CallInstance, nxt: Optional[CallInstance]) -> None:
        """Time tables for the completed call `inst`; `nxt` its successor (None: session end)."""
        if nxt is not None:
            self.gap[(inst.key, nxt.key)].append(nxt.t_arrive - inst.t_done)
        self.exec[inst.key].append(inst.exec_s)
        self.turns[inst.key].append(len(inst.turns))
        self.tool_gap[inst.key].extend(inst.tool_gaps)

    def observe(self, session: Session) -> None:
        """Offline: fold a whole finished session in (same totals as the online path)."""
        active = self.begin_session()
        seq = session.calls
        for i, inst in enumerate(seq):
            self.observe_times(inst, seq[i + 1] if i + 1 < len(seq) else None)
            active = self.extend(active, inst.key)
        self.finish_session(active, session)

    # ------------------------------------------------------------------ queries (planner interface)
    def _node(self, ctx: List[str]) -> Optional[TrieNode]:
        node = self.root
        for k in ctx:
            node = node.children.get(k)
            if node is None:
                return None
        return node

    def context_node(self, walked: List[str]) -> Tuple[TrieNode, int]:
        """The deepest tree node matching a suffix of [START] + walked whose continuation
        has been observed at least once (resolved > 0); (root, 0) if none."""
        ctx = [START] + list(walked)
        for i in range(len(ctx)):
            node = self._node(ctx[i:])
            if node is not None and node.resolved > 0:
                return node, len(ctx) - i
        return self.root, 0

    def branch_candidates(self, walked) -> List[Tuple[str, float]]:
        """Continuations of the live context, most likely first. `walked` is the session's
        callsite sequence so far; a bare key gives the context-free (depth-1) marginal."""
        node, depth = self.context_node([walked] if isinstance(walked, str) else list(walked))
        if depth == 0 or node.resolved == 0:
            return []
        out = [(k, c.count / node.resolved) for k, c in node.children.items()]
        out.sort(key=lambda kv: -kv[1])
        return out

    def end_prob(self, walked) -> float:
        node, depth = self.context_node([walked] if isinstance(walked, str) else list(walked))
        return node.end / node.resolved if depth and node.resolved else 0.0

    def model_of(self, key_or_site: Any, fallback: str = "") -> str:
        key = key_or_site.key if isinstance(key_or_site, LLMCallsite) else key_or_site
        site = self.callsites.get(key)
        return site.model if site else fallback

    def exec_s(self, key: str, default: float = 0.0) -> float:
        xs = self.exec.get(key)
        return median(xs) if xs else default

    def gap_s(self, key: str, nxt: str, default: float = 0.0) -> float:
        xs = self.gap.get((key, nxt))
        return median(xs) if xs else default

    def turns_of(self, key: str) -> float:
        xs = self.turns.get(key)
        return median(xs) if xs else 1.0

    def duration_s(self, key: str, default: float = 0.0) -> float:
        """Wall seconds one execution of `key` occupies end to end. For a multi-turn ReAct
        call this is the whole conversation: engine time plus the client-side tool gaps
        between its turns (exec + tool_gap * (turns - 1))."""
        exec_s = self.exec_s(key)
        if not exec_s:
            return default
        return exec_s + self.tool_gap_s(key) * max(0.0, self.turns_of(key) - 1.0)

    def value_prefix(self, key: str, calls: List[CallInstance]) -> str:
        """The head of `key`'s next user message reconstructable from values the live
        session already carries: context_desc, then `name = value` rows in signature
        order under the learned flow rules, stopping at the first parameter whose value
        cannot be resolved. Empty when no binding resolves."""
        site = self.callsites.get(key)
        if not isinstance(site, ByLLMCallsite) or not calls:
            return ""
        out = [site.context_desc]
        for name, _, _ in site.params:
            v = self._flow_value(key, name, calls)
            if v is None:
                break
            out.append(f"{name} = {v}")
        return "\n".join(out) if len(out) > 1 else ""

    def _flow_value(self, key: str, name: str, calls: List[CallInstance]) -> Optional[str]:
        """Apply the dominant (majority) provenance rule of (key, name) to the session."""
        rules = self.flow.get((key, name))
        if not rules:
            return None
        rule, n = max(rules.items(), key=lambda kv: kv[1])
        if 2 * n < sum(rules.values()):
            return None
        if rule == "copy":
            return next((e.bindings[name] for e in reversed(calls) if name in e.bindings), None)
        kind, _, rest = rule.partition(":")
        if kind == "take":
            src, param = rest.split("\x00")
            return next((e.bindings[param] for e in reversed(calls) if e.key == src and param in e.bindings), None)
        if kind == "resp":
            src, path = rest.split("\x00", 1)
            return next((_value_at(e.result, json.loads(path)) for e in reversed(calls) if e.key == src), None)
        if kind == "const":
            return rest
        return None

    def tool_gap_s(self, key: str, default: float = 0.0) -> float:
        """Typical client-side gap between two ReAct turns of `key` (the tool runs then)."""
        xs = self.tool_gap.get(key)
        return median(xs) if xs else default

    def predict(self, walked: List[str], max_steps: int = 8) -> List[Tuple[str, float, float]]:
        """Greedy walk over the tree from the live context: [(next key, probability of
        reaching it, seconds until it arrives)]. At every step the deepest matching
        context votes; the most probable child is taken until END outweighs every
        child (the predicted leaf), the context leaves observed data, or max_steps.
        The time axis charges each call its full duration — a multi-turn ReAct call
        counts engine time plus its tool gaps (duration_s), then the edge gap."""
        ctx = list(walked)
        out: List[Tuple[str, float, float]] = []
        p, t = 1.0, 0.0
        while len(out) < max_steps:
            node, depth = self.context_node(ctx)
            if depth == 0 or not node.children:
                break
            key, child = max(node.children.items(), key=lambda kv: kv[1].count)
            if node.end > child.count:
                break  # leaf: ending here is more likely than any continuation
            p *= child.count / node.resolved
            if ctx:
                t += self.gap_s(ctx[-1], key)
            out.append((key, p, t))
            t += self.duration_s(key)
            ctx.append(key)
        return out

    def predict_models(self, walked: List[str], max_steps: int = 8) -> List[Tuple[str, str]]:
        """[(callsite key, model)] of the greedy prediction — what the planner consumes."""
        return [(key, self.model_of(key)) for key, _, _ in self.predict(walked, max_steps)]

    # ------------------------------------------------------------------ inspection
    def label(self, key: str) -> str:
        return self.callsites[key].label if key in self.callsites else key

    def describe(self) -> str:
        lines = [f"program {self.name}: {len(self.callsites)} callsites, {len(self.sessions)} sessions"]
        start = self.root.children.get(START)
        if start is not None and start.resolved:
            for key, child in sorted(start.children.items(), key=lambda kv: -kv[1].count):
                lines.append(f"  entry  {self.label(key):45} p={child.count / start.resolved:.2f} (n={child.count})")
        for key in self.callsites:
            site = self.callsites[key]
            xs = self.exec.get(key, [])
            turns = self.turns.get(key, [])
            tg = self.tool_gap.get(key)
            lines.append(f"  {site.label:45} {site.kind.name:7} {site.model:28} "
                         f"n={len(xs)} exec={median(xs) if xs else 0:.2f}s"
                         + (f" turns={median(turns):.0f}" if turns and max(turns) > 1 else "")
                         + (f" tool_gap={median(tg):.2f}s" if tg else ""))
            node = self.root.children.get(key)  # depth-1 context: the context-free marginal
            if node is None or not node.resolved:
                continue
            conts = sorted(node.children.items(), key=lambda kv: -kv[1].count)
            for nxt, child in conts:
                g = self.gap.get((key, nxt))
                lines.append(f"      -> {self.label(nxt):42} p={child.count / node.resolved:.2f} (n={child.count})"
                             + (f" gap={median(g):.2f}s" if g else ""))
            if node.end:
                lines.append(f"      -> {END:42} p={node.end / node.resolved:.2f} (n={node.end})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        def site_dict(s: LLMCallsite) -> Dict[str, Any]:
            d = dict(vars(s))
            d["kind"] = s.kind.name
            return d
        return {
            "name": self.name,
            "callsites": {k: site_dict(s) for k, s in self.callsites.items()},
            "tree": self.root.to_dict(),
            "gap": {f"{a}\x00{b}": xs for (a, b), xs in self.gap.items()},
            "exec": dict(self.exec),
            "turns": dict(self.turns),
            "tool_gap": dict(self.tool_gap),
            "flow": {f"{k}\x00{p}": dict(rules) for (k, p), rules in self.flow.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Program":
        if "tree" not in d:
            raise ValueError("old (Markov-table) history format; rebuild it from a trace with trace_extractor.parser")
        p = cls(d["name"])
        for k, s in d["callsites"].items():
            s = dict(s)
            kind = CallsiteType[s.pop("kind")]
            site_cls = VisitByCallsite if kind is CallsiteType.VISITBY else ByLLMCallsite
            if site_cls is ByLLMCallsite:
                s["params"] = [tuple(p_) for p_ in s.get("params", [])]
            p.callsites[k] = site_cls(kind=kind, **s)
        p.root = TrieNode.from_dict(d["tree"])
        for k, xs in d["gap"].items():
            a, b = k.split("\x00")
            p.gap[(a, b)] = list(xs)
        for name in ("exec", "turns", "tool_gap"):
            for k, xs in d[name].items():
                getattr(p, name)[k] = list(xs)
        for k, rules in d.get("flow", {}).items():
            key, param = k.rsplit("\x00", 1)
            p.flow[(key, param)] = defaultdict(int, rules)
        return p
