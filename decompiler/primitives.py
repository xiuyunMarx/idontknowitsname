import json
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from statistics import median
from collections import defaultdict, Counter
from math import log, sqrt
TOOL_BLOCK_HEADER = "\n# Calling tools\n"  # Header for tools included in the system prompt.


def _quantile(xs: List[float], q: float) -> float:
    """Empirical quantile of raw samples; 0.0 when empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


@dataclass
class ExecStats:
    """Store timing data for completed calls."""
    engine_s: List[float] = field(default_factory=list)    # Model time per call.
    turns: List[int] = field(default_factory=list)         # HTTP requests per call.
    tool_gap_s: List[float] = field(default_factory=list)  # Time between requests.

    def observe(self, engine_s: float, turns: int, tool_gaps: List[float]) -> None:
        self.engine_s.append(engine_s)
        self.turns.append(turns)
        self.tool_gap_s.extend(tool_gaps)

    @property
    def exec_s(self) -> float:
        """Return the typical model time for one call."""
        return median(self.engine_s) if self.engine_s else 0.0

    @property
    def duration_s(self) -> float:
        """Return the typical total duration of one call."""
        turns = median(self.turns) if self.turns else 1.0
        gap = median(self.tool_gap_s) if self.tool_gap_s else 0.0
        return self.exec_s + gap * max(0.0, turns - 1.0)

    def duration_q(self, q: float) -> float:
        """Quantile analogue of duration_s, for deadline estimates."""
        turns = median(self.turns) if self.turns else 1.0
        return _quantile(self.engine_s, q) + _quantile(self.tool_gap_s, q) * max(0.0, turns - 1.0)


@dataclass
class CallMetadata:
    model: str
    temperature: Optional[float] = None      # Optional request setting.
    max_tokens: Optional[int] = None
    system_prompt: str = ""                  # Required prefix from the first token.
    response_format: Optional[Dict[str, Any]] = None   # Shared request format.
    hint: str = ""                           # Schema-requirements tail appended to the user message.
    exec_stats: ExecStats = field(default_factory=ExecStats)  # Running timing data.

    @property
    def key(self) -> str:
        """Return a stable identifier for this call."""
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError

    @property
    def fixed_head(self) -> str:
        """Bytes of the user message that are the same in every call of this site,
        known from the layout (not from comparing observations)."""
        raise NotImplementedError


@dataclass
class ByLLMCallsite(CallMetadata):
    """Describe a function implemented by an LLM."""
    context_desc: str = ""                                    # Fixed header and schema text.
    signature: str = ""                                       # Full name, such as Agent.summarize.
    params: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # Name to type and description.
    param_type_str_view: Dict[str, str] = field(default_factory=dict)  # Schema text for each parameter.
    return_type: str = ""                                     # Declared return type.
    sem: str = ""                                             # Function description.
    owner_sem: str = ""                                       # Description of the owning type.
    tool_schema: Optional[List[Dict[str, Any]]] = None        # Tools sent with the request.
    layout: Optional[List[str]] = None                        # Frozen binding order (Program._freeze_layout).

    @property
    def is_react(self) -> bool:
        return bool(self.tool_schema) or TOOL_BLOCK_HEADER in self.system_prompt

    @property
    def key(self) -> str:
        param_desc = ",".join(f"{n}:{t}:{s}" for n, (t, s) in self.params.items())
        return f"{self.signature}({param_desc}) by {self.model}"

    @property
    def label(self) -> str:
        return f"{self.signature}({self.params})->{self.return_type} by {self.model}"

    @property
    def fixed_head(self) -> str:
        return self.context_desc   # header line (signature + sem) and schema rows


@dataclass
class VisitByCallsite(CallMetadata):
    """Describe an LLM call that selects from a list."""
    intent: str = ""
    select: str = ""                                          # "exactly one" / "exactly 3" / "between 1 and 3" / "all"
    resp_head: str = ""                                       # Shared prefix of observed replies (probe scaffold).
    resp_n: int = 0                                           # Replies folded into resp_head.

    @property
    def key(self) -> str:
        return f"visit: {self.intent} by {self.model}"

    @property
    def label(self) -> str:
        intent = self.intent or "(no intent)"
        return "visit: " + (intent[:50] + "..." if len(intent) > 50 else intent)

    @property
    def fixed_head(self) -> str:
        return f"Goal: {self.intent}" if self.intent else ""

# ----------------------------------------------------------------------- program model

Callsite = Union[ByLLMCallsite, VisitByCallsite]

START = "^"  # Marker placed before the first call in a session.
STABILITY = {"const": 0, "copy": 1, "extend": 2}  # flow-rule kind -> how long the bytes stay a valid prefix


def _lcp(a: str, b: str) -> str:
    i, n = 0, min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class Edge:
    """Store an observed move between two graph states."""
    endpoint: str                            # Destination state ID.
    edge_str_view: Optional[str] = None      # Display text for a visit option.
    seen_freq: int = 0                       # Times the option was offered.
    taken_freq: int = 0                      # Times the move occurred.
    gap_time: List[float] = field(default_factory=list)  # Delay before the next call.

    @property
    def gap_s(self) -> float:
        return median(self.gap_time) if self.gap_time else 0.0


@dataclass
class TrieNode:
    """Store call history used to predict the next call."""
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    count: int = 0
    end: int = 0

    @property
    def resolved(self) -> int:
        """Return the number of observed outcomes."""
        return sum(c.count for c in self.children.values()) + self.end


@dataclass
class CallObservation:
    """Store data collected from one completed call."""
    key: str
    t_arrive: float
    t_done: float
    engine_s: float = 0.0
    n_turns: int = 1 # turns of this logical call.
    tool_gaps: List[float] = field(default_factory=list)
    candidates: List[Tuple[str, str]] = field(default_factory=list)  # (handle, node repr) offered by a visit.
    bindings: Dict[str, str] = field(default_factory=dict)  # Parameter reprs of a byllm call.
    self_view: Optional[str] = None      # Repr of the owning object.
    walker: Optional[str] = None         # Repr of a visit call's walker.
    here: Optional[str] = None           # Repr of a visit call's current node.
    cand_block: str = ""                 # Raw candidate lines of a visit call.
    response: str = ""                   # Assistant reply text.
    user_text: str = ""                  # Raw first user message, for reconstruction checks.


# ------------------------------------------------------------------------- value flow

def _split_top(body: str) -> Optional[List[str]]:
    """Split on commas at nesting depth zero, honouring quotes and escapes."""
    parts, depth, quote, esc, start = [], 0, None, False, 0
    for j, ch in enumerate(body):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif ch == "," and depth == 0:
            parts.append(body[start:j])
            start = j + 1
    if quote or depth:
        return None
    parts.append(body[start:])
    return parts


def _valid_key(k: str) -> bool:
    """A field key is `name` or `name (its sem text)` — byllm renders both."""
    if k.isidentifier():
        return True
    head, sep, _ = k.partition(" (")
    return bool(sep) and head.isidentifier() and k.endswith(")")


def field_name(key: str) -> str:
    return key.partition(" (")[0]


def parse_fields(text: str) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """Split a repr like Type(a=1, b='x') into its type text and field rows."""
    text = text.strip()
    if len(text) < 3 or not text.endswith(")"):
        return None
    stack: List[int] = []
    quote: Optional[str] = None
    esc = False
    opener = None
    for j, ch in enumerate(text):
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            stack.append(j)
        elif ch == ")":
            if not stack:
                return None
            k = stack.pop()
            if j == len(text) - 1:
                opener = k
    if quote or stack or not opener:
        return None
    parts = _split_top(text[opener + 1:-1])
    if parts is None:
        return None
    fields: List[Tuple[str, str]] = []
    for part in parts:
        if not part.strip():
            continue
        k, eq, v = part.partition("=")
        if not eq or not _valid_key(k.strip()):
            return None
        fields.append((k.strip(), v.strip()))
    return text[:opener], fields


def build_fields(name: str, fields: List[Tuple[str, str]]) -> str:
    return f"{name}({', '.join(f'{k}={v}' for k, v in fields)})"


def _find_in_result(result: Any, v: str) -> Optional[list]:
    """Where repr-text `v` sits inside a call's result: [] when it is the raw text,
    ["j", *path] when it is the JSON value at `path` (the whole parse included),
    None when absent."""
    if not isinstance(result, str):
        return None
    if repr(result) == v:
        return []
    try:
        parsed = json.loads(result)
    except Exception:
        return None
    stack: List[Tuple[Any, list]] = [(parsed, ["j"])]
    while stack:
        x, path = stack.pop()
        if repr(x) == v:
            return path
        if isinstance(x, dict):
            stack.extend((val, path + [k]) for k, val in x.items())
        elif isinstance(x, list):
            stack.extend((val, path + [i]) for i, val in enumerate(x))
    return None


def _value_at(result: Any, path: list) -> Optional[str]:
    """The repr text at `path` of a result (inverse of _find_in_result)."""
    if not isinstance(result, str):
        return None
    if not path:
        return repr(result)
    if path[0] != "j":
        return None
    try:
        x: Any = json.loads(result)
        for p in path[1:]:
            x = x[p]
    except Exception:
        return None
    return repr(x)


def _find_word(text: str, w: str) -> int:
    """First occurrence of `w` in `text` not embedded in a larger identifier."""
    i = 0
    while True:
        i = text.find(w, i)
        if i < 0:
            return -1
        before = text[i - 1] if i else ""
        after = text[i + len(w):i + len(w) + 1]
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return i
        i += 1


def chosen_candidates(ob: CallObservation) -> List[Tuple[str, str]]:
    """The (handle, node repr) candidates a visit reply selected, in reply order."""
    picks = []
    for h, node in ob.candidates:
        i = ob.response.find(f'"{h}"')
        if i < 0:
            i = _find_word(ob.response, h)
        if i >= 0:
            picks.append((i, h, node))
    picks.sort()
    return [(h, node) for _, h, node in picks]


def _named_values(ob: CallObservation) -> Dict[str, str]:
    """Every value of one call a later call could draw from — and, symmetrically,
    every dynamic part of this call that needs explaining. Structured reprs
    contribute both the whole text and each field."""
    out = dict(ob.bindings)
    for tag, text in (("self", ob.self_view), ("walker", ob.walker), ("here", ob.here)):
        if text is None:
            continue
        out[tag] = text
        parsed = parse_fields(text)
        if parsed is not None:
            for f, v in parsed[1]:
                out[f"{tag}.{field_name(f)}"] = v
    if ob.cand_block:
        out["cands"] = ob.cand_block
    return out


def _provenance(earlier: List[CallObservation], name: str, v: str) -> str:
    """Which earlier value of the session explains `name = v`: the same name again
    (copy), a call's response or a JSON field of it (resp), any other named value
    (take), the candidate a visit reply chose (cand), else the literal itself — a
    value that only ever repeats across sessions accumulates support as a constant,
    a fresh one never does."""
    if any(_named_values(e).get(name) == v for e in earlier):
        return "copy"
    for e in reversed(earlier):
        # the same name's latest value minus its closing delimiter is a prefix: an
        # accumulating history (list/str/dict repr grown by appending)
        w = _named_values(e).get(name)
        if w is not None and len(w) > 2 and len(v) > len(w) and v.startswith(w[:-1]):
            return "extend"
    for e in reversed(earlier):
        path = _find_in_result(e.response, v)
        if path is not None:
            return f"resp:{e.key}\x00{json.dumps(path)}"
    for e in reversed(earlier):
        for n2, w in _named_values(e).items():
            if w == v and n2 != name:
                return f"take:{e.key}\x00{n2}"
    for e in reversed(earlier):
        for _, node in chosen_candidates(e):
            if node == v:
                return f"cand:{e.key}\x00"
            parsed = parse_fields(node)
            if parsed is not None:
                for f, w in parsed[1]:
                    if w == v:
                        return f"cand:{e.key}\x00{field_name(f)}"
    return f"const:{v}"


def node_type(node_repr: str) -> str:
    return node_repr.split("(", 1)[0].strip()


@dataclass
class Node:
    """Represent one call in the learned execution graph."""
    id: str
    symbol: str                                          # Key of the call definition.
    stats: ExecStats = field(default_factory=ExecStats)  # Timing data for this state.


# -------------------------------------------------------------------- graph rebuilding

class _State:
    """Store temporary state while rebuilding the graph."""
    __slots__ = ("symbol", "n", "end", "kids")

    def __init__(self, symbol: Optional[str]) -> None:
        self.symbol = symbol
        self.n = 0          # Sessions that reached this state.
        self.end = 0        # Sessions that ended here.
        self.kids: Dict[str, list] = {}


def _close(c1: int, n1: int, c2: int, n2: int, eps: float) -> bool:
    """Return whether two observed rates are statistically similar."""
    if n1 == 0 or n2 == 0:
        return True
    return abs(c1 / n1 - c2 / n2) < eps * (1 / sqrt(n1) + 1 / sqrt(n2))


def _compat(a: _State, b: _State, eps: float, seen: Optional[set] = None) -> bool:
    """Return whether two graph states can be merged."""
    seen = set() if seen is None else seen
    if (id(a), id(b)) in seen:
        return True
    seen.add((id(a), id(b)))
    if not _close(a.end, a.n, b.end, b.n, eps):
        return False
    for s in set(a.kids) | set(b.kids):
        la, lb = a.kids.get(s), b.kids.get(s)
        if not _close(la[1] if la else 0, a.n, lb[1] if lb else 0, b.n, eps):
            return False
        if la and lb and la[0] is not lb[0] and not _compat(la[0], lb[0], eps, seen):
            return False
    return True


def _fold(a: _State, b: _State) -> None:
    """Merge state `b` into state `a`."""
    a.n += b.n
    a.end += b.end
    for s, (bs, bc) in b.kids.items():
        la = a.kids.get(s)
        if la is None:
            a.kids[s] = [bs, bc]
        else:
            la[1] += bc
            if la[0] is not bs:
                _fold(la[0], bs)


class Program:
    """Store the learned execution flow for one agent."""

    def __init__(self, entry: str = "", k: int = 4) -> None:
        self.entry = entry                   # Key of the first call.
        self.k = k                           # Number of previous calls to remember.
        self.sites: Dict[str, Callsite] = {}                 # Call definitions by key.
        self.nodes: Dict[str, Node] = {}                     # Learned graph states.
        self.graph: Dict[str, List[Edge]] = defaultdict(list)  # Outgoing moves by state ID.
        self.exit_freq: Dict[str, int] = defaultdict(int)    # Sessions ending at each state.
        self.gap: Dict[Tuple[str, str], List[float]] = defaultdict(list)  # Delays between call pairs.
        self.flow: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # (symbol, name) -> provenance rule -> count.
        self.type_succ: Dict[str, Counter] = defaultdict(Counter)  # Node type -> symbol called on it.
        self.proto: Dict[str, Dict[str, Any]] = {}           # Symbol -> user-message layout template.
        self.seqs: Counter = Counter()                       # Completed call sequences.
        self.ctx = TrieNode()
        self.n_sessions = 0

    # Data collection
    def match(self, site: Callsite) -> bool:
        """Return whether `site` is the first call."""
        return bool(self.entry) and site.key == self.entry

    def add_callsite(self, site: Callsite) -> Callsite:
        """Add a call definition or update the stored one."""
        known = self.sites.get(site.key)
        if known is None:
            self.sites[site.key] = site
            return site
        return known

    def edge(self, a: str, b: str) -> Edge:
        """Return the graph edge from `a` to `b`, creating it if needed."""
        for e in self.graph[a]:
            if e.endpoint == b:
                return e
        e = Edge(endpoint=b)
        self.graph[a].append(e)
        return e

    def _step(self, q: str, key: str) -> str:
        """Find or create the next state for a call."""
        for e in self.graph.get(q, ()):
            n = self.nodes.get(e.endpoint)
            if n is not None and n.symbol == key:
                return e.endpoint
        nid = f"{key}#0"
        if nid not in self.nodes:
            self.nodes[nid] = Node(id=nid, symbol=key)
        return nid

    def update_graph(self, walked: List[str], obs: List[CallObservation]) -> List[str]:
        """Record calls and timing data from one completed session, and update the
        value-flow rules. Returns the keys of byllm sites whose binding layout got
        frozen by this session (see _freeze_layout)."""
        if not obs:
            return []
        self.seqs[tuple(walked)] += 1
        self.n_sessions += 1
        q, prev = START, None
        for i, (key, ob) in enumerate(zip(walked, obs)):
            nid = self._step(q, key)
            e = self.edge(q, nid)
            e.taken_freq += 1
            if prev is not None:
                g = ob.t_arrive - prev.t_done
                e.gap_time.append(g)
                self.gap[(prev.key, key)].append(g)
            self.nodes[nid].stats.observe(ob.engine_s, ob.n_turns, ob.tool_gaps)
            site = self.sites.get(key)
            if site is not None:
                site.exec_stats.observe(ob.engine_s, ob.n_turns, ob.tool_gaps)
            self._observe_values(key, ob, obs[:i], walked[i + 1:])
            q, prev = nid, ob
        self.exit_freq[q] += 1
        self._trie_fold(walked)
        self._observe_offers(walked, obs)
        frozen = []
        for k in dict.fromkeys(walked):
            site = self.sites.get(k)
            if isinstance(site, ByLLMCallsite) and self._freeze_layout(site):
                frozen.append(k)
        return frozen

    def _trie_fold(self, keys: List[str]) -> None:
        """Add recent call sequences to the prediction history."""
        active: List[Tuple[TrieNode, int]] = [(self.ctx, 0)]
        for key in [START] + list(keys):
            nxt: List[Tuple[TrieNode, int]] = [(self.ctx, 0)]
            for node, d in active:
                if d >= self.k:
                    continue
                child = node.children.get(key)
                if child is None:
                    child = node.children[key] = TrieNode()
                child.count += 1
                nxt.append((child, d + 1))
            active = nxt
        for node, _ in active:
            if node is not self.ctx:
                node.end += 1

    # Value flow
    def _observe_values(self, key: str, ob: CallObservation, earlier: List[CallObservation],
                        rest: List[str]) -> None:
        """Fold one call's values into the flow rules, routing linkage and template."""
        for name, v in _named_values(ob).items():
            self.flow[(key, name)][_provenance(earlier, name, v)] += 1
        site = self.sites.get(key)
        if isinstance(site, VisitByCallsite) and ob.response:
            site.resp_head = _lcp(site.resp_head, ob.response) if site.resp_n else ob.response
            site.resp_n += 1
            # The reply names the nodes the walker visits next, in order: link each
            # chosen node's type to the symbol that then ran on it.
            for j, (_, node) in enumerate(chosen_candidates(ob)):
                if j < len(rest):
                    self.type_succ[node_type(node)][rest[j]] += 1
        self._observe_proto(key, ob, earlier)

    def _observe_proto(self, key: str, ob: CallObservation, earlier: List[CallObservation]) -> None:
        """Remember how this call's user message is laid out, and count how often the
        learned rules rebuild it byte-for-byte (speculation trusts only checked layouts)."""
        p = self.proto.setdefault(key, {"ok": 0, "n": 0})
        p["order"] = list(ob.bindings)
        p["self"] = ob.self_view is not None
        p["self_gap"] = "\n\nself = " in ob.user_text
        p["self_fields"] = parse_fields(ob.self_view) if ob.self_view is not None else None
        if ob.self_view is not None and ob.user_text:
            # The type-member rows after the self line are part of the message and
            # byte-stable for the site.
            site = self.sites.get(key)
            hint = site.hint if site is not None else ""
            stripped = ob.user_text[:-len(hint)] if hint and ob.user_text.endswith(hint) else ob.user_text
            i = stripped.find("\nself = ")
            j = stripped.find("\n", i + 1) if i >= 0 else -1
            p["self_rows"] = stripped[j:] if j >= 0 else ""
        for tag, text in (("walker", ob.walker), ("here", ob.here)):
            if text is not None:
                p[tag] = parse_fields(text)
        if ob.user_text:
            p["n"] += 1
            built, complete = self.resolve_user(key, list(earlier))
            if complete and built == ob.user_text:
                p["ok"] += 1

    def _observe_offers(self, walked: List[str], obs: List[CallObservation]) -> None:
        """Count each offered candidate against its learned successor edge, so
        seen_freq/taken_freq gives P(branch taken | branch offered)."""
        q = START
        for key, ob in zip(walked, obs):
            q = self._step_ro(q, key)
            for _, node in ob.candidates:
                succ = self.type_succ.get(node_type(node))
                if not succ:
                    continue
                s = succ.most_common(1)[0][0]
                for e in self.graph.get(q, ()):
                    n = self.nodes.get(e.endpoint)
                    if n is not None and n.symbol == s:
                        e.seen_freq += 1
                        break

    def _dominant(self, key: str, name: str) -> Optional[str]:
        """The flow rule of (key, name) seen at least twice and more often than all
        others together; None while the value's origin is still unsettled."""
        rules = self.flow.get((key, name))
        if not rules:
            return None
        rule, n = max(rules.items(), key=lambda kv: kv[1])
        return rule if n >= 2 and 2 * n >= sum(rules.values()) else None

    def _stability(self, key: str, name: str) -> int:
        """How long a binding's bytes stay valid as a cache prefix: 0 across sessions
        (const), 1 within a session (copy), 2 growing within a session (extend),
        3 fresh every call (anything else, or unsettled)."""
        rule = self._dominant(key, name)
        return STABILITY.get(rule.partition(":")[0], 3) if rule else 3

    def _freeze_layout(self, site: ByLLMCallsite) -> bool:
        """Decide the site's binding order once: stable-sort the observed order by
        stability, so the session-constant and accumulating values lead and the fresh
        ones trail. Waits until every binding has two observations, then never moves
        again — each reorder breaks the prefix once."""
        p = self.proto.get(site.key)
        names = list(p.get("order", ())) if p else []
        if p is None or site.layout is not None or not names:
            return False
        if any(sum(self.flow.get((site.key, n), {}).values()) < 2 for n in names):
            return False
        site.layout = sorted(names, key=lambda n: self._stability(site.key, n))
        p["order"] = list(site.layout)   # the rebuild follows the order requests now use
        return True

    def _flow_prefix(self, key: str, name: str, obs_list: List[CallObservation]) -> Optional[str]:
        """For an accumulating binding, the bytes its next value is known to start
        with: the latest same-name value minus its closing delimiter."""
        if self._dominant(key, name) != "extend":
            return None
        w = next((v for e in reversed(obs_list) for nm, v in _named_values(e).items() if nm == name), None)
        return w[:-1] if w is not None and len(w) > 2 else None

    def _flow_value(self, key: str, name: str, obs_list: List[CallObservation]) -> Optional[str]:
        """Apply the dominant flow rule of (key, name) to the live session."""
        rule = self._dominant(key, name)
        if rule is None or rule == "extend":   # extend: only a prefix is known, see _flow_prefix
            return None
        if rule == "copy":
            return next((v for e in reversed(obs_list)
                         for nm, v in _named_values(e).items() if nm == name), None)
        kind, _, rest = rule.partition(":")
        if kind == "const":
            return rest
        src, _, arg = rest.partition("\x00")
        for e in reversed(obs_list):
            if e.key != src:
                continue
            if kind == "resp":
                return _value_at(e.response, json.loads(arg))
            if kind == "take":
                return _named_values(e).get(arg)
            if kind == "cand":
                picks = chosen_candidates(e)
                if not picks:
                    return None
                node = picks[0][1]
                if not arg:
                    return node
                parsed = parse_fields(node)
                return next((w for f, w in (parsed[1] if parsed else ()) if field_name(f) == arg), None)
        return None

    def resolve_user(self, key: str, obs_list: List[CallObservation]) -> Tuple[str, bool]:
        """Rebuild the head of `key`'s next user message from values the live session
        already carries; the bool says whether the whole message resolved."""
        site = self.sites.get(key)
        p = self.proto.get(key)
        if site is None or p is None:
            return "", False
        if isinstance(site, VisitByCallsite):
            return self._resolve_visit(site, p, obs_list)
        return self._resolve_byllm(site, p, obs_list)

    def _resolve_byllm(self, site: ByLLMCallsite, p: Dict[str, Any],
                       obs_list: List[CallObservation]) -> Tuple[str, bool]:
        parts = [site.context_desc]
        for name in p.get("order", ()):
            v = self._flow_value(site.key, name, obs_list)
            if v is None:
                pre = self._flow_prefix(site.key, name, obs_list)
                if pre is not None:              # the history so far: known head, open tail
                    parts.append(f"{name} = {pre}")
                return "\n".join(parts), False
            parts.append(f"{name} = {v}")
        if p.get("self"):
            v = self._resolve_repr(site.key, "self", p.get("self_fields"), obs_list)
            if v is None:
                return "\n".join(parts), False
            if p.get("self_gap"):
                parts.append("")
            line = f"self = {v} ---- {site.owner_sem}" if site.owner_sem else f"self = {v}"
            parts.append(line + p.get("self_rows", ""))
        return "\n".join(parts) + site.hint, True

    def _resolve_repr(self, key: str, tag: str, spec: Optional[Tuple[str, List[Tuple[str, str]]]],
                      obs_list: List[CallObservation]) -> Optional[str]:
        """Rebuild one structured repr field by field, or as a whole when it never
        parsed as one."""
        if spec is not None:
            tname, rows = spec
            vals = []
            for f, _ in rows:
                v = self._flow_value(key, f"{tag}.{field_name(f)}", obs_list)
                if v is None:
                    break
                vals.append((f, v))
            else:
                return build_fields(tname, vals)
        return self._flow_value(key, tag, obs_list)

    def _resolve_visit(self, site: VisitByCallsite, p: Dict[str, Any],
                       obs_list: List[CallObservation]) -> Tuple[str, bool]:
        zones = [f"Goal: {site.intent}"] if site.intent else []
        for tag, header in (("walker", "Walker:\n"), ("here", "Current node:\n")):
            if tag not in p:
                continue
            body = self._resolve_repr(site.key, tag, p[tag], obs_list)
            if body is None:
                return "\n\n".join(zones), False
            zones.append(header + body)
        block = self._flow_value(site.key, "cands", obs_list)
        if block is None:
            return "\n\n".join(zones), False
        zones.append("Candidates (choose by handle):\n" + block)
        return "\n\n".join(zones) + site.hint, True

    def rebuild(self, alpha: float = 0.05) -> None:
        """Rebuild the automaton from the observed call sequences"""
        if not self.seqs:
            return
        root = _State(None)
        for seq, cnt in self.seqs.items():
            root.n += cnt
            cur = root
            for key in seq:
                link = cur.kids.get(key)
                if link is None:
                    link = cur.kids[key] = [_State(key), 0]
                link[1] += cnt
                cur = link[0]
                cur.n += cnt
            cur.end += cnt
        eps = sqrt(0.5 * log(2 / alpha))
        red: List[_State] = [root]
        while True:
            blue = [(r, s) for r in red for s, (c, _) in r.kids.items() if all(c is not x for x in red)]
            if not blue:
                break
            parent, s = blue[0]
            child = parent.kids[s][0]
            for r in red:
                if r.symbol == child.symbol and _compat(r, child, eps):
                    parent.kids[s][0] = r
                    _fold(r, child)
                    break
            else:
                red.append(child)
        # Assign state IDs, then rebuild nodes and edges.
        ids: Dict[int, str] = {id(root): START}
        per_symbol: Dict[str, int] = defaultdict(int)
        order = [root]
        for st in order:  # Appending to `order` continues the traversal.
            for s, (c, _) in sorted(st.kids.items()):
                if id(c) not in ids:
                    ids[id(c)] = f"{c.symbol}#{per_symbol[c.symbol]}"
                    per_symbol[c.symbol] += 1
                    order.append(c)
        nodes: Dict[str, Node] = {}
        graph: Dict[str, List[Edge]] = defaultdict(list)
        exits: Dict[str, int] = defaultdict(int)
        for st in order:
            sid = ids[id(st)]
            if st is not root:
                nodes[sid] = Node(id=sid, symbol=st.symbol) #type: ignore
                if st.end:
                    exits[sid] = st.end
            for s, (c, cnt) in sorted(st.kids.items()):
                graph[sid].append(Edge(endpoint=ids[id(c)], taken_freq=cnt))
        self.nodes, self.graph, self.exit_freq = nodes, graph, exits

    # Prediction
    def _dist(self, walked: List[str]) -> Tuple[Dict[str, float], float, int]:
        """Next-symbol distribution under Witten-Bell backoff: every observed
        suffix of the history is blended shallow-to-deep, each with the say
        w = n/(n+T) its sample size n earns (T = distinct outcomes seen there).
        A deep context seen once shades — not overrides — the well-supported
        shorter ones. Returns (probs, end probability, deepest support n)."""
        ctx = ([START] + list(walked))[-self.k:]
        probs: Dict[str, float] = {}
        end = 0.0
        support = 0
        for i in range(len(ctx), -1, -1):  # i == len(ctx): empty suffix, the root
            node: Optional[TrieNode] = self.ctx
            for key in ctx[i:]:
                node = node.children.get(key)
                if node is None:
                    break
            if node is None or node.resolved == 0:
                continue
            n = node.resolved
            t = len(node.children) + (1 if node.end else 0)
            w = n / (n + t)
            probs = {x: (1 - w) * p for x, p in probs.items()}
            for x, c in node.children.items():
                probs[x] = probs.get(x, 0.0) + w * (c.count / n)
            end = (1 - w) * end + w * (node.end / n)
            support = n
        return probs, end, support

    def branch_probs(self, walked: List[str]) -> List[Tuple[str, float]]:
        """Return possible next calls ordered by probability."""
        probs, _, _ = self._dist(walked)
        return sorted(probs.items(), key=lambda kv: -kv[1])

    def end_prob(self, walked: List[str]) -> float:
        """Return the probability that the current session is complete."""
        return self._dist(walked)[1]

    def _step_ro(self, q: str, key: str) -> str:
        """Find the next state without changing the graph."""
        for e in self.graph.get(q, ()):
            n = self.nodes.get(e.endpoint)
            if n is not None and n.symbol == key:
                return e.endpoint
        return f"{key}#0"

    def _trace(self, walked: List[str]) -> str:
        """Find the graph state reached after the given calls."""
        q = START
        for key in walked:
            q = self._step_ro(q, key)
        return q

    def _gap_s(self, qa: str, qb: str, ka: str, kb: str) -> float:
        """Estimate the delay between two consecutive calls."""
        for e in self.graph.get(qa, ()):
            if e.endpoint == qb and e.gap_time:
                return e.gap_s
        xs = self.gap.get((ka, kb))
        return median(xs) if xs else 0.0

    def _gap_q(self, qa: str, qb: str, ka: str, kb: str, q: float) -> float:
        """Quantile of the delay between two consecutive calls."""
        for e in self.graph.get(qa, ()):
            if e.endpoint == qb and e.gap_time:
                return _quantile(e.gap_time, q)
        return _quantile(self.gap.get((ka, kb), []), q)

    def _duration_s(self, nid: str, key: str) -> float:
        """Estimate the total duration of a call."""
        node = self.nodes.get(nid)
        if node is not None and node.stats.engine_s:
            return node.stats.duration_s
        site = self.sites.get(key)
        return site.exec_stats.duration_s if site is not None else 0.0

    def _duration_qq(self, nid: str, key: str, q: float) -> float:
        """Quantile of a call's total duration."""
        node = self.nodes.get(nid)
        if node is not None and node.stats.engine_s:
            return node.stats.duration_q(q)
        site = self.sites.get(key)
        return site.exec_stats.duration_q(q) if site is not None else 0.0

    def predict(self, walked: List[str], max_steps: int = 8) -> List[Tuple[str, float, float]]:
        """Predict likely next calls, probabilities, and arrival times."""
        ctx = list(walked)
        q = self._trace(ctx)
        out: List[Tuple[str, float, float]] = []
        p, t = 1.0, 0.0
        while len(out) < max_steps:
            probs, end, support = self._dist(ctx)
            if not probs or support == 0:
                break
            key, p_step = max(probs.items(), key=lambda kv: kv[1])
            if end > p_step:
                break
            p *= p_step
            nq = self._step_ro(q, key)
            if ctx:
                t += self._gap_s(q, nq, ctx[-1], key)
            out.append((key, p, t))
            t += self._duration_s(nq, key)
            ctx.append(key)
            q = nq
        return out

    def predict_tree(self, walked: List[str], p_min: float = 0.02, horizon_s: float = 120.0,
                     max_nodes: int = 64, top_k: int = 3, max_depth: int = 8) -> List["PredictedCall"]:
        """Fan-out prediction: expand the top_k continuations at every step (not just
        the argmax chain) and report, per future callsite, the total probability mass
        of the paths that reach it and its earliest-arrival time quantiles. Times are
        seconds after the last call's completion — the anchor the gap samples share.
        Arrival spread compounds hop-wise as the root of the summed squared
        quantile gaps (independent-hop approximation)."""
        out: Dict[str, PredictedCall] = {}
        # frontier rows: (path probability, ctx, graph state, t50 so far, spread² so far, depth)
        frontier: List[Tuple[float, List[str], str, float, float, int]] = [
            (1.0, list(walked), self._trace(walked), 0.0, 0.0, 0)]
        expanded = 0
        while frontier and expanded < max_nodes:
            frontier.sort(key=lambda row: -row[0])
            path_p, ctx, q, t50, var, depth = frontier.pop(0)
            expanded += 1
            probs, _, support = self._dist(ctx)
            if support == 0:
                continue
            last = ctx[-1] if ctx else START
            for key, p_step in sorted(probs.items(), key=lambda kv: -kv[1])[:top_k]:
                p = path_p * p_step
                if p < p_min:
                    continue
                nq = self._step_ro(q, key)
                g50 = self._gap_q(q, nq, last, key, 0.5)
                g90 = self._gap_q(q, nq, last, key, 0.9)
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
                    d50 = self._duration_qq(nq, key, 0.5)
                    d90 = self._duration_qq(nq, key, 0.9)
                    frontier.append((p, ctx + [key], nq, a50 + d50,
                                     spread2 + (d90 - d50) ** 2, depth + 1))
        return sorted(out.values(), key=lambda c: -c.p)


@dataclass
class PredictedCall:
    """One future call the tree search expects, with arrival-time quantiles."""
    key: str
    p: float      # probability mass over every tree path reaching the call in the horizon
    t50: float    # earliest-arrival median, seconds after the last call's completion
    t90: float    # p90 of that same arrival
