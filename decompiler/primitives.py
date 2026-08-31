from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from statistics import median
from collections import defaultdict, Counter
from math import log, sqrt
TOOL_BLOCK_HEADER = "\n# Calling tools\n"  # Header for tools included in the system prompt.


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


@dataclass
class CallMetadata:
    model: str
    temperature: Optional[float] = None      # Optional request setting.
    max_tokens: Optional[int] = None
    system_prompt: str = ""                  # Required prefix from the first token.
    response_format: Optional[Dict[str, Any]] = None   # Shared request format.
    stable_prefix: str = ""                  # Shared prefix of first user messages.
    prefix_n: int = 0                        # Messages included in stable_prefix.
    exec_stats: ExecStats = field(default_factory=ExecStats)  # Running timing data.

    @property
    def key(self) -> str:
        """Return a stable identifier for this call."""
        raise NotImplementedError

    @property
    def label(self) -> str:
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


@dataclass
class VisitByCallsite(CallMetadata):
    """Describe an LLM call that selects from a list."""
    intent: str = ""
    select: str = ""                                          # "exactly one" / "exactly 3" / "between 1 and 3" / "all"

    @property
    def key(self) -> str:
        return f"visit: {self.intent} by {self.model}"

    @property
    def label(self) -> str:
        intent = self.intent or "(no intent)"
        return "visit: " + (intent[:50] + "..." if len(intent) > 50 else intent)

# ----------------------------------------------------------------------- program model

Callsite = Union[ByLLMCallsite, VisitByCallsite]

START = "^"  # Marker placed before the first call in a session.


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
    n_turns: int = 1
    tool_gaps: List[float] = field(default_factory=list)
    candidates: List[Tuple[str, str]] = field(default_factory=list)  # Options offered by a visit call.


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
        if site.prefix_n:
            if known.prefix_n:
                known.stable_prefix = _lcp(known.stable_prefix, site.stable_prefix)
            else:
                known.stable_prefix = site.stable_prefix
            known.prefix_n += site.prefix_n
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

    def update_graph(self, walked: List[str], obs: List[CallObservation]) -> None:
        """Record calls and timing data from one completed session, and update"""
        if not obs:
            return
        self.seqs[tuple(walked)] += 1
        self.n_sessions += 1
        q, prev = START, None
        for key, ob in zip(walked, obs):
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
            # Candidate counts are updated after their call keys are known.
            q, prev = nid, ob
        self.exit_freq[q] += 1
        self._trie_fold(walked)

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
    def _context_node(self, walked: List[str]) -> Tuple[TrieNode, int]:
        """Find the longest observed suffix of the current call history."""
        ctx = ([START] + list(walked))[-self.k:]
        for i in range(len(ctx)):
            node: Optional[TrieNode] = self.ctx
            for key in ctx[i:]:
                node = node.children.get(key)
                if node is None:
                    break
            if node is not None and node.resolved > 0:
                return node, len(ctx) - i
        return self.ctx, 0

    def branch_probs(self, walked: List[str]) -> List[Tuple[str, float]]:
        """Return possible next calls ordered by probability."""
        node, depth = self._context_node(walked)
        if depth == 0 or node.resolved == 0:
            return []
        out = [(k, c.count / node.resolved) for k, c in node.children.items()]
        out.sort(key=lambda kv: -kv[1])
        return out

    def end_prob(self, walked: List[str]) -> float:
        """Return the probability that the current session is complete."""
        node, depth = self._context_node(walked)
        return node.end / node.resolved if depth and node.resolved else 0.0

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

    def _duration_s(self, nid: str, key: str) -> float:
        """Estimate the total duration of a call."""
        node = self.nodes.get(nid)
        if node is not None and node.stats.engine_s:
            return node.stats.duration_s
        site = self.sites.get(key)
        return site.exec_stats.duration_s if site is not None else 0.0

    def predict(self, walked: List[str], max_steps: int = 8) -> List[Tuple[str, float, float]]:
        """Predict likely next calls, probabilities, and arrival times."""
        ctx = list(walked)
        q = self._trace(ctx)
        out: List[Tuple[str, float, float]] = []
        p, t = 1.0, 0.0
        while len(out) < max_steps:
            node, depth = self._context_node(ctx)
            if depth == 0 or not node.children:
                break
            key, child = max(node.children.items(), key=lambda kv: kv[1].count)
            if node.end > child.count:
                break
            p *= child.count / node.resolved
            nq = self._step_ro(q, key)
            if ctx:
                t += self._gap_s(q, nq, ctx[-1], key)
            out.append((key, p, t))
            t += self._duration_s(nq, key)
            ctx.append(key)
            q = nq
        return out
