"""Data model for history-based workflow recovery.

Nothing here parses text. `parser.py` turns raw /v1/chat/completions requests into
these objects; this module only holds them and the statistics derived from them.

Naming: a *callsite* is one `by llm()` / `visit by` in the source program. We never
see the source, so a callsite is identified by its `key`: a hash of the parts of the
request that are the same every time that line of code runs (system prompt, the
signature/description header, model, tool names). See parser.callsite_key.
"""
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

@dataclass
class LLMCallsite:
    key: str
    kind: CallsiteType
    label: str                       # human name: "Agent.summarize" or "visit: <intent...>"
    model: str
    system_prompt: str
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class ByLLMCallsite(LLMCallsite):
    context_desc: str = ""                                    # header + schema rows, fixed text
    qualname: str = ""                                        # "Agent.summarize"
    params: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, type, sem)
    return_type: str = ""
    sem: str = ""
    tools: List[str] = field(default_factory=list)            # tool names incl. finish_tool


@dataclass
class VisitByCallsite(LLMCallsite):
    intent: str = ""
    select: str = ""                                          # "exactly one" / "exactly 3" / "between 1 and 3" / "all"
    candidates: List[str] = field(default_factory=list)       # union of handles seen across calls
    edges: List[str] = field(default_factory=list)            # union of edge descriptions seen


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

class Program:
    """Everything learned about one program from its sessions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.callsites: Dict[str, LLMCallsite] = {}
        self.sessions: Dict[str, Session] = {}
        # order statistics
        self.entry: Counter = Counter()                                # first callsite of a session
        self.succ: Dict[str, Counter] = defaultdict(Counter)           # key -> Counter[next key | END]
        self.succ2: Dict[Tuple[str, str], Counter] = defaultdict(Counter)  # (prev, key) -> Counter[next]
        # time statistics
        self.gap: Dict[Tuple[str, str], List[float]] = defaultdict(list)   # (a, b) -> b.arrive - a.done
        self.exec: Dict[str, List[float]] = defaultdict(list)          # key -> engine seconds per instance
        self.turns: Dict[str, List[int]] = defaultdict(list)           # key -> ReAct turns per instance
        self.tool_gap: Dict[str, List[float]] = defaultdict(list)      # key -> seconds between turns

    # ------------------------------------------------------------------ building
    def add_callsite(self, site: LLMCallsite) -> LLMCallsite:
        """Register a callsite; a VISITBY site seen again merges its candidate/edge sets."""
        known = self.callsites.get(site.key)
        if known is None:
            self.callsites[site.key] = site
            return site
        if isinstance(known, VisitByCallsite) and isinstance(site, VisitByCallsite):
            for h in site.candidates:
                if h not in known.candidates:
                    known.candidates.append(h)
            for e in site.edges:
                if e not in known.edges:
                    known.edges.append(e)
        return known

    def observe_call(self, prev: Optional[CallInstance], inst: CallInstance, nxt: Optional[CallInstance]) -> None:
        """Fold one finished call instance into the statistics: `prev` is the instance before
        it in the session (None at entry), `nxt` the one after (None: the session ended)."""
        nxt_key = nxt.key if nxt is not None else END
        if prev is None:
            self.entry[inst.key] += 1
        self.succ[inst.key][nxt_key] += 1
        if prev is not None:
            self.succ2[(prev.key, inst.key)][nxt_key] += 1
        if nxt is not None:
            self.gap[(inst.key, nxt.key)].append(nxt.t_arrive - inst.t_done)
        self.exec[inst.key].append(inst.exec_s)
        self.turns[inst.key].append(len(inst.turns))
        self.tool_gap[inst.key].extend(inst.tool_gaps)

    def observe(self, session: Session) -> None:
        """Offline: fold a whole finished session in."""
        self.sessions[session.id] = session
        seq = session.calls
        for i, inst in enumerate(seq):
            self.observe_call(seq[i - 1] if i else None, inst, seq[i + 1] if i + 1 < len(seq) else None)

    def close_session(self, session: Session) -> None:
        """Online: the session is over. Every instance but the last was observed when its
        successor arrived (see serve.session); this folds the last one with END."""
        self.sessions[session.id] = session
        seq = session.calls
        if seq:
            self.observe_call(seq[-2] if len(seq) > 1 else None, seq[-1], None)

    # ------------------------------------------------------------------ queries (planner interface)
    def branch_candidates(self, key: str, prev: Optional[str] = None) -> List[Tuple[str, float]]:
        """Successor callsites of `key`, most likely first, END excluded. With `prev` the
        second-order table is used when it has data (loops look different by lap)."""
        table = self.succ2.get((prev, key)) if prev is not None else None
        if not table:
            table = self.succ.get(key)
        if not table:
            return []
        total = sum(table.values())
        out = [(k, n / total) for k, n in table.items() if k != END]
        out.sort(key=lambda kv: -kv[1])
        return out

    def end_prob(self, key: str) -> float:
        table = self.succ.get(key)
        return table[END] / sum(table.values()) if table else 0.0

    def support(self, key: str, nxt: str) -> int:
        """How many times the edge key -> nxt was observed."""
        return self.succ.get(key, Counter())[nxt]

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

    def tool_gap_s(self, key: str, default: float = 0.0) -> float:
        """Typical client-side gap between two ReAct turns of `key` (the tool runs then)."""
        xs = self.tool_gap.get(key)
        return median(xs) if xs else default

    def predict(self, key: str, prev: Optional[str] = None, max_steps: int = 8) -> List[Tuple[str, float, float]]:
        """Most likely path after `key`: [(next key, probability of reaching it, seconds until it
        arrives)], following the top successor at every step and compounding probabilities."""
        out: List[Tuple[str, float, float]] = []
        p, t = 1.0, 0.0
        cur, before = key, prev
        while len(out) < max_steps:
            cands = self.branch_candidates(cur, before)
            if not cands:
                break
            nxt, pn = cands[0]
            p *= pn
            t += self.gap_s(cur, nxt)
            out.append((nxt, p, t))
            t += self.exec_s(nxt)
            before, cur = cur, nxt
        return out

    # ------------------------------------------------------------------ inspection
    def label(self, key: str) -> str:
        return self.callsites[key].label if key in self.callsites else key

    def describe(self) -> str:
        lines = [f"program {self.name}: {len(self.callsites)} callsites, {len(self.sessions)} sessions"]
        total_entry = sum(self.entry.values()) or 1
        for key, n in self.entry.most_common():
            lines.append(f"  entry  {self.label(key):45} p={n / total_entry:.2f} (n={n})")
        for key in self.callsites:
            site = self.callsites[key]
            xs = self.exec.get(key, [])
            turns = self.turns.get(key, [])
            tg = self.tool_gap.get(key)
            lines.append(f"  {site.label:45} {site.kind.name:7} {site.model:28} "
                         f"n={len(xs)} exec={median(xs) if xs else 0:.2f}s"
                         + (f" turns={median(turns):.0f}" if turns and max(turns) > 1 else "")
                         + (f" tool_gap={median(tg):.2f}s" if tg else ""))
            table = self.succ.get(key, Counter())
            total = sum(table.values()) or 1
            for nxt, cnt in table.most_common():
                g = self.gap.get((key, nxt))
                lines.append(f"      -> {self.label(nxt):42} p={cnt / total:.2f} (n={cnt})"
                             + (f" gap={median(g):.2f}s" if g else ""))
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
            "entry": dict(self.entry),
            "succ": {k: dict(c) for k, c in self.succ.items()},
            "succ2": {f"{a}\x00{b}": dict(c) for (a, b), c in self.succ2.items()},
            "gap": {f"{a}\x00{b}": xs for (a, b), xs in self.gap.items()},
            "exec": dict(self.exec),
            "turns": dict(self.turns),
            "tool_gap": dict(self.tool_gap),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Program":
        p = cls(d["name"])
        for k, s in d["callsites"].items():
            s = dict(s)
            kind = CallsiteType[s.pop("kind")]
            site_cls = VisitByCallsite if kind is CallsiteType.VISITBY else ByLLMCallsite
            if site_cls is ByLLMCallsite:
                s["params"] = [tuple(p_) for p_ in s.get("params", [])]
            p.callsites[k] = site_cls(kind=kind, **s)
        p.entry = Counter(d["entry"])
        for k, c in d["succ"].items():
            p.succ[k] = Counter(c)
        for k, c in d["succ2"].items():
            a, b = k.split("\x00")
            p.succ2[(a, b)] = Counter(c)
        for k, xs in d["gap"].items():
            a, b = k.split("\x00")
            p.gap[(a, b)] = list(xs)
        for name in ("exec", "turns", "tool_gap"):
            for k, xs in d[name].items():
                getattr(p, name)[k] = list(xs)
        return p
