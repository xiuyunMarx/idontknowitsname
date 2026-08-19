"""Incremental prefill: sessions, feed store, and prompt builders (engine-free).

Three tiers of prompt stability:
  invariant     — compile-time constant; AsyncByLLM prebuilds it, warmed at deploy(if enabled).
  quasi-static  — graph-dependent but stable across requests (a visit-by site's
                  here/candidates zones and its choice schema); fed when the graph
                  is known, reused until the graph changes. If the visit-by is the
                  very first LLM call on a fresh graph, nothing can prewarm it —
                  accepted limitation; every later request rides the fed prefix.
  per-request   — argument bindings; fed one by one as values become ready and
                  rendered in ARRIVAL order. Declaration order has no cross-request
                  cache value (binding values are unique per request), so arrival
                  order is strictly better for overlap.

Feeding is ADVISORY: the final prompt is always rebuilt from the real arguments,
replaying the session's arrival order only while the values still match — a stale
feed degrades to a cache miss from the first divergent line, never a wrong prompt.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FeedItem:
    name: str
    repr_text: str
    t: float = field(default_factory=time.monotonic)


@dataclass
class PrefillSession:
    """Arrival-ordered record of the bindings speculatively fed for one upcoming call."""

    key: str
    sid: str
    fed: List[FeedItem] = field(default_factory=list)

    def feed(self, name: str, repr_text: str) -> bool:
        """Record one binding. Returns True if the partial prefix changed (a new warm
        is worth issuing). Re-feeding a name with the same text is a no-op; with a
        different text it truncates the session at that item — the cached suffix
        beyond it is stale anyway."""
        for i, item in enumerate(self.fed):
            if item.name == name:
                if item.repr_text == repr_text:
                    return False
                del self.fed[i:]
                break
        self.fed.append(FeedItem(name=name, repr_text=repr_text))
        return True

    def lines(self) -> List[str]:
        return [f"{item.name} = {item.repr_text}" for item in self.fed]


class FeedStore:
    """All live prefill state: per-request sessions and per-visit-site graph context."""

    def __init__(self) -> None:
        self.sessions: Dict[Tuple[str, str], PrefillSession] = {}
        self.visit_ctx: Dict[str, Dict[str, str]] = {}  # key -> {here, candidates, schema?}

    def session(self, key: str, sid: str, create: bool = False) -> Optional[PrefillSession]:
        s = self.sessions.get((key, sid))
        if s is None and create:
            s = PrefillSession(key=key, sid=sid)
            self.sessions[(key, sid)] = s
        return s

    def feed_param(self, key: str, sid: str, name: str, repr_text: str) -> Tuple[PrefillSession, bool]:
        s = self.session(key, sid, create=True)
        assert s is not None
        return s, s.feed(name, repr_text)

    def drop_session(self, key: str, sid: str) -> None:
        self.sessions.pop((key, sid), None)

    def sessions_for(self, key: str, ttl: float = 600.0) -> List[PrefillSession]:
        """Live sessions pending for a call site, purging ones whose last feed is
        older than `ttl` (a call that never came; the cache has moved on anyway)."""
        now = time.monotonic()
        dead = [ks for ks, s in self.sessions.items() if s.fed and now - s.fed[-1].t > ttl]
        for ks in dead:
            self.sessions.pop(ks, None)
        return [s for (k, _sid), s in self.sessions.items() if k == key]

    def set_visit_ctx(self, key: str, here: str = "", candidates: str = "", schema: Optional[Any] = None) -> bool:
        """Store a visit site's graph-derived stable zones. Returns True on change."""
        ctx: Dict[str, Any] = {"here": here or "", "candidates": candidates or ""}
        if schema is not None:
            ctx["schema"] = schema
        if self.visit_ctx.get(key) == ctx:
            return False
        self.visit_ctx[key] = ctx
        return True

    def get_visit_ctx(self, key: str) -> Optional[Dict[str, Any]]:
        return self.visit_ctx.get(key)


# ---------------------------------------------------------------------------
# Prompt builders (take an AsyncByLLM-shaped `fn`; no engine, no vllm)
# ---------------------------------------------------------------------------
def partial_user_prompt(fn: Any, session: PrefillSession) -> str:
    """The warmable prefix for a function call site: invariant head + fed bindings
    in arrival order. Extending this string is exactly extending the KV prefix."""
    return "\n".join([fn.invariant_user_prefix, *session.lines()])


def final_messages(fn: Any, params: Dict[str, Any], session: Optional[PrefillSession] = None) -> List[Dict[str, str]]:
    """Rebuild the full prompt from the REAL arguments, replaying the session's
    arrival order while values still match. Verification cuts at the first
    divergence: bindings after it (and unfed ones) render in declaration order,
    so a stale feed costs cache misses from that line on — never a wrong prompt."""
    used: set = set()
    lines = [fn.invariant_user_prefix]
    if session is not None:
        for item in session.fed:
            if item.name in params and item.name not in used and repr(params[item.name]) == item.repr_text:
                lines.append(f"{item.name} = {item.repr_text}")
                used.add(item.name)
            else:
                break
    for p in fn.decl.params:
        name = p["name"]
        if name in params and name not in used:
            lines.append(f"{name} = {params[name]!r}")
            used.add(name)
    zones = ["\n".join(lines)]
    if params.get("self") is not None and fn.decl.owner_sem:
        zones.append(f"self = {params['self']!r} ---- {fn.decl.owner_sem}")
    return [{"role": "system", "content": fn.invariant_system}, {"role": "user", "content": "\n\n".join(zones)}]


# ---------------------------------------------------------------------------
# Provenance evaluation: turn compile-time value-source specs into repr texts
# using the runtime state visible at a predecessor call
# ---------------------------------------------------------------------------
def _split_top_commas(text: str) -> List[str]:
    parts: List[str] = []
    buf = ""
    depth = 0
    in_str = False
    quote = ""
    esc = False
    for ch in text:
        if in_str:
            buf += ch
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        elif ch in "\"'":
            in_str = True
            quote = ch
            buf += ch
        elif ch in "([{":
            depth += 1
            buf += ch
        elif ch in ")]}":
            depth -= 1
            buf += ch
        elif ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    return parts


def parse_arch_fields(desc: str) -> Dict[str, str]:
    """Field repr texts out of a byllm `_describe_arch` rendering:
    `interact(message='hi', chat_history=[...])` -> {name: repr_text}.
    The head may be a sem string containing parens, so the field group is the
    balanced (...) that closes exactly at the end of the description."""
    text = desc.rstrip()
    if not text.endswith(")"):
        return {}
    for i, ch in enumerate(text):
        if ch != "(":
            continue
        depth = 0
        in_str = False
        quote = ""
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    in_str = False
            elif c in "\"'":
                in_str = True
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
        else:
            return {}
        if j == len(text) - 1:  # this "(" closes at the very end: the field group
            out: Dict[str, str] = {}
            for part in _split_top_commas(text[i + 1:j]):
                m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s\([^=]*\))?=(.*)$", part, re.S)
                if m:
                    out[m.group(1)] = m.group(2)
            return out
    return {}


def extract_walker_fields(user_text: str) -> Dict[str, str]:
    """Walker field reprs from a served route prompt's `Walker:` zone."""
    m = re.search(r"(?:^|\n\n)Walker:\n(.*?)(?:\n\n[A-Z]|\Z)", user_text, re.S)
    if m is None:
        return {}
    return parse_arch_fields(m.group(1).strip())


_SLICE_RE = re.compile(r"\[[-\d:, ]*\]")


def eval_provenance(spec: Dict[str, Any], walker_fields: Optional[Dict[str, str]] = None, results: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Repr text for one provenance spec, or None when not evaluable yet (or ever).
    `walker_fields` maps field name -> repr text seen in a predecessor's prompt;
    `results` maps callsite key -> repr text of its parsed return value."""
    kind = spec.get("kind")
    if kind == "const":
        return repr(spec.get("value"))
    if kind == "ret":
        return (results or {}).get(spec.get("of") or "")
    if kind == "field" and spec.get("scope") == "visitor" and walker_fields:
        raw = walker_fields.get(spec.get("attr") or "")
        if raw is None:
            return None
        raw = raw.strip()
        sl = spec.get("slice") or ""
        if not sl:
            return raw  # byte-exact repr as it appeared in the predecessor's prompt
        if not _SLICE_RE.fullmatch(sl):
            return None
        try:
            value = ast.literal_eval(raw)  # fails on _safe_repr-truncated text -> None
            return repr(eval("v" + sl, {"__builtins__": {}}, {"v": value}))
        except Exception:
            return None
    return None  # here/self scopes need the successor's own node instance; unknown never feeds


# ---------------------------------------------------------------------------
# Approach B: arrival-order reorder of a byllm-rendered binding zone
# ---------------------------------------------------------------------------
_BIND_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*) = ")


def _binding_block(lines: List[str], param_names: set) -> Optional[Tuple[int, int]]:
    """[start, end) of the contiguous binding block in a byllm user message.

    A binding is one line (`repr` escapes newlines) of the form `name = ...` with
    a declared param name. Bails (None) if matching lines are non-contiguous —
    e.g. a sem text line collides with the pattern — reordering then would risk
    corrupting prose, and bailing just means decl-order service as usual."""
    idxs = [i for i, ln in enumerate(lines) if (m := _BIND_RE.match(ln)) and m.group(1) in param_names]
    if not idxs or idxs != list(range(idxs[0], idxs[-1] + 1)):
        return None
    return idxs[0], idxs[-1] + 1


def _match_len(session: PrefillSession, block: List[str]) -> int:
    """Longest prefix of the session's fed items present VERBATIM in the block."""
    present = set(block)
    n = 0
    for item in session.fed:
        if f"{item.name} = {item.repr_text}" in present:
            n += 1
        else:
            break
    return n


def reorder_binding_zone(user_text: str, fn: Any, sessions: List[PrefillSession]) -> Tuple[str, Optional[PrefillSession], int]:
    """Reorder the binding zone of a byllm-rendered user message into the arrival
    order of the best-matching session, so the fed KV prefix becomes a strict
    prefix of the served prompt. Only lines that arrived verbatim are moved —
    fed text is never injected — and any mismatch degrades to the original
    declaration order. Returns (text, matched_session, matched_count)."""
    param_names = {p["name"] for p in fn.decl.params}
    lines = user_text.split("\n")
    span = _binding_block(lines, param_names)
    if span is None:
        return user_text, None, 0
    block = lines[span[0]:span[1]]
    best: Optional[PrefillSession] = None
    best_n = 0
    for s in sessions:
        n = _match_len(s, block)
        if n > best_n:
            best, best_n = s, n
    if best is None or best_n == 0:
        return user_text, None, 0
    fed_lines = [f"{item.name} = {item.repr_text}" for item in best.fed[:best_n]]
    moved = set(fed_lines)
    lines[span[0]:span[1]] = fed_lines + [ln for ln in block if ln not in moved]
    return "\n".join(lines), best, best_n


# ---------------------------------------------------------------------------
# Visit-by context extraction (server-side learning from served route prompts)
# ---------------------------------------------------------------------------
_VISIT_CTX_RE = re.compile(r"\n\nCurrent node:\n(.*?)\n\nCandidates \(choose by handle\):\n(.*?)(?:\n\nWalker:\n|\Z)", re.S)


def extract_visit_ctx(user_text: str) -> Optional[Tuple[str, str]]:
    """Pull (here, candidates) out of a served route prompt — only under the
    cache-aware layout (stable zones BEFORE Walker); in the legacy layout the
    Walker zone precedes them, the pair is not a prefix, and warming it would
    never hit, so we learn nothing."""
    walker_pos = user_text.find("\n\nWalker:\n")
    node_pos = user_text.find("\n\nCurrent node:\n")
    if node_pos < 0 or (0 <= walker_pos < node_pos):
        return None
    m = _VISIT_CTX_RE.search(user_text)
    if m is None:
        return None
    return m.group(1), m.group(2)


_CAND_LINE_RE = re.compile(r"^([A-Za-z0-9_]+)\)\s")


def parse_candidate_handles(candidates: str) -> List[str]:
    """The choice handles of a route prompt's candidates zone, in zone order.
    Lines are `handle) <description>`; the description's shape varies (sems,
    arrows, field dumps), but the handle itself is `_slugify(NodeType.__name__)`
    plus an optional `_suffix` — so callers map handles back to node types by
    slug matching, never by parsing the description."""
    out: List[str] = []
    for line in candidates.splitlines():
        m = _CAND_LINE_RE.match(line.strip())
        if m is not None:
            out.append(m.group(1))
    return out


def slugify_handle(text: str) -> str:
    """Byte-mirror of visit_routing.jac's _slugify (handle base construction)."""
    s = re.sub(r"\W+", "_", str(text)).strip("_")
    if not s:
        return "n"
    if not (s[0].isalpha() or s[0] == "_"):
        s = "n_" + s
    return s
