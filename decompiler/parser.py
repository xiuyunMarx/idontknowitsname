"""Decompile one /v1/chat/completions request body into the callsite that produced it.

The text formats parsed here are byllm's (jaclang/byllm/impl/mtir.impl.jac::MTRuntime.factory,
jaclang/byllm/visit_routing.jac::route_visit, jaclang/byllm/schema.jac::inject_schema_hint,
jaclang/byllm/tool_protocol.jac).

    site, extras = decompose(body)        # body: parsed request JSON
    is_continuation(prev_body, body)      # next ReAct turn / typed retry of the same call
    relayout_body(body, layout)           # re-emit the call message in a frozen binding order
    parse_fields(repr_text)               # `Type(a=1, b='x')` -> type text and field rows
    chosen_candidates(observation)        # the handles a visit reply selected, in reply order
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import primitives as _prim   # callsite dataclasses; module import: primitives imports this module too

ROUTE_SYSTEM = ("You are routing a graph walker. Choose which candidate node(s) the walker should "
                "visit next, by handle. Return only valid handles.")
HINT_HEADER = "Schema requirements:"
TOOL_RESPONSE_TAG = "<tool_response>"
SCHEMA_INDENT = "      "  # 6 spaces before every schema row
TOOL_BLOCK_HEADER = "\n# Calling tools\n"  # Header for tools included in the system prompt.

# "Owner.func(a: T, b: U) -> R --- sem"; the sem may itself contain " -> " so ret is lazy
_HEADER = re.compile(r"^(?P<qual>[\w.]+)\((?P<sig>.*?)\)(?: -> (?P<ret>.*?))?(?: --- (?P<sem>.*))?$")
_SCHEMA_ROW = re.compile(r"^(?P<name>\w+): (?P<type>.*?)(?: ---- (?P<sem>.*))?$")
_BINDING = re.compile(r"^(?P<name>[A-Za-z_]\w*) = (?P<value>.*)$")
_SELF = re.compile(r"^self = (?P<view>.*?)(?: ---- (?P<sem>.*))?$")
_CANDIDATE = re.compile(r"^(?P<handle>\w+)\) (?P<rest>.*)$")
_RUNS = re.compile(r" \[runs: (?P<runs>.*)\]$")


@dataclass
class CallExtras:
    """Per-call facts that identify nothing; they feed the session's CallObservation."""
    turn: int = 0                                                    # ReAct turn of this request
    candidates: List[Tuple[str, str]] = field(default_factory=list)  # (handle, node repr) offered
    bindings: Dict[str, str] = field(default_factory=dict)           # Parameter reprs of a byllm call
    self_view: Optional[str] = None                                  # Repr of the owning object
    walker: Optional[str] = None                                     # Repr of a visit call's walker
    here: Optional[str] = None                                       # Repr of the visit call's node
    cand_block: str = ""                                             # Raw candidate lines
    user_text: str = ""                                              # Raw first user message


# --------------------------------------------------------------------------- small helpers
def _content(msg: Dict[str, Any]) -> str:
    """Text of a message; multimodal content lists are reduced to their text parts."""
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


def _system(req: Dict[str, Any]) -> str:
    msgs = req.get("messages") or []
    return _content(msgs[0]) if msgs and msgs[0].get("role") == "system" else ""


def _call_user(req: Dict[str, Any]) -> str:
    """The user message that carries the call: the FIRST plain user message. Later user
    messages are tool responses (text tool protocol) or byllm's typed-retry feedback."""
    for m in req.get("messages") or []:
        if m.get("role") == "user" and not _content(m).startswith(TOOL_RESPONSE_TAG):
            return _content(m)
    return ""


def _strip_hint(text: str) -> str:
    """inject_schema_hint appends 'Schema requirements:\\n- ...' to the user message — after a
    blank line in string content, or as a separate text part in multimodal content (which
    _content joins with a single newline)."""
    for sep in ("\n\n" + HINT_HEADER + "\n", "\n" + HINT_HEADER + "\n"):
        i = text.rfind(sep)
        if i >= 0:
            return text[:i]
    return text


def _turn(req: Dict[str, Any]) -> int:
    """ReAct turn index = number of tool results already in the transcript."""
    msgs = req.get("messages") or []
    native = sum(1 for m in msgs if m.get("role") == "tool")
    if native:
        return native
    return sum(1 for m in msgs if m.get("role") == "user" and _content(m).startswith(TOOL_RESPONSE_TAG))


def _balanced(text: str, start: int) -> int:
    """Index just past the ')' matching the '(' at text[start]; -1 if unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


# --------------------------------------------------------------------------- by llm()
def split_byllm_user(user: str) -> Tuple[str, List[str], Dict[str, str], Optional[str], Optional[str]]:
    """Split a byllm user message into (context_desc, param names, bindings, self view, self sem).

    Layout (mtir.impl.jac): header line, schema rows (6-space indent), bindings
    `name = repr`, then optionally a blank line and `self = repr ---- sem`
    (+ indented member rows). repr() escapes newlines inside strings, so a binding
    is one line except for objects with a custom multi-line __repr__; continuation
    lines that are not a known binding are glued to the previous value."""
    user = _strip_hint(user)
    lines = user.split("\n")
    h, end = _locate_header(lines)
    names = _sig_names(lines[h])
    ctx = lines[h:end]
    lo, hi = (0, h) if h > 0 else (end, len(lines))   # the value region: before a trailing header, else after it
    spans, self_view, self_sem = _scan_bindings(lines, names, lo, hi)
    bindings: Dict[str, str] = {}
    for name, start, stop in spans:
        bindings[name] = "\n".join(lines[start:stop])[len(name) + 3:]   # past "name = "
    return "\n".join(ctx), names, bindings, self_view, self_sem


def _locate_header(lines: List[str]) -> Tuple[int, int]:
    """(index of the header line, index past its schema rows). byllm puts the header
    first; the server's cache-aware layout moves header and schema rows to the END
    of the message so the values in front are a prefix shared across callsites."""
    for h in range(len(lines) - 1, 0, -1):
        if _HEADER.match(lines[h]) and all(l.startswith(SCHEMA_INDENT) for l in lines[h + 1:]):
            return h, len(lines)
    return 0, _schema_end(lines)


def _sig_names(header: str) -> List[str]:
    """Parameter names in the header's signature, in declaration order."""
    m = _HEADER.match(header)
    if not m:
        return []
    return [part.split(":")[0].split("=")[0].strip() for part in m["sig"].split(",") if part.strip()]


def _schema_end(lines: List[str]) -> int:
    """Index of the first line after the header and its schema rows."""
    i = 1
    while i < len(lines) and lines[i].startswith(SCHEMA_INDENT):
        i += 1
    return i


def _scan_bindings(lines: List[str], names: List[str], lo: int, hi: int) -> Tuple[List[Tuple[str, int, int]], Optional[str], Optional[str]]:
    """Binding blocks in lines[lo:hi] as (name, first line, line past the last),
    plus the self view and sem. A line that is not a known binding continues the
    previous binding's value (multi-line __repr__)."""
    spans: List[List[Any]] = []
    self_view: Optional[str] = None
    self_sem: Optional[str] = None
    last: Optional[str] = None
    in_self = False
    for i in range(lo, hi):
        line = lines[i]
        sm = _SELF.match(line)
        if sm and not in_self:
            self_view = sm["view"]
            self_sem = sm["sem"]
            in_self = True
            last = None
        elif in_self:
            pass  # self's indented type-member rows: not a binding
        else:
            bm = _BINDING.match(line)
            if bm and (bm["name"] in names or not names or last is not None or not spans):
                spans.append([bm["name"], i, i + 1])
                last = bm["name"]
            elif last is not None and line != "":
                spans[-1][2] = i + 1
    return [(n, s, e) for n, s, e in spans], self_view, self_sem


def relayout_user(user: str, layout: List[str], header_last: bool = False) -> str:
    """The same byllm user message with its binding blocks in `layout` order (names
    absent from `layout` keep their relative order after the listed ones) and, with
    `header_last`, the header and schema rows moved behind the values so the
    accumulating values in front are a prefix every callsite of the session
    shares. Blocks and hint tail are untouched byte for byte; idempotent. Returns
    the input unchanged when the blocks are not contiguous."""
    body = _strip_hint(user)
    lines = body.split("\n")
    h, end = _locate_header(lines)
    lo, hi = (0, h) if h > 0 else (end, len(lines))
    spans, _, _ = _scan_bindings(lines, _sig_names(lines[h]), lo, hi)
    if len(spans) < 1 or sum(e - s for _, s, e in spans) != spans[-1][2] - spans[0][1]:
        return user
    rank = {n: i for i, n in enumerate(layout)}
    order = sorted(spans, key=lambda sp: rank.get(sp[0], len(layout)))   # stable
    if order == spans and (h > 0) == header_last:
        return user
    a, b = spans[0][1], spans[-1][2]
    values = lines[lo:a] + [l for _, s, e in order for l in lines[s:e]] + lines[b:hi]
    ctx = lines[h:end]
    out = values + ctx if header_last else ctx + values
    return "\n".join(out) + user[len(body):]


def relayout_body(body: Dict[str, Any], layout: List[str], header_last: bool = False) -> bool:
    """Apply relayout_user to the request's call message in place (string content
    only). True when the message changed."""
    for m in body.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):   # byllm sends the message as text parts: rewrite them as one
            parts = [p for p in c if isinstance(p, dict) and p.get("type") == "text"]
            if not parts or len(parts) != len(c):
                return False
            text = _content(m)
            if text.startswith(TOOL_RESPONSE_TAG):
                return False
            new = relayout_user(text, layout, header_last)
            if new != text:
                m["content"] = [{"type": "text", "text": new}]
                return True
            return False
        if isinstance(c, str) and not c.startswith(TOOL_RESPONSE_TAG):
            new = relayout_user(c, layout, header_last)
            if new != c:
                m["content"] = new
                return True
            return False
    return False


def _parse_byllm(body: Dict[str, Any]) -> Tuple[_prim.ByLLMCallsite, CallExtras]:
    system = _system(body)
    ctx, names, bindings, self_view, self_sem = split_byllm_user(_call_user(body))
    header = ctx.split("\n", 1)[0]
    m = _HEADER.match(header)
    qual = m["qual"] if m else header[:40]
    sig_types: Dict[str, str] = {}
    if m:
        for part in m["sig"].split(","):
            if ":" in part:
                n, t = part.split(":", 1)
                sig_types[n.strip()] = t.strip()
    sems: Dict[str, str] = {}
    views: Dict[str, str] = {}
    for row in ctx.split("\n")[1:]:
        rm = _SCHEMA_ROW.match(row[len(SCHEMA_INDENT):])
        if rm and rm["name"] in names:
            views[rm["name"]] = rm["type"]
            if rm["sem"] is not None:
                sems[rm["name"]] = rm["sem"]
    site = _prim.ByLLMCallsite(
        model=body.get("model", ""), temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"), system_prompt=system,
        response_format=body.get("response_format"), context_desc=ctx, signature=qual,
        params={n: (sig_types.get(n, ""), sems.get(n, "")) for n in names},
        param_type_str_view=views,
        return_type=(m["ret"] or "").strip() if m else "",
        sem=(m["sem"] or "").strip() if m else "",
        owner_sem=self_sem or "", tool_schema=body.get("tools"))
    return site, CallExtras(turn=_turn(body), bindings=bindings, self_view=self_view)


# --------------------------------------------------------------------------- visit by
def parse_candidate_line(line: str) -> Optional[Dict[str, str]]:
    """'<handle>) here --(<edge>)--> <node> [runs: ...]' or '<handle>) <node>'."""
    cm = _CANDIDATE.match(line)
    if not cm:
        return None
    rest = cm["rest"]
    runs = ""
    rm = _RUNS.search(rest)
    if rm:
        runs = rm["runs"]
        rest = rest[:rm.start()]
    edge, direction = "", ""
    if rest.startswith("here "):
        body = rest[5:]
        for arrow_open, arrow_close, d in (("--(", ")-->", "out"), ("<--(", ")--", "in"), ("<-(", ")->", "undir")):
            if body.startswith(arrow_open):
                end = _balanced(body, len(arrow_open) - 1)
                if end > 0 and body[end - 1:].startswith(")") and body[end - 1:end - 1 + len(arrow_close)] == arrow_close:
                    edge = body[len(arrow_open):end - 1]
                    rest = body[end - 1 + len(arrow_close):].lstrip()
                    direction = d
                    break
    return {"handle": cm["handle"], "edge": edge, "direction": direction, "node": rest, "runs": runs}


def split_visit_user(user: str) -> Dict[str, Any]:
    """Zones of a route_visit user message, whatever their order."""
    user = _strip_hint(user)
    out: Dict[str, Any] = {"intent": "", "walker": None, "here": None, "candidates": [],
                           "cand_block": "", "extra": []}
    for part in user.split("\n\n"):
        if part.startswith("Goal: "):
            out["intent"] = part[len("Goal: "):]
        elif part.startswith("Walker:\n"):
            out["walker"] = part[len("Walker:\n"):]
        elif part.startswith("Current node:\n"):
            out["here"] = part[len("Current node:\n"):]
        elif part.startswith("Candidates (choose by handle):"):
            block = part.split("\n", 1)[1] if "\n" in part else ""
            out["cand_block"] = block
            out["candidates"] = [c for c in (parse_candidate_line(l) for l in block.split("\n")) if c]
        elif part:
            out["extra"].append(part)
    return out


def _select_text(system: str) -> str:
    tail = system[len(ROUTE_SYSTEM):].strip()
    return tail[len("Choose "):].rstrip(".") if tail.startswith("Choose ") else "all"


def _parse_visit(body: Dict[str, Any]) -> Tuple[_prim.VisitByCallsite, CallExtras]:
    system = _system(body)
    zones = split_visit_user(_call_user(body))
    site = _prim.VisitByCallsite(model=body.get("model", ""), temperature=body.get("temperature"),
                           max_tokens=body.get("max_tokens"), system_prompt=system,
                           response_format=body.get("response_format"),
                           intent=zones["intent"], select=_select_text(system))
    cands = [(c["handle"], c["node"]) for c in zones["candidates"]]
    return site, CallExtras(turn=_turn(body), candidates=cands, walker=zones["walker"],
                            here=zones["here"], cand_block=zones["cand_block"])


# --------------------------------------------------------------------------- one request
def decompose(body: Dict[str, Any]) -> Tuple[_prim.Callsite, CallExtras]:
    """One request body -> (its callsite, the per-call leftovers)."""
    if _system(body).startswith(ROUTE_SYSTEM):
        site, extras = _parse_visit(body)
    else:
        site, extras = _parse_byllm(body)
    if extras.turn == 0:  # later ReAct turns re-send the same first user message
        user = _call_user(body)
        site.hint = user[len(_strip_hint(user)):]  # per-site stable schema tail
        extras.user_text = user
    return site, extras


# --------------------------------------------------------------------------- byllm repr text
# Static text helpers over what byllm renders: object reprs `Type(a=1, b='x')`, the
# JSON of a reply, and the candidate handles a visit reply names. Pure string parsing;
# the value-flow rules that consume them live in primitives.

def _lcp(a: str, b: str) -> str:
    i, n = 0, min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


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


def chosen_candidates(ob: _prim.CallObservation) -> List[Tuple[str, str]]:
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


def node_type(node_repr: str) -> str:
    return node_repr.split("(", 1)[0].strip()


# --------------------------------------------------------------------------- continuation
def _norm_msgs(msgs: list) -> list:
    """Messages reduced to what identifies a transcript position: role, text content with
    the injected "Schema requirements" hint stripped (byllm appends it to the LAST user
    message, so it hops to the feedback turn on a typed retry), and the tool calls."""
    return [(m.get("role"), _strip_hint(_content(m)), str(m.get("tool_calls") or "")) for m in msgs]


def is_continuation(prev: Dict[str, Any], cur: Dict[str, Any]) -> bool:
    """True when `cur` continues the call `prev` started: the next ReAct turn or a typed
    retry. Same model, and cur.messages extends prev.messages (the transcript grows by
    tool/feedback turns) — or equals it, byllm's retry re-sending the same body after a
    failed parse. Two genuinely separate calls with byte-identical consecutive bodies
    would fold too; that shape does not occur in byllm programs."""
    pm, cm = _norm_msgs(prev.get("messages") or []), _norm_msgs(cur.get("messages") or [])
    if len(cm) < len(pm) or not pm or prev.get("model") != cur.get("model"):
        return False
    # The system message may change between turns: byllm's forced final pass (budget
    # exhausted, tool_choice="none") drops the tool block from it. Compare the rest.
    start = 1 if pm[0][0] == "system" and cm[0][0] == "system" else 0
    return cm[start:len(pm)] == pm[start:]

