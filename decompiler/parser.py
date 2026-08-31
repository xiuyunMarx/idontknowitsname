"""Decompile one /v1/chat/completions request body into the callsite that produced it.

The text formats parsed here are byllm's (jaclang/byllm/impl/mtir.impl.jac::MTRuntime.factory,
jaclang/byllm/visit_routing.jac::route_visit, jaclang/byllm/schema.jac::inject_schema_hint,
jaclang/byllm/tool_protocol.jac).

    site, extras = decompose(body)        # body: parsed request JSON
    is_continuation(prev_body, body)      # next ReAct turn / typed retry of the same call
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .primitives import ByLLMCallsite, Callsite, VisitByCallsite

ROUTE_SYSTEM = ("You are routing a graph walker. Choose which candidate node(s) the walker should "
                "visit next, by handle. Return only valid handles.")
HINT_HEADER = "Schema requirements:"
TOOL_RESPONSE_TAG = "<tool_response>"
SCHEMA_INDENT = "      "  # 6 spaces before every schema row

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
    candidates: List[Tuple[str, str]] = field(default_factory=list)  # (handle, node type) offered


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
    header = lines[0]
    m = _HEADER.match(header)
    names: List[str] = []
    if m:
        for part in m["sig"].split(","):
            part = part.strip()
            if part:
                names.append(part.split(":")[0].split("=")[0].strip())
    ctx = [header]
    i = 1
    while i < len(lines) and lines[i].startswith(SCHEMA_INDENT):
        ctx.append(lines[i])
        i += 1
    bindings: Dict[str, str] = {}
    self_view: Optional[str] = None
    self_sem: Optional[str] = None
    last: Optional[str] = None
    in_self = False
    while i < len(lines):
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
            if bm and (bm["name"] in names or not names or last is not None or not bindings):
                bindings[bm["name"]] = bm["value"]
                last = bm["name"]
            elif last is not None and line != "":
                bindings[last] += "\n" + line
        i += 1
    return "\n".join(ctx), names, bindings, self_view, self_sem


def _parse_byllm(body: Dict[str, Any]) -> Tuple[ByLLMCallsite, CallExtras]:
    system = _system(body)
    ctx, names, _bindings, _self_view, self_sem = split_byllm_user(_call_user(body))
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
    site = ByLLMCallsite(
        model=body.get("model", ""), temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"), system_prompt=system,
        response_format=body.get("response_format"), context_desc=ctx, signature=qual,
        params={n: (sig_types.get(n, ""), sems.get(n, "")) for n in names},
        param_type_str_view=views,
        return_type=(m["ret"] or "").strip() if m else "",
        sem=(m["sem"] or "").strip() if m else "",
        owner_sem=self_sem or "", tool_schema=body.get("tools"))
    return site, CallExtras(turn=_turn(body))


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
    out: Dict[str, Any] = {"intent": "", "walker": None, "here": None, "candidates": [], "extra": []}
    for part in user.split("\n\n"):
        if part.startswith("Goal: "):
            out["intent"] = part[len("Goal: "):]
        elif part.startswith("Walker:\n"):
            out["walker"] = part[len("Walker:\n"):]
        elif part.startswith("Current node:\n"):
            out["here"] = part[len("Current node:\n"):]
        elif part.startswith("Candidates (choose by handle):"):
            block = part.split("\n", 1)[1] if "\n" in part else ""
            out["candidates"] = [c for c in (parse_candidate_line(l) for l in block.split("\n")) if c]
        elif part:
            out["extra"].append(part)
    return out


def _select_text(system: str) -> str:
    tail = system[len(ROUTE_SYSTEM):].strip()
    return tail[len("Choose "):].rstrip(".") if tail.startswith("Choose ") else "all"


def _parse_visit(body: Dict[str, Any]) -> Tuple[VisitByCallsite, CallExtras]:
    system = _system(body)
    zones = split_visit_user(_call_user(body))
    site = VisitByCallsite(model=body.get("model", ""), temperature=body.get("temperature"),
                           max_tokens=body.get("max_tokens"), system_prompt=system,
                           response_format=body.get("response_format"),
                           intent=zones["intent"], select=_select_text(system))
    cands = [(c["handle"], c["node"].split("(", 1)[0].strip()) for c in zones["candidates"]]
    return site, CallExtras(turn=_turn(body), candidates=cands)


# --------------------------------------------------------------------------- one request
def decompose(body: Dict[str, Any]) -> Tuple[Callsite, CallExtras]:
    """One request body -> (its callsite, the per-call leftovers)."""
    if _system(body).startswith(ROUTE_SYSTEM):
        site, extras = _parse_visit(body)
    else:
        site, extras = _parse_byllm(body)
    if extras.turn == 0:  # later ReAct turns re-send the same first user message
        site.stable_prefix, site.prefix_n = _call_user(body), 1
    return site, extras


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
