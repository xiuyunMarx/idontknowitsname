"""Turn raw /v1/chat/completions requests into callsites, calls, sessions and a Program.

Input: a trace, one request per line (JSONL):
    {"session": "triage:41321", "t_arrive": 1.23, "t_done": 1.98,
     "request": {"model": ..., "messages": [...], "tools": ..., "response_format": ...},
     "response": "<assistant text>" | {"role": "assistant", ...}}

The text formats parsed here are byllm's (jaclang/byllm/impl/mtir.impl.jac::MTRuntime.factory,
jaclang/byllm/visit_routing.jac::route_visit, jaclang/byllm/schema.jac::inject_schema_hint,
jaclang/byllm/tool_protocol.jac). See memo.txt for examples of every shape.

Pipeline:
    load_trace(path)              -> [TraceRecord]
    parse_request(record)         -> (LLMCallsite, Call)      one request
    build_sessions(records)       -> {session id: Session}    ReAct turns folded, order kept
    build_programs(records)       -> {program name: Program}  statistics over sessions
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .primitives import (ByLLMCallsite, Call, CallInstance, CallsiteType, LLMCallsite, Program,
                         Session, TraceRecord, VisitByCallsite)

# --------------------------------------------------------------------------- byllm constants
ROUTE_SYSTEM = ("You are routing a graph walker. Choose which candidate node(s) the walker should "
                "visit next, by handle. Return only valid handles.")
HINT_HEADER = "Schema requirements:"
TOOL_RESPONSE_TAG = "<tool_response>"
TOOL_BLOCK_HEADER = "\n# Calling tools\n"   # tool_protocol.jac: tools rendered into the system prompt
SCHEMA_INDENT = "      "  # 6 spaces before every schema row

# "Owner.func(a: T, b: U) -> R --- sem"; the sem may itself contain " -> " so ret is lazy
_HEADER = re.compile(r"^(?P<qual>[\w.]+)\((?P<sig>.*?)\)(?: -> (?P<ret>.*?))?(?: --- (?P<sem>.*))?$")
_SCHEMA_ROW = re.compile(r"^(?P<name>\w+): (?P<type>.*?)(?: ---- (?P<sem>.*))?$")
_BINDING = re.compile(r"^(?P<name>[A-Za-z_]\w*) = (?P<value>.*)$")
_SELF = re.compile(r"^self = (?P<view>.*?)(?: ---- (?P<sem>.*))?$")
_CANDIDATE = re.compile(r"^(?P<handle>\w+)\) (?P<rest>.*)$")
_RUNS = re.compile(r" \[runs: (?P<runs>.*)\]$")
_TEXT_TOOL = re.compile(r"^- (\w+): ", re.M)


# --------------------------------------------------------------------------- input
def load_trace(path: str) -> List[TraceRecord]:
    out: List[TraceRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(TraceRecord(session=str(d.get("session") or d.get("user") or "default"),
                                   t_arrive=float(d["t_arrive"]), t_done=float(d["t_done"]),
                                   request=d["request"], response=d.get("response"),
                                   engine_s=d.get("engine_s")))
    return out


# --------------------------------------------------------------------------- small helpers
def _sha(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


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


# --------------------------------------------------------------------------- classification
def classify(req: Dict[str, Any]) -> CallsiteType:
    if _system(req).startswith(ROUTE_SYSTEM):
        return CallsiteType.VISITBY
    return CallsiteType.TOOL if req.get("tools") else CallsiteType.BYLLM


# --------------------------------------------------------------------------- by llm()
def split_byllm_user(user: str) -> Tuple[str, List[str], Dict[str, str], Optional[str]]:
    """Split a byllm user message into (context_desc, param names, bindings, self view).

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
    last: Optional[str] = None
    in_self = False
    while i < len(lines):
        line = lines[i]
        sm = _SELF.match(line)
        if sm and not in_self:
            self_view = sm["view"]
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
    return "\n".join(ctx), names, bindings, self_view


def parse_byllm(rec: TraceRecord) -> Tuple[ByLLMCallsite, Call]:
    req = rec.request
    system = _system(req)
    user = _call_user(req)
    ctx, names, bindings, self_view = split_byllm_user(user)
    header = ctx.split("\n", 1)[0]
    m = _HEADER.match(header)
    qual = m["qual"] if m else header[:40]
    ret = (m["ret"] or "").strip() if m else ""
    sem = (m["sem"] or "").strip() if m else ""
    sig_types: Dict[str, str] = {}
    if m:
        for part in m["sig"].split(","):
            if ":" in part:
                n, t = part.split(":", 1)
                sig_types[n.strip()] = t.strip()
    sems: Dict[str, str] = {}
    for row in ctx.split("\n")[1:]:
        rm = _SCHEMA_ROW.match(row[len(SCHEMA_INDENT):])
        if rm and rm["name"] in names and rm["sem"] is not None:
            sems[rm["name"]] = rm["sem"]
    params = [(n, sig_types.get(n, ""), sems.get(n, "")) for n in names]
    tools = sorted(t["function"]["name"] for t in (req.get("tools") or []) if t.get("function", {}).get("name"))
    if not tools and TOOL_BLOCK_HEADER in system:  # text tool protocol: tools live in the system prompt
        tools = sorted(_TEXT_TOOL.findall(system.split(TOOL_BLOCK_HEADER, 1)[1]))
    # Identity (site.key) is derived from qualname + typed params + model, so it is stable
    # across the tool block hopping in and out of the system prompt (byllm's forced final
    # pass), typed retries, and sem-only edits.
    site = ByLLMCallsite(kind=CallsiteType.TOOL if tools else CallsiteType.BYLLM, label=qual,
                         model=req.get("model", ""), system_prompt=system,
                         response_format=req.get("response_format"), temperature=req.get("temperature"),
                         max_tokens=req.get("max_tokens"), context_desc=ctx, qualname=qual, params=params,
                         return_type=ret, sem=sem, tools=tools)
    call = Call(key=site.key, session=rec.session, t_arrive=rec.t_arrive, t_done=rec.t_done,
                model=site.model, turn=_turn(req), bindings=bindings, self_view=self_view,
                response=rec.response, raw=req, engine_s=rec.engine_s)
    return site, call


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


def parse_visit(rec: TraceRecord) -> Tuple[VisitByCallsite, Call]:
    req = rec.request
    system = _system(req)
    zones = split_visit_user(_call_user(req))
    handles = [c["handle"] for c in zones["candidates"]]
    edges = sorted({c["edge"] for c in zones["candidates"] if c["edge"]})
    intent = zones["intent"]
    label = "visit: " + (intent[:50] + ("..." if len(intent) > 50 else "") if intent else "(no intent)")
    site = VisitByCallsite(kind=CallsiteType.VISITBY, label=label, model=req.get("model", ""),
                           system_prompt=system, response_format=req.get("response_format"),
                           temperature=req.get("temperature"), max_tokens=req.get("max_tokens"),
                           intent=intent, select=_select_text(system), candidates=handles, edges=edges)
    call = Call(key=site.key, session=rec.session, t_arrive=rec.t_arrive, t_done=rec.t_done, model=site.model,
                walker=zones["walker"], here=zones["here"], candidates=handles, response=rec.response, raw=req,
                engine_s=rec.engine_s)
    return site, call


# --------------------------------------------------------------------------- one request
def parse_request(rec: TraceRecord) -> Tuple[LLMCallsite, Call]:
    kind = classify(rec.request)
    site, call = parse_visit(rec) if kind is CallsiteType.VISITBY else parse_byllm(rec)
    if call.turn == 0:  # later ReAct turns re-send the same first user message
        site.stable_prefix, site.prefix_n = _call_user(rec.request), 1
    return site, call


# --------------------------------------------------------------------------- sessions
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


class SessionBuilder:
    """Turns one session's parsed requests, in arrival order, into CallInstances: a request
    that continues the previous one's transcript (next ReAct turn, typed retry) is folded
    into the same instance. Used offline (build_sessions) and online (serve.session)."""

    def __init__(self, session_id: str, program: str) -> None:
        self.session = Session(id=session_id, program=program)
        self._last_call: Optional[Call] = None

    def step(self, call: Call) -> Tuple[CallInstance, bool]:
        """Add one parsed request; returns (its instance, whether that instance is new)."""
        inst = self.session.calls[-1] if self.session.calls else None
        if (inst is not None and self._last_call is not None and call.key == inst.key
                and is_continuation(self._last_call.raw or {}, call.raw or {})):
            inst.turns.append((call.t_arrive, call.t_done))
            inst.t_done = call.t_done
            inst.result = call.response
            inst.engine_s += call.engine_s or 0.0
            new = False
        else:
            inst = CallInstance(key=call.key, session=self.session.id, model=call.model,
                                t_arrive=call.t_arrive, t_done=call.t_done,
                                turns=[(call.t_arrive, call.t_done)], bindings=call.bindings,
                                self_view=call.self_view, result=call.response,
                                engine_s=call.engine_s or 0.0)
            self.session.calls.append(inst)
            new = True
        self._last_call = call
        return inst, new


def build_sessions(records: Iterable[TraceRecord]) -> Dict[str, Session]:
    """Order every session's requests by arrival and fold ReAct turns into one CallInstance."""
    return {sid: builder.session for sid, (builder, _) in _fold(records).items()}


def _fold(records: Iterable[TraceRecord]) -> Dict[str, Tuple[SessionBuilder, list]]:
    """session id -> (its folded builder, the parsed callsites in arrival order)."""
    by_session: Dict[str, List[TraceRecord]] = defaultdict(list)
    for r in records:
        by_session[r.session].append(r)
    out: Dict[str, Tuple[SessionBuilder, list]] = {}
    for sid, recs in by_session.items():
        recs.sort(key=lambda r: r.t_arrive)
        builder, sites = SessionBuilder(sid, ""), []
        for rec in recs:
            site, call = parse_request(rec)
            sites.append(site)
            builder.step(call)
        out[sid] = (builder, sites)
    return out


def build_programs(records: Iterable[TraceRecord]) -> Dict[str, Program]:
    """Programs discovered from content: a session belongs to the program whose entry
    callsite (its first request) it shares. No program name crosses the wire; a program's
    id is its entry callsite's key, its display name that callsite's label."""
    programs: Dict[str, Program] = {}
    folded = sorted(_fold(records).values(),
                    key=lambda bs: bs[0].session.calls[0].t_arrive if bs[0].session.calls else 0.0)
    for builder, sites in folded:
        if not sites:
            continue
        entry = sites[0]
        prog = programs.get(entry.key)
        if prog is None:
            prog = programs[entry.key] = Program(entry.label)
        for site in sites:
            prog.add_callsite(site)
        builder.session.program = prog.name
        prog.observe(builder.session)
    return programs


# --------------------------------------------------------------------------- cli
def main(argv: List[str]) -> int:
    if not argv:
        print("usage: python -m trace_extractor.parser TRACE.jsonl [--save OUT.json]", file=sys.stderr)
        return 2
    save = argv[argv.index("--save") + 1] if "--save" in argv else None
    programs = build_programs(load_trace(argv[0]))
    for p in programs.values():
        print(p.describe())
        for sess in p.sessions.values():
            print(f"    session {sess.id}: " + " -> ".join(p.label(k) for k in sess.sequence))
    if save:
        with open(save, "w") as f:
            json.dump({name: p.to_dict() for name, p in programs.items()}, f, indent=1)
        print(f"saved {save}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
