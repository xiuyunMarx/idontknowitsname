"""Engine-free helpers for the static pass: UniIR extraction utilities and the
prompt-text constants. The server owns prompt construction end to end (the
client wires only argument reprs), so these constants ARE the canonical bytes —
the only consistency that matters is warm path == serve path, both rendered by
ByLLMCallsite from this module's pieces."""

import re
from typing import Any, Dict, List, Tuple

import jaclang  # registers the .jac meta importer — must come first
import jaclang.jac0core.unitree as uni  #type: ignore

SYSTEM_PERSONA = "This is a task you must complete by returning only the output. The task will be expressed in the form of function call with arguments. Do not include explanations, code, or extra text—only the result."
TOOL_INSTRUCTION = " Use the tools provided to reach the goal. Call one tool at a time with proper args—no explanations, no narration. Think step by step, invoking tools as needed. When done, always call finish_tool(output) to return the final output. Only use tools."
TOOL_PROTOCOL_HEADER = "# Calling tools"

JSON_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "tuple": "array", "set": "array", "dict": "object"}
FIELD_EXPR_RE = re.compile(r"^(visitor|here|self)\.([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\])?$")
LOOP_STMTS = (uni.WhileStmt, uni.IterForStmt, uni.InForStmt)

# --- visit routing (`visit <edges> by llm(...)`) -----------------------------
# byllm builds the routing prompt in jaclang/byllm/visit_routing.jac::route_visit.
# These mirror it byte for byte: the warm prefix the server prefills must be a
# real prefix of what the client later sends on the `generate` path.
ROUTE_SYSTEM = ("You are routing a graph walker. Choose which candidate node(s) the "
                "walker should visit next, by handle. Return only valid handles.")
# Zone label -> the header route_visit emits above the zone's runtime body.
ROUTE_ZONE_LABEL = {
    "walker": "Walker:",
    "here": "Current node:",
    "candidates": "Candidates (choose by handle):",
}
# route_visit orders the runtime zones by JAC_ROUTE_CACHE_LAYOUT; `Goal: <intent>`
# precedes both layouts, which is why it alone can be warmed at compile time.
ROUTE_LAYOUT_DEFAULT = ("walker", "here", "candidates")
ROUTE_LAYOUT_CACHE = ("here", "candidates", "walker")


def norm(text: str) -> str:
    """unparse() is space-padded (`self . request`); collapse before any matching."""
    return re.sub(r"\s+", "", text or "")


def clean_type(type_text: str) -> str:
    """Normalize a type annotation's unparse (': list [ dict [ str , str ] ]') to 'list[dict[str, str]]'."""
    t = type_text.strip()
    if t.startswith(":"):
        t = t[1:].strip()
    t = re.sub(r"\s*\[\s*", "[", t)
    t = re.sub(r"\s*\]", "]", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    return t


def json_type_of(py_type: str) -> str:
    base = py_type.split("[", 1)[0].strip()
    return JSON_TYPE.get(base, base or "any")


def literal_of(expr: uni.UniNode) -> Tuple[bool, Any]:
    """Best-effort constant-fold of an expression node. Returns (is_literal, value)."""
    if isinstance(expr, (uni.Int, uni.Float, uni.String, uni.MultiString)):
        return True, expr.lit_value
    if isinstance(expr, uni.Bool):
        return True, expr.lit_value
    if isinstance(expr, (uni.ListVal, uni.TupleVal)):
        vals = []
        for item in expr.values or []:
            ok, v = literal_of(item)
            if not ok:
                return False, None
            vals.append(v)
        return True, tuple(vals) if isinstance(expr, uni.TupleVal) else vals
    if isinstance(expr, uni.DictVal):
        out = {}
        for kv in expr.kv_pairs or []:
            ok_k, k = literal_of(kv.key) if kv.key is not None else (False, None)
            ok_v, v = literal_of(kv.value)
            if not (ok_k and ok_v):
                return False, None
            out[k] = v
        return True, out
    return False, None


def params_of(sig: uni.FuncSignature) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sig.get_parameters():
        out.append({
            "name": p.name.value,
            "type": clean_type(p.type_tag.unparse()) if p.type_tag else "",
            "sem": (p.semstr or "").strip(),
            "required": p.value is None,
        })
    return out


def extract_llm_call(call: uni.UniNode) -> Tuple[List[str], Dict[str, Any], str]:
    """Pull tool names, literal kwargs, and a literal system_prompt out of the `by llm(...)` expression."""
    tool_names: List[str] = []
    call_params: Dict[str, Any] = {}
    extra_sys = ""
    if not isinstance(call, uni.FuncCall):
        return tool_names, call_params, extra_sys
    for kw in call.params or []:
        if not isinstance(kw, uni.KWPair) or kw.key is None:
            continue
        key = kw.key.unparse().strip()
        if key == "tools":
            if isinstance(kw.value, uni.ListVal):
                for item in kw.value.values or []:
                    tool_names.append(re.sub(r"\s*\.\s*", ".", item.unparse().strip()))
            else:
                print(f"[jac-static] warning: non-literal tools list ({kw.value.unparse()}); tool schemas unavailable statically")
        elif key == "system_prompt":
            ok, v = literal_of(kw.value)
            if ok:
                extra_sys = str(v)
            else:
                print(f"[jac-static] warning: dynamic system_prompt ({kw.value.unparse()}); invariant excludes it")
        else:
            ok, v = literal_of(kw.value)
            if ok:
                call_params[key] = v
    return tool_names, call_params, extra_sys


def finish_tool_schema(return_type: str) -> Dict[str, Any]:
    # Primitive returns only; obj returns need the deferred type-materialization pass.
    prop = {"type": json_type_of(return_type), "description": "The final output of the tool calls."}
    return {"type": "function", "function": {"name": "finish_tool", "description": "This tool is used to finish the tool calls and return the final output.", "parameters": {"type": "object", "properties": {"final_output": prop}, "required": ["final_output"], "additionalProperties": False}}}


def render_tool(tool: Dict[str, Any]) -> str:
    """One tool block, matching byllm tool_protocol._render_tool byte for byte."""
    fn = tool.get("function", {}) or {}
    name = str(fn.get("name", "") or "")
    if not name:
        return ""
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = params.get("required", []) or []
    lines = [f"- {name}: {fn.get('description') or name}"]
    for pname, pinfo in props.items():
        flag = " (required)" if pname in required else ""
        lines.append(f"    - {pname} [{pinfo.get('type', 'any')}]{flag}: {pinfo.get('description', '')}")
    return "\n".join(lines)


def schema_entry(name: str, type_str: str, sem: str) -> str:
    """Static schema-zone row, following byllm _schema_entry's rule: a param nobody
    described earns no entry (its type already shows in the signature line).
    Field/member expansion for obj/enum params needs the deferred
    type-materialization pass, so statically the row is the described line only."""
    if not sem:
        return ""
    return f"{name}: {type_str or 'any'} ---- {sem}"


def render_bindings(params: List[Dict[str, Any]], args: Dict[str, str]) -> List[str]:
    """Bindings zone lines in declaration order. Wire args are already reprs (the
    Jac client sends str_views), so values go in verbatim — never re-repr'd."""
    return [f"{p['name']} = {args[p['name']]}" for p in params if p["name"] in args]


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """The text tool-protocol block; the server's own ReAct parser consumes replies in this shape."""
    rendered = [b for b in (render_tool(t) for t in tools) if b]
    if not rendered:
        return ""
    return TOOL_PROTOCOL_HEADER + "\n\nYou can call tools. The tools available to you are:\n\n" + "\n".join(rendered) + "\n\nTo call a tool, reply with ONLY a tool call in exactly this form and nothing else:\n\n<tool_call>{\"name\": \"<tool_name>\", \"arguments\": {<args>}}</tool_call>" + "\n\nRules:\n- Call exactly one tool per reply.\n- \"arguments\" must be a JSON object whose keys match the tool's parameters.\n- Emit nothing outside the <tool_call> tags when calling a tool.\n- After a tool result is returned, call the next tool, or call finish_tool with your final answer to end."
