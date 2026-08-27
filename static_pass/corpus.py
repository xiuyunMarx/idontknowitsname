SYSTEM_PERSONA = "This is a task you must complete by returning only the output. The task will be expressed in the form of function call with arguments. Do not include explanations, code, or extra text—only the result."
TOOL_INSTRUCTION = " Use the tools provided to reach the goal. Call one tool at a time with proper args—no explanations, no narration. Think step by step, invoking tools as needed. When done, always call finish_tool(output) to return the final output. Only use tools."
TOOL_PROTOCOL_HEADER = "# Calling tools"

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


def route_system_prompt(select) -> str:
    """byllm's routing system text for a given `select=`; mirrors route_visit."""
    if isinstance(select, int) and not isinstance(select, bool) and select < 1:
        select = "all"  # route_visit normalizes a nonsense count away before use
    text = ROUTE_SYSTEM
    if select == 1:
        text += " Choose exactly one."
    elif isinstance(select, int) and not isinstance(select, bool):
        text += f" Choose exactly {select}."
    elif isinstance(select, tuple) and len(select) == 2:
        text += f" Choose between {select[0]} and {select[1]} (inclusive)."
    return text


def route_layout() -> tuple:
    """Runtime zone order route_visit will use. JAC_ROUTE_CACHE_LAYOUT is read in
    the *client* process; the server only mirrors it for diagnostics, since the
    warmable prefix (system + `Goal:`) precedes the zones under either layout."""
    import os
    return ROUTE_LAYOUT_CACHE if os.environ.get("JAC_ROUTE_CACHE_LAYOUT") == "1" else ROUTE_LAYOUT_DEFAULT


# --- tool protocol (byllm tool_protocol.jac) ----------------------------------

def render_tool(tool: dict) -> str:
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


def format_tools_for_prompt(tools: list) -> str:
    """The text tool-protocol block; the server's own ReAct parser consumes replies in this shape."""
    rendered = [b for b in (render_tool(t) for t in tools) if b]
    if not rendered:
        return ""
    return TOOL_PROTOCOL_HEADER + "\n\nYou can call tools. The tools available to you are:\n\n" + "\n".join(rendered) + "\n\nTo call a tool, reply with ONLY a tool call in exactly this form and nothing else:\n\n<tool_call>{\"name\": \"<tool_name>\", \"arguments\": {<args>}}</tool_call>" + "\n\nRules:\n- Call exactly one tool per reply.\n- \"arguments\" must be a JSON object whose keys match the tool's parameters.\n- Emit nothing outside the <tool_call> tags when calling a tool.\n- After a tool result is returned, call the next tool, or call finish_tool with your final answer to end."
