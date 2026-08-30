"""Request bodies in the exact shapes stock byllm puts on the wire (see jaclang/byllm), used by
the offline tests: plain by-llm, structured return, ReAct (native and text tool protocol),
visit routing, typed retry."""
import json

PERSONA = ("This is a task you must complete by returning only the output. The task will be expressed in the "
           "form of function call with arguments. Do not include explanations, code, or extra text—only the result.")
TOOL_INSTR = (" Use the tools provided to reach the goal. Call one tool at a time with proper args—no explanations, "
              "no narration. Think step by step, invoking tools as needed. When done, always call finish_tool(output) "
              "to return the final output. Only use tools.")
TOOL_BLOCK = ("\n\n# Calling tools\n\nYou can call tools. The tools available to you are:\n\n"
              "- web_search: Search\n    - query [string] (required): q\n- open_page: Read\n- finish_tool: finish\n\n"
              "Rules:\n- Call exactly one tool per reply.")
ROUTE = ("You are routing a graph walker. Choose which candidate node(s) the walker should visit next, by handle. "
         "Return only valid handles.")
SUMMARIZE_MODEL, ROUTER_MODEL, RESEARCH_MODEL, DECOMPOSE_MODEL = (
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-4B")


def visit_req(request: str):
    user = (
        "Goal: Send the customer's request to the one specialist desk whose duty covers it.\n\n"
        f"Walker:\nthe triage walker carrying one customer request through the desks(request (the customer's request in their own words)={request!r}, route (the desk the request was routed to)='', answer (the desk's resolution)='', summary='')\n\n"
        "Current node:\nthe front desk that owns triage and nothing else(queue (the intake queue this desk drains)='front desk')\n\n"
        "Candidates (choose by handle):\n"
        "BillingAgent) here --(the class of request this desk is on the hook for(duty (what the desk at the far end of this link handles)='charges, invoices and disputed amounts'))--> the desk for charges, invoices and disputed amounts(desk (the desk's name)='billing', playbook (the procedure this desk follows before it answers)='check (the) ledger') [runs: handle]\n"
        "ShippingAgent) here --(the class of request this desk is on the hook for(duty (what the desk at the far end of this link handles)='delivery and tracking'))--> the desk for delivery and tracking(desk (the desk's name)='shipping', playbook (the procedure this desk follows before it answers)='...') [runs: handle]\n"
        "Orphan) a node with no edge\n\n"
        "Schema requirements:\n- schema_object_wrapper (array)")
    return {"model": ROUTER_MODEL, "temperature": 0.0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "list"}},
            "messages": [{"role": "system", "content": ROUTE + " Choose exactly one."}, {"role": "user", "content": user}]}


def resolve_req(desk: str, model: str, request: str):
    cls = desk.capitalize() + "Agent"
    user = (f"{cls}.resolve_{desk}(request: str, playbook: str) -> str\n"
            f"request = {request!r}\nplaybook = 'check the ledger'\n\n"
            f"self = {cls}(desk={desk!r}, playbook='check the ledger') ---- the desk for {desk}")
    return {"model": model, "temperature": 0.0, "max_tokens": 220,
            "messages": [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]}


def summarize_req(request: str, desk: str, answer: str):
    user = ("Agent.summarize(request: str, desk: str, answer: str) -> str --- Compress the desk's resolution into a two-sentence note for the ticket log.\n"
            "      desk: str ---- the desk that handled the request\n"
            "      answer: str ---- the resolution the desk produced\n"
            f"request = {request!r}\ndesk = {desk!r}\nanswer = {answer!r}\n\n"
            f"self = Agent(request={request!r}, route={desk!r}, answer={answer!r}, summary='') ---- the triage walker carrying one customer request through the desks")
    return {"model": SUMMARIZE_MODEL, "temperature": 0.0, "max_tokens": 140,
            "messages": [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]}


TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
         for n in ("web_search", "open_page", "finish_tool")]


def research_turns(task: str, native: bool = True):
    """The requests of one ReAct call: turn 1, then each turn re-sends the grown transcript."""
    user = ("WebSearchAgent.web_research(task: str, plan: list[str]) -> str --- Answer the plan's background questions from the web index.\n"
            "      task: str ---- the research task being investigated\n"
            "      plan: list[str] ---- this round's open questions\n"
            f"task = {task!r}\nplan = ['q1', 'q2']")
    if native:
        base = [{"role": "system", "content": PERSONA + TOOL_INSTR}, {"role": "user", "content": user}]
        a1 = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"x\"}"}}]}
        t1 = {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "1. A -- ..."}
        a2 = {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "open_page", "arguments": "{\"title\": \"A\"}"}}]}
        t2 = {"role": "tool", "tool_call_id": "c2", "name": "open_page", "content": "page text"}
        msgs, tools = [base, base + [a1, t1], base + [a1, t1, a2, t2]], TOOLS
    else:
        base = [{"role": "system", "content": PERSONA + TOOL_INSTR + TOOL_BLOCK}, {"role": "user", "content": user}]
        a1 = {"role": "assistant", "content": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'}
        t1 = {"role": "user", "content": "<tool_response>web_search -> 1. A -- ...</tool_response>"}
        a2 = {"role": "assistant", "content": '<tool_call>{"name": "open_page", "arguments": {"title": "A"}}</tool_call>'}
        t2 = {"role": "user", "content": "<tool_response>open_page -> page text</tool_response>"}
        # budget exhausted: byllm's forced final pass drops the tool block and sets tool_choice="none"
        final = [{"role": "system", "content": PERSONA + TOOL_INSTR}, {"role": "user", "content": user}, a1, t1, a2, t2]
        msgs, tools = [base, base + [a1, t1], final], None
    out = []
    for i, m in enumerate(msgs):
        req = {"model": RESEARCH_MODEL, "temperature": 0.0, "max_tokens": 320, "messages": m, "stop": ["</tool_call>"]}
        if tools:
            req["tools"] = tools
        if not native and i == len(msgs) - 1:
            req["tool_choice"] = "none"
            del req["stop"]
        out.append(req)
    return out


def decompose_req(task: str, web: str):
    user = ("ResearchSupervisor.decompose_task(task: str, web: str) -> list[str] --- Name the open questions.\n"
            "      task: str ---- the research task being investigated\n"
            f"task = {task!r}\nweb = {web!r}\n\nSchema requirements:\n- schema_object_wrapper (array)")
    return {"model": DECOMPOSE_MODEL, "temperature": 0.0, "max_tokens": 512,
            "response_format": {"type": "json_schema", "json_schema": {"name": "list"}},
            "messages": [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]}


def typed_retry(req, bad_reply: str, feedback: str):
    """byllm's typed retry: the same call re-sent with the failed reply and a feedback user turn."""
    return {**req, "messages": req["messages"] + [{"role": "assistant", "content": bad_reply},
                                                  {"role": "user", "content": feedback}]}


def trace_rows():
    """A small trace: two interleaved triage sessions and one deep_research session with a loop
    (decompose -> research(3 native turns) -> decompose -> research(2 text-protocol turns))
    and a typed retry on the first decompose."""
    rows = []

    def add(session, t, dur, req, resp):
        rows.append({"session": session, "t_arrive": t, "t_done": t + dur, "request": req, "response": resp})

    r1, r2 = "The tracking has not moved in nine days.", "I was charged twice for one order."
    add("triage:1", 0.0, 0.2, visit_req(r1), '{"schema_object_wrapper": ["ShippingAgent"]}')
    add("triage:2", 0.1, 0.2, visit_req(r2), '{"schema_object_wrapper": ["BillingAgent"]}')
    add("triage:1", 1.0, 0.5, resolve_req("shipping", SUMMARIZE_MODEL, r1), "reshipped")
    add("triage:2", 1.2, 0.8, resolve_req("billing", RESEARCH_MODEL, r2), "refunded")
    add("triage:1", 2.0, 0.3, summarize_req(r1, "shipping", "reshipped"), "note1")
    add("triage:2", 2.5, 0.3, summarize_req(r2, "billing", "refunded"), "note2")
    t = 10.0
    d1 = decompose_req("why cold starts", "")
    add("deep_research:7", t, 1.0, d1, "not json")
    add("deep_research:7", t + 1.2, 0.8, typed_retry(d1, "not json", "Output must be valid JSON."), '{"schema_object_wrapper": ["q1", "q2"]}')
    for i, req in enumerate(research_turns("why cold starts")):
        add("deep_research:7", t + 3 + i * 1.0, 0.4, req, "tool call" if i < 2 else "final")
    add("deep_research:7", t + 7.0, 1.0, decompose_req("why cold starts", "found A"), '{"schema_object_wrapper": ["q3"]}')
    for i, req in enumerate(research_turns("why cold starts", native=False)):
        add("deep_research:7", t + 9 + i * 1.0, 0.4, req, "x")
    return rows


def write_jsonl(path: str, rows) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
