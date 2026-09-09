"""Mock chat-completions server that records every request body.

Serves byllm (InterceptorLLM) and DSPy (via litellm/openai) traffic. Replies are
built from the request itself so the frameworks accept them:
  * byllm: an instance of the request's response_format JSON schema; every N-th
    call of a chosen callsite answers garbage once so byllm's typed retry (the
    hopping schema hint) is on the log too.
  * DSPy ChatAdapter: the output fields named in the user message's reminder
    ("Respond with the corresponding output fields, starting with ...").

Usage: python recorder.py <port> <log.jsonl> [<wiki_cache_dir>]
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
LOG = sys.argv[2]
CACHE = sys.argv[3] if len(sys.argv) > 3 else ""
LOCK = threading.Lock()
COUNT = {"n": 0, "bad": 0}


def _content(m):
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


# ----------------------------------------------------------------- byllm: instance of the schema
def _resolve(schema, root):
    if "$ref" in schema:
        path = schema["$ref"].lstrip("#/").split("/")
        node = root
        for p in path:
            node = node[p]
        return _resolve(node, root)
    return schema


def _instance(schema, root, name="", depth=0):
    schema = _resolve(schema, root)
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return _instance(schema["anyOf"][0], root, name, depth)
    if "allOf" in schema:
        return _instance(schema["allOf"][0], root, name, depth)
    t = schema.get("type")
    if isinstance(t, list):
        t = [x for x in t if x != "null"][0] if any(x != "null" for x in t) else "null"
    if t == "object":
        return {k: _instance(v, root, k, depth + 1) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        items = schema.get("items", {"type": "string"})
        n = 2 if depth < 2 else 1
        return [_instance(items, root, name, depth + 1) for _ in range(n)]
    if t == "integer":
        return 1
    if t == "number":
        return 0.5
    if t == "boolean":
        return True
    if t == "null":
        return None
    return _text(name, COUNT["n"])


def _text(name, n):
    if name in ("questions", "question"):
        return f"What is fact {n} of the claim?"
    if name == "source":
        return f"Wiki page {n} https://en.wikipedia.org/wiki/Page_{n}"
    if name == "excerpt":
        return f"excerpt {n}"
    if name == "answer":
        return f"answer {n}"
    if name == "rationale":
        return f"rationale {n}: the evidence covers the claim"
    return f"{name or 'text'} {n}"


def _byllm_reply(body):
    rf = body.get("response_format")
    n = COUNT["n"]
    if not rf:
        # a plain string return (coding agent's implement): a wrong function body
        return f"def solution(*args, **kwargs):\n    return None  # attempt {n}\n"
    schema = rf.get("json_schema", {}).get("schema", rf.get("schema", rf))
    inst = _instance(schema, schema)
    # the claim's questions decide the scouts' wiki queries: pre-fill the cache so
    # the program never hits the network
    if CACHE and isinstance(inst, dict):
        for v in inst.values():
            if isinstance(v, list) and v and isinstance(v[0], str):
                for q in v:
                    p = os.path.join(CACHE, hashlib.md5(q.encode()).hexdigest() + ".txt")
                    if not os.path.exists(p):
                        with open(p, "w") as f:
                            f.write(f"Title: Page about {q}\nURL: https://en.wikipedia.org/wiki/Page\n"
                                    f"Excerpt: A passage answering {q}")
    return json.dumps(inst)


# ----------------------------------------------------------------- DSPy: fields from the reminder
_FIELD = re.compile(r"`\[\[ ## (\w+) ## \]\]`(?: \((.*?)\))?")
_TOOLS = re.compile(r"\(\d+\) (\w+), whose description is .*?It takes arguments (\{.*?\})\.", re.S)


def _want_tool(user):
    """Sessions differ in how many searches the agent makes: 1 + (claim hash % 3)."""
    m = re.search(r"\[\[ ## claim ## \]\]\n(.*)", user)
    want = 1 + (sum(map(ord, m.group(1))) % 3 if m else 0)
    return user.count("[[ ## observation_") < want


def _dspy_value(name, typ, system, user):
    if name == "next_tool_name":
        tools = [t for t, _ in _TOOLS.findall(system)]
        if _want_tool(user):
            return next((t for t in tools if t != "finish"), "finish")
        return "finish"
    if name == "next_tool_args":
        tools = dict(_TOOLS.findall(system))
        first = next((t for t in tools if t != "finish"), None)
        if first and _want_tool(user):
            try:
                args = eval(tools[first], {}, {})     # {'query': {'type': 'string'}}
                m = re.search(r"\[\[ ## claim ## \]\]\n(\S+)", user)
                topic = (m.group(1) if m else "topic").strip(".,")
                return json.dumps({k: f"{topic} step {user.count('[[ ## observation_')}" for k in args})
            except Exception:
                return "{}"
        return "{}"
    if not typ:
        return _text(name, COUNT["n"])
    if "Literal[" in typ:
        return re.findall(r"'([^']*)'", typ)[0]
    if "float" in typ:
        return "0.5"
    if "int" in typ:
        return "1"
    if "bool" in typ:
        return "True"
    if "list[" in typ:
        return json.dumps([_text(name, COUNT["n"]), _text(name, COUNT["n"] + 1)])
    if "dict[" in typ:
        return "{}"
    return _text(name, COUNT["n"])


def _dspy_reply(body):
    msgs = body.get("messages") or []
    system = _content(msgs[0]) if msgs and msgs[0].get("role") == "system" else ""
    user = _content(msgs[-1])
    m = re.search(r"Respond with the corresponding output fields, starting with the field (.*)$", user, re.S)
    if not m:
        return "[[ ## completed ## ]]"
    parts = [f"[[ ## {name} ## ]]\n{_dspy_value(name, typ, system, user)}" for name, typ in _FIELD.findall(m.group(1))
             if name != "completed"]
    return "\n\n".join(parts) + "\n\n[[ ## completed ## ]]\n"


# ----------------------------------------------------------------- server
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.endswith("/chat/completions"):
            return self._send(200, {"ok": True})
        with LOCK:
            COUNT["n"] += 1
            msgs = body.get("messages") or []
            first_user = next((_content(m) for m in msgs if m.get("role") == "user"), "")
            is_dspy = "[[ ## " in first_user or "[[ ## " in (_content(msgs[0]) if msgs else "")
            if is_dspy:
                content = _dspy_reply(body)
            else:
                # one garbage answer per verify_claim call so the typed retry shows up
                if first_user.startswith("EvidenceAnalyzer.verify_claim") and body.get("response_format") \
                        and len(msgs) == 2:
                    content = "I cannot decide."
                    COUNT["bad"] += 1
                else:
                    content = _byllm_reply(body)
            rec = {"t": time.time(), "framework": "dspy" if is_dspy else "byllm", "body": body, "reply": content}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self._send(200, {
            "id": f"mock-{COUNT['n']}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


if __name__ == "__main__":
    if CACHE:
        os.makedirs(CACHE, exist_ok=True)
    print(f"recorder on {PORT} -> {LOG}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
