"""What a chat-completions body looks like on the wire, independent of any program
knowledge: message text, the ReAct-continuation test. Shared by the controller
and the opaque baselines (CacheScout, Continuum)."""
from typing import Any, Dict, List, Tuple

HINT_HEADER = "Schema requirements:"   # byllm's typed-output hint, appended to the last user message


def content(msg: Dict[str, Any]) -> str:
    """Text of a message as the chat renderer sees it: text parts joined by one
    newline (model.model.Engine.render does the same)."""
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


def set_content(msg: Dict[str, Any], text: str) -> None:
    """Replace a message's content with one text, keeping the part form byllm used."""
    if isinstance(msg.get("content"), list):
        msg["content"] = [{"type": "text", "text": text}]
    else:
        msg["content"] = text


def first_user(body: Dict[str, Any]) -> Dict[str, Any] | None:
    """The call message: the first user message of the transcript."""
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            return m
    return None


def strip_hint(text: str) -> str:
    """The user text without byllm's schema hint (generic: the hint's own bytes
    are known only to a registered template)."""
    i = text.rfind(HINT_HEADER)
    return text[:i].rstrip("\n") if i >= 0 else text


def _norm_msgs(msgs: list) -> List[Tuple[Any, str, str]]:
    """Messages reduced to what identifies a transcript position: role, text with
    the hint stripped (byllm appends it to the LAST user message, so it hops to the
    feedback turn on a typed retry), and the tool calls."""
    return [(m.get("role"), strip_hint(content(m)), str(m.get("tool_calls") or "")) for m in msgs]


def is_continuation(prev: Dict[str, Any], cur: Dict[str, Any]) -> bool:
    """True when `cur` continues the call `prev` started: the next ReAct turn or a
    typed retry. Same model, and cur.messages extends prev.messages (the transcript
    grows by tool/feedback turns) or equals it (byllm re-sending the same body after
    a failed parse)."""
    pm, cm = _norm_msgs(prev.get("messages") or []), _norm_msgs(cur.get("messages") or [])
    if len(cm) < len(pm) or not pm or prev.get("model") != cur.get("model"):
        return False
    # The system message may change between turns: byllm's forced final pass (budget
    # exhausted, tool_choice="none") drops the tool block from it. Compare the rest.
    start = 1 if pm[0][0] == "system" and cm[0][0] == "system" else 0
    return cm[start:len(pm)] == pm[start:]
