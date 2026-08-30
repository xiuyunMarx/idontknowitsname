"""RequestHandle: one generation turn, shared between the HTTP session that needs the
text and the controller/engine that produces it."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(eq=False)  # identity: handles live in sets
class RequestHandle:
    """The session awaits `done`; the server fills `text` or `error` and sets it.
    `callsite_key` is the history Program's key for the callsite (parsed from the
    request text), the planner's unit of prediction and cost."""
    session: Any                             # serve.session.Session
    callsite_key: str
    model_name: str
    messages: List[Dict[str, Any]]
    kind: str = "call"                       # call | tool_turn | generate (kept for profile.py)
    schema: Optional[Dict[str, Any]] = None  # OpenAI response_format (structured-output grammar)
    call_params: Dict[str, Any] = field(default_factory=dict)   # temperature / max_tokens / stop
    cache_salt: str = ""                     # prefix-cache tenant = the session id; speculation uses the same
    user: str = ""                           # the OpenAI `user` field as sent
    turn: int = 0                            # 0 for the first request of a call, k for its k-th continuation
    text: str = ""
    error: Optional[str] = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    # live progress, written by the engine while the request runs (planner input)
    first_token_at: float = 0.0              # perf_counter of the first token; 0 while prefilling
    out_tokens: int = 0
    prompt_tokens: int = 0
    # lifecycle, written by the controller (queue wait = dispatched - created)
    created_at: float = 0.0                  # perf_counter at submit
    dispatched_at: float = 0.0               # perf_counter when its model was resident and it was released
    done_at: float = 0.0
