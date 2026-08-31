"""HTTP transport that owns all connection management.

The server accepts OpenAI-style requests, validates them, and parks each one in
`pool` as a PendingRequest: the parsed JSON body plus an unresolved future. The
HTTP response is written when that future resolves; nothing else ever reaches
the controller, whose whole interface is:

    server = HttpServer(port=8964)
    await server.start()
    while True:
        req = await server.pool.get()          # PendingRequest
        asyncio.create_task(work(req))         # task per request: reply() is what
                                               # unblocks the HTTP response, a
                                               # sequential loop serializes clients
    ... req.reply(text_or_message_dict) ... or req.fail(exc)  -> HTTP 500

kind == "completion": body is a /v1/chat/completions JSON; reply with the
    assistant text (str) or a full assistant message dict (native tool calls).
kind == "close": explicit session end (/v1/sessions/close, body {"user": id} —
    the path InterceptorLLM's atexit hook posts to);
    reply with a bool (was the session known).
GET /health is answered here directly and never enters the pool.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Union

from aiohttp import web

MAX_BODY_BYTES = 64 * 1024 * 1024  # byllm prompts far exceed aiohttp's 1MB default
REPLY_TIMEOUT_S = 600.0            # no reply by then: the client gets a 504


@dataclass
class PendingRequest:
    """One accepted HTTP request, parked in the pool until the controller replies."""
    kind: str                      # "completion" | "close"
    body: Dict[str, Any]           # parsed JSON body, as sent
    peer: str                      # remote address
    session: str                   # body["user"], or "anon:<peer>" when absent
    t_arrive: float                # time.monotonic() at acceptance
    _fut: asyncio.Future = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fut = asyncio.get_running_loop().create_future()

    def reply(self, result: Union[str, Dict[str, Any], bool]) -> None:
        if not self._fut.done():   # the client may have disconnected or timed out
            self._fut.set_result(result)

    def fail(self, exc: BaseException) -> None:
        if not self._fut.done():
            self._fut.set_exception(exc)


def _error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "invalid_request_error"}}, status=status)


def _envelope(model: str, message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class HttpServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8964,
                 reply_timeout_s: float = REPLY_TIMEOUT_S) -> None:
        self.host, self.port = host, port
        self.reply_timeout_s = reply_timeout_s
        self.pool: "asyncio.Queue[PendingRequest]" = asyncio.Queue()
        self._runner: web.AppRunner | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Bind and listen; returns once the port is open. Requests fetched from
        `pool` before the controller is ready simply wait there."""
        app = web.Application(client_max_size=MAX_BODY_BYTES)
        app.router.add_post("/v1/chat/completions", self._completions)
        app.router.add_post("/v1/sessions/close", self._close)
        app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()
        print(f"[http] listening on {self.host}:{self.port}", flush=True)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------------ handlers
    async def _submit(self, kind: str, body: Dict[str, Any], peer: str) -> Any:
        """Park the request in the pool and wait for the controller's reply."""
        sid = str(body.get("user") or f"anon:{peer}")
        req = PendingRequest(kind=kind, body=body, peer=peer, session=sid,
                             t_arrive=time.monotonic())
        await self.pool.put(req)
        return await asyncio.wait_for(asyncio.shield(req._fut), self.reply_timeout_s)

    async def _completions(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "request body is not valid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return _error(400, "body must be a JSON object with a `messages` list")
        if body.get("stream"):
            return _error(400, "streaming is not supported")
        try:
            result = await self._submit("completion", body, request.remote or "")
        except asyncio.TimeoutError:
            return _error(504, f"no reply within {self.reply_timeout_s:.0f}s")
        except Exception as e:  # controller called fail(); it stays free of HTTP codes
            return _error(500, f"{type(e).__name__}: {e}")
        message = result if isinstance(result, dict) else {"role": "assistant", "content": result}
        return web.json_response(_envelope(body.get("model", ""), message))

    async def _close(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "request body is not valid JSON")
        if not isinstance(body, dict) or not body.get("user"):
            return _error(400, "body must carry the session id in `user`")
        try:
            closed = await self._submit("close", body, request.remote or "")
        except asyncio.TimeoutError:
            return _error(504, f"no reply within {self.reply_timeout_s:.0f}s")
        except Exception as e:
            return _error(500, f"{type(e).__name__}: {e}")
        return web.json_response({"closed": bool(closed), "session": str(body["user"])})

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "pooled": self.pool.qsize()})
