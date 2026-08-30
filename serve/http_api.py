"""OpenAI-compatible HTTP front of the controller (aiohttp, on the controller's loop).

    POST /v1/chat/completions   the only request the LLM client makes
    POST /v1/sessions/close     {"user": "program:pid"} from the client's atexit hook
    GET  /health

Every completed request is appended to the trace file as one JSON line
{"session", "t_arrive", "t_done", "request", "response"} — the format
trace_extractor.parser.load_trace reads, so the server's own trace can be replayed
offline into a history file."""
import json
import time
import traceback
import uuid
from typing import Any, Dict, Optional

from aiohttp import web

from serve.handle import RequestHandle

SAMPLING_KEYS = ("temperature", "max_tokens", "stop")


def _text_messages(messages: list) -> list:
    """byllm sends user content as OpenAI multimodal parts ([{"type": "text", "text": ...}]);
    the engine's chat template wants plain strings. Non-text parts are dropped."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
        out.append({**m, "content": c})
    return out


def _error(status: int, message: str, code: str = "invalid_request_error") -> web.Response:
    return web.json_response({"error": {"message": message, "type": code, "code": code}}, status=status)


class Api:
    def __init__(self, ctl: Any, trace_path: Optional[str]) -> None:
        self.ctl = ctl
        self.trace = open(trace_path, "a") if trace_path else None

    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.on_cleanup.append(self._cleanup)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/v1/sessions/close", self.close)
        app.router.add_get("/health", self.health)
        return app

    async def _cleanup(self, app: web.Application) -> None:
        if self.trace is not None:
            self.trace.close()

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "models": sorted(self.ctl.engine_pool)})

    async def close(self, request: web.Request) -> web.Response:
        body = await request.json()
        user = str(body.get("user") or "")
        closed = self.ctl.close_session(user)
        return web.json_response({"closed": closed, "user": user})

    async def chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "body is not JSON")
        model, messages = body.get("model"), body.get("messages")
        if not isinstance(model, str) or not isinstance(messages, list) or not messages:
            return _error(400, "`model` and a non-empty `messages` list are required")
        if model not in self.ctl.engine_pool:
            return _error(404, f"model {model!r} is not served", "model_not_found")
        peer = request.remote or "?"
        user = str(body.get("user") or f"anon:{peer}")
        t_arrive = time.time()
        sess = self.ctl.session(user)
        try:
            site, inst, kind, turn = sess.begin(body, t_arrive)
        except Exception as e:
            traceback.print_exc()
            return _error(400, f"cannot parse request as a byllm callsite: {e!r}")
        handle = RequestHandle(session=sess, callsite_key=site.key, model_name=model, messages=_text_messages(messages), kind=kind,
                               schema=body.get("response_format"),
                               call_params={k: body[k] for k in SAMPLING_KEYS if body.get(k) is not None},
                               cache_salt=sess.salt(), user=user, turn=turn)
        try:
            await self.ctl.submit(handle)
            await handle.done.wait()
        except Exception as e:
            traceback.print_exc()
            handle.error = repr(e)
        t_done = time.time()
        engine_s = max(0.0, handle.done_at - handle.dispatched_at) if handle.dispatched_at else 0.0
        if handle.error is not None:
            sess.finish(site, inst, "", t_done)
            return _error(500, handle.error, "server_error")
        sess.finish(site, inst, handle.text, t_done, engine_s)
        self._trace_line(user, t_arrive, t_done, body, handle.text, engine_s)
        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(t_done),
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": handle.text}}],
            "usage": {"prompt_tokens": handle.prompt_tokens, "completion_tokens": handle.out_tokens,
                      "total_tokens": handle.prompt_tokens + handle.out_tokens},
        })

    def _trace_line(self, user: str, t_arrive: float, t_done: float, body: Dict[str, Any], text: str,
                    engine_s: float) -> None:
        if self.trace is None:
            return
        self.trace.write(json.dumps({"session": user, "t_arrive": t_arrive, "t_done": t_done,
                                     "engine_s": round(engine_s, 4), "request": body, "response": text}) + "\n")
        self.trace.flush()


async def start(ctl: Any, host: str, port: int, trace_path: Optional[str] = None) -> web.AppRunner:
    runner = web.AppRunner(Api(ctl, trace_path).app())
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
