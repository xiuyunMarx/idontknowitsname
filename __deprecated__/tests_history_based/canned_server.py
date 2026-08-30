"""A stand-in guard server for the Jac client smoke test: records every request and answers
a scripted ReAct (one tool call, then finish_tool). Usage: python -m tests.canned_server PORT OUT.json"""
import asyncio
import json
import sys

from aiohttp import web

REPLIES = ['<tool_call>{"name": "get_count", "arguments": {}}',
           '<tool_call>{"name": "finish_tool", "arguments": {"final_output": "apples: 7 items"}}']


def make_app(log: list) -> web.Application:
    async def chat(request):
        body = await request.json()
        log.append({"path": "/v1/chat/completions", "body": body})
        n = sum(1 for e in log if e["path"] == "/v1/chat/completions") - 1
        text = REPLIES[min(n, len(REPLIES) - 1)]
        return web.json_response({"model": body.get("model"), "choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": text}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    async def close(request):
        body = await request.json()
        log.append({"path": "/v1/sessions/close", "body": body})
        return web.json_response({"closed": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/sessions/close", close)
    return app


async def serve(port: int, out: str, log: list) -> None:
    runner = web.AppRunner(make_app(log))
    await runner.setup()
    await web.TCPSite(runner, "localhost", port).start()
    try:
        await asyncio.Event().wait()
    finally:
        with open(out, "w") as f:
            json.dump(log, f)


if __name__ == "__main__":
    asyncio.run(serve(int(sys.argv[1]), sys.argv[2], []))
