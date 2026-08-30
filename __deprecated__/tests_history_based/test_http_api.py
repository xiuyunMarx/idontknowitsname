"""The OpenAI-compatible endpoint end to end on a fake engine: request shape, sessions keyed by
`user`, ReAct open-call tracking, trace file, session close -> learned workflow."""
import asyncio
import json
import os
import tempfile
import unittest

from aiohttp.test_utils import TestClient, TestServer

from serve.controller import Controller
from serve.http_api import Api
from tests import wire
from tests.fakes import FakeEngine


def reply(prompt):
    if "web_research" in prompt and "<tool_response>" not in prompt:
        return '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}'
    if "web_research" in prompt:
        return '<tool_call>{"name": "finish_tool", "arguments": {"final_output": "done"}}'
    return "canned"


class Http(unittest.TestCase):
    def setUp(self):
        self.c = Controller(gpu_slots=2, host_slots=3, speculate=False)
        for m in (wire.SUMMARIZE_MODEL, wire.ROUTER_MODEL, wire.RESEARCH_MODEL, wire.DECOMPOSE_MODEL):
            self.c.engine_pool[m] = FakeEngine(m, "gpu", reply=reply)
            self.c.cost.register(m)
        fd, self.trace = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.trace)

    def run_client(self, coro):
        async def go():
            async with TestClient(TestServer(Api(self.c, self.trace).app())) as client:
                return await coro(client)
        return asyncio.run(go())

    def test_plain_request_and_session(self):
        async def go(client):
            body = {**wire.summarize_req("r", "shipping", "a"), "user": "triage:11"}
            r = await client.post("/v1/chat/completions", json=body)
            self.assertEqual(r.status, 200)
            j = await r.json()
            self.assertEqual(j["object"], "chat.completion")
            self.assertEqual(j["model"], wire.SUMMARIZE_MODEL)
            self.assertEqual(j["choices"][0]["message"], {"role": "assistant", "content": "canned"})
            self.assertEqual(j["choices"][0]["finish_reason"], "stop")
            self.assertEqual(j["usage"]["completion_tokens"], 3)
            self.assertGreater(j["usage"]["prompt_tokens"], 0)
            return j
        self.run_client(go)
        sess = self.c.sessions["triage:11"]
        self.assertEqual(sess.program.name, "triage")
        self.assertEqual([self.c.programs["triage"].label(k) for k in sess.builder.session.sequence], ["Agent.summarize"])
        self.assertEqual(sess.inflight, 0)
        self.assertIsNone(sess.open_call)
        self.assertEqual(len(self.c.requests), 1)
        self.assertEqual(self.c.requests[0]["kind"], "call")
        with open(self.trace) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session"], "triage:11")
        self.assertEqual(rows[0]["response"], "canned")
        self.assertEqual(rows[0]["request"]["messages"], wire.summarize_req("r", "shipping", "a")["messages"])
        self.assertEqual(self.c.stats[0][0], sess.builder.session.sequence[0])   # cost keyed by the callsite key

    def test_list_content_is_rendered_as_text(self):
        async def go(client):
            body = wire.summarize_req("r", "shipping", "a")
            body["messages"][1]["content"] = [{"type": "text", "text": body["messages"][1]["content"]}]
            r = await client.post("/v1/chat/completions", json={**body, "user": "triage:12"})
            self.assertEqual(r.status, 200)
        self.run_client(go)
        eng = self.c.engine_pool[wire.SUMMARIZE_MODEL]
        self.assertIn("<user>Agent.summarize(request: str", eng.rendered[-1])
        self.assertNotIn("'type': 'text'", eng.rendered[-1])
        self.assertEqual(self.c.requests[-1]["callsite"], "Agent.summarize")

    def test_errors(self):
        async def go(client):
            r = await client.post("/v1/chat/completions", json={"model": "nope", "messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(r.status, 404)
            self.assertEqual((await r.json())["error"]["code"], "model_not_found")
            r = await client.post("/v1/chat/completions", json={"model": wire.SUMMARIZE_MODEL})
            self.assertEqual(r.status, 400)
            r = await client.get("/health")
            self.assertEqual(r.status, 200)
        self.run_client(go)

    def test_react_visit_close_and_learning(self):
        async def go(client):
            user = "deep_research:5"
            turns = wire.research_turns("t", native=False)
            r = await client.post("/v1/chat/completions", json={**turns[0], "user": user})
            self.assertIn("web_search", (await r.json())["choices"][0]["message"]["content"])
            sess = self.c.sessions[user]
            self.assertIsNotNone(sess.open_call)              # mid-call: the client runs the tool now
            self.assertIn(sess.open_call[1], self.c._pinned())
            r = await client.post("/v1/chat/completions", json={**turns[1], "user": user})
            self.assertIn("finish_tool", (await r.json())["choices"][0]["message"]["content"])
            self.assertIsNone(sess.open_call)
            self.assertEqual(len(sess.calls), 1)
            self.assertEqual(len(sess.calls[0].turns), 2)
            self.assertEqual([r["kind"] for r in self.c.requests], ["call", "tool_turn"])
            r = await client.post("/v1/chat/completions", json={**turns[2], "user": user})   # forced final pass
            self.assertEqual(r.status, 200)
            self.assertEqual(len(sess.calls), 1)
            self.assertEqual(len(sess.calls[0].turns), 3)
            self.assertEqual(len(self.c.programs["deep_research"].callsites), 1)
            r = await client.post("/v1/chat/completions", json={**wire.visit_req("q"), "user": user})
            self.assertEqual(r.status, 200)
            self.assertEqual(self.c.requests[-1]["kind"], "generate")
            self.assertEqual(len(sess.calls), 2)
            r = await client.post("/v1/sessions/close", json={"user": user})
            self.assertEqual((await r.json())["closed"], True)
            self.assertNotIn(user, self.c.sessions)
        self.run_client(go)
        prog = self.c.programs["deep_research"]
        research, visit = list(prog.callsites)
        self.assertEqual(prog.branch_candidates(research), [(visit, 1.0)])
        self.assertEqual(prog.end_prob(visit), 1.0)
        self.assertEqual(prog.turns[research], [3])
        self.assertEqual(prog.entry[research], 1)
        # a second session of the same program predicts from the learned workflow
        async def again(client):
            r = await client.post("/v1/chat/completions", json={**wire.research_turns("u", native=False)[0], "user": "deep_research:6"})
            self.assertEqual(r.status, 200)
        self.run_client(again)
        self.assertEqual(self.c.sessions["deep_research:6"].predicted, [(visit, wire.ROUTER_MODEL)])

    def test_idle_reaper(self):
        async def go(client):
            r = await client.post("/v1/chat/completions", json={**wire.summarize_req("r", "d", "a"), "user": "triage:1"})
            self.assertEqual(r.status, 200)
            self.c.session_idle_s = 0.0
            self.c.sessions["triage:1"].last_seen -= 1.0
            import serve.controller as sc
            sc.REAP_EVERY_S = 0.01
            task = asyncio.ensure_future(self.c._reap_sessions())
            await asyncio.sleep(0.05)
            task.cancel()
        self.run_client(go)
        self.assertEqual(self.c.sessions, {})
        self.assertEqual(sum(self.c.programs["triage"].entry.values()), 1)


if __name__ == "__main__":
    unittest.main()
