"""The rewritten InterceptorLLM on the wire: run tests/smoke_client.jac (a real `jac run`, skipped
when `jac` is not on PATH) against the canned server and check what it sent."""
import asyncio
import os
import shutil
import subprocess
import unittest

from tests.canned_server import make_app

PORT = 8970


def text(content):
    """Message content as text: byllm sends user content as [{"type": "text", "text": ...}] parts."""
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if p.get("type") == "text")
    return content
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(shutil.which("jac"), "jac not on PATH")
class ClientWire(unittest.TestCase):
    def test_react_call_over_http(self):
        log = []

        async def go():
            from aiohttp import web
            runner = web.AppRunner(make_app(log))
            await runner.setup()
            await web.TCPSite(runner, "localhost", PORT).start()
            try:
                proc = await asyncio.create_subprocess_exec(
                    "jac", "run", os.path.join(ROOT, "tests", "smoke_client.jac"), cwd=ROOT,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await asyncio.wait_for(proc.communicate(), 300)
                return proc.returncode, out.decode(errors="replace"), proc.pid
            finally:
                await runner.cleanup()
        rc, out, pid = asyncio.run(go())
        self.assertEqual(rc, 0, out[-2000:])
        self.assertIn("RESULT: apples: 7 items", out)
        chats = [e["body"] for e in log if e["path"] == "/v1/chat/completions"]
        self.assertEqual(len(chats), 2, [e["path"] for e in log])
        first, second = chats
        self.assertEqual(first["user"], f"smoke_client:{pid}")
        self.assertEqual(first["model"], "fake-model")
        self.assertNotIn("tools", first)                                  # text tool protocol
        self.assertNotIn("metadata", first)
        self.assertEqual(first["stop"], ["</tool_call>"])
        self.assertEqual(first["temperature"], 0.0)
        self.assertEqual(first["max_tokens"], 64)
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertIn("# Calling tools", first["messages"][0]["content"])
        self.assertIn("- get_count:", first["messages"][0]["content"])
        self.assertIsInstance(first["messages"][1]["content"], list)              # multimodal parts
        self.assertTrue(text(first["messages"][1]["content"]).startswith("describe(topic: str) -> str --- Say how many items topic has"), text(first["messages"][1]["content"]))
        self.assertIn("topic = 'apples'", text(first["messages"][1]["content"]))
        # turn 2 extends turn 1's transcript with the tool call and its result as text
        self.assertEqual(second["messages"][:2], first["messages"])
        self.assertEqual(second["messages"][2]["role"], "assistant")
        self.assertIn("<tool_call>", text(second["messages"][2]["content"]))
        self.assertEqual(second["messages"][3]["role"], "user")
        self.assertIn("<tool_response>get_count -> 7", text(second["messages"][3]["content"]))
        closes = [e["body"] for e in log if e["path"] == "/v1/sessions/close"]
        self.assertEqual(closes, [{"user": f"smoke_client:{pid}"}])


if __name__ == "__main__":
    unittest.main()
