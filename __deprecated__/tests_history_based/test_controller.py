"""Controller planning wiring with fake engines and history-driven sessions (no GPU; imports vllm
for the SamplingParams import in serve.controller, so this is slower than the pure planner tests)."""
import asyncio
import unittest
from types import SimpleNamespace

from serve.controller import Controller
from serve.handle import RequestHandle
from serve.predict import predicted_chain
from serve.session import Session
from tests.fakes import FakeEngine, line_program


def session(ctl, user, chain, current=None):
    """A live session of a program whose history is the straight line `chain`; `current` is the
    callsite it is on now (its prediction is what follows `current`)."""
    sess = Session(user, line_program(user.split(":")[0], chain), 12)
    if current is not None:
        sess.predicted = predicted_chain(sess.program, current, "", 12)
    ctl.sessions[user] = sess
    return sess


def handle(sess, key, model):
    return RequestHandle(sess, key, model, [{"role": "user", "content": "x"}], cache_salt=sess.salt())


class Planning(unittest.TestCase):
    def setUp(self):
        self.c = Controller(gpu_slots=1, host_slots=3, speculate=False)
        for m, secs in [("big-4B", 8.0), ("small-0.5B", 1.0), ("mid-1.5B", 3.0)]:
            self.c.engine_pool[m] = FakeEngine(m, "host")
            self.c.cost.register(m)
            self.c.cost.observe_load(m, "host", secs)
            self.c.cost.observe_load(m, "ssd", 3 * secs)
        self.a = session(self.c, "p:1", [("a1", "big-4B"), ("a2", "small-0.5B"), ("a3", "big-4B")], current="a1")
        self.b = session(self.c, "p:2", [("b1", "small-0.5B")], current="b1")
        for sess, key, model in [(self.a, "a1", "big-4B"), (self.b, "b1", "small-0.5B")]:
            asyncio.run(self.c.submit(handle(sess, key, model)))

    def test_fifo_unchanged(self):
        self.assertEqual(self.c.sequences(), [["big-4B", "small-0.5B", "big-4B"], ["small-0.5B"]])
        self.assertEqual(self.c.plan(), ["big-4B", "small-0.5B"])

    def test_cost_aware_planners_reorder(self):
        for planner in ("greedy", "dp"):
            asyncio.run(self.c.control({"set": {"planner": planner}}))
            order = self.c.plan()
            self.assertEqual(order[0], "small-0.5B", planner)
            self.assertEqual(self.c.plan_stats["deviations"], self.c.plan_stats["plans"])
        plans = self.c.plan_stats["plans"]
        self.c.plan()
        self.assertEqual(self.c.plan_stats["plans"], plans)   # same state -> cached

    def test_victim_by_keep_value(self):
        asyncio.run(self.c.control({"set": {"planner": "dp", "gpu_slots": 2}}))
        self.c.engine_pool["big-4B"].tier = self.c.engine_pool["mid-1.5B"].tier = "gpu"
        self.c.pending.clear()
        asyncio.run(self.c.submit(handle(self.b, "b1", "small-0.5B")))   # only the small model is wanted
        self.assertEqual(self.c.plan()[0], "small-0.5B")
        victim = self.c._victim(self.c.plan())
        self.assertEqual(victim.name, "mid-1.5B")     # big-4B is on a's predicted chain, mid is on nobody's
        self.assertEqual(self.c._pinned(), set())

    def test_open_call_is_pinned_and_a_head(self):
        sess = session(self.c, "p:3", [("z1", "big-4B"), ("z2", "mid-1.5B")], current="z1")
        sess.open_call = ("z1", "big-4B")
        self.assertIn("big-4B", self.c._pinned())
        self.assertEqual(self.c.sequences()[-1], ["big-4B", "mid-1.5B"])
        pi = self.c.plan_input()
        self.assertEqual(pi.chains[-1].head, "open")
        self.assertIn("big-4B", pi.pinned)

    def test_load_prefills_predicted_prefixes(self):
        self.c.pending.clear()
        eng = self.c.engine_pool["small-0.5B"]
        asyncio.run(self.c._load("small-0.5B"))          # a's next predicted callsite a2 is on this model
        self.assertEqual(eng.loaded, 1)
        self.assertEqual(len(eng.prompts), 1)
        prompt, salt = eng.prompts[0]
        self.assertEqual(salt, "p:1")
        self.assertTrue(prompt.endswith("a2()"))         # the fixed prefix, no assistant prompt
        self.assertEqual(len(self.c.loads), 1)

    def test_stats_reset_and_frozen_history(self):
        h = next(iter(self.c.pending["big-4B"]))
        h.dispatched_at, h.first_token_at, h.done_at = h.created_at + 1.0, h.created_at + 1.5, h.created_at + 2.0
        self.c.requests.append({"callsite": "a1", "kind": "call", "model": "big-4B", "created_at": h.created_at,
                                "dispatched_at": h.dispatched_at, "first_token_at": h.first_token_at,
                                "done_at": h.done_at, "output_tokens": 3, "error": None})
        out = asyncio.run(self.c.control({"stats": True, "detail": True, "cost": True, "history": True}))
        self.assertAlmostEqual(out["stats"]["queue_wait_ms"]["mean"], 1000.0)
        self.assertAlmostEqual(out["stats"]["ttft_submit_ms"]["mean"], 1500.0)
        self.assertEqual(len(out["requests"]), 1)
        self.assertIn("big-4B|host", out["cost"]["load_s"])
        self.assertEqual(out["history"], {})               # sessions were built on private programs
        with self.assertRaises(RuntimeError):
            asyncio.run(self.c.control({"reset": True}))    # requests pending
        self.c.pending.clear()
        for e in self.c.engine_pool.values():
            e.engine = SimpleNamespace(reset_prefix_cache=self._noop)
        # freeze the learned workflow, learn something new, reset -> the snapshot is back
        self.c.programs["p"] = self.a.program
        asyncio.run(self.c.control({"freeze_branches": True}))
        self.c.programs["p"].succ["a1"]["zzz"] += 5
        asyncio.run(self.c.control({"reset": True, "cold": True}))
        self.assertTrue(all(e.tier == "ssd" for e in self.c.engine_pool.values()))
        self.assertEqual(self.c.requests, [])
        self.assertEqual(self.c.sessions, {})
        self.assertNotIn("zzz", self.c.programs["p"].succ["a1"])
        self.assertIn("big-4B|host", self.c.cost.export()["load_s"])   # knowledge survives reset

    @staticmethod
    async def _noop():
        return None


if __name__ == "__main__":
    unittest.main()
