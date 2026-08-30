"""Controller planning wiring with fake engines and instances (no GPU; imports vllm for the
SamplingParams import in serve.controller, so this is slower than the pure planner tests)."""
import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from serve.controller import Controller
from static_pass.primitives import RequestHandle


class FakeEngine:
    def __init__(self, name, tier="ssd"):
        self.name, self.tier, self.transitioning, self.serving, self.busy = name, tier, False, False, False
        self.killed = self.offloaded = 0

    @property
    def resident(self):
        return self.tier == "gpu"

    async def offload(self):
        self.tier, self.offloaded = "host", self.offloaded + 1

    async def kill(self):
        self.tier, self.killed = "ssd", self.killed + 1


@dataclass
class FakeCall:
    model_name: str
    site: object
    finished: bool = False


class FakeProgram:
    """Branch predictor stub: a straight line through the given (callsite, model) list."""

    def __init__(self, predicted):
        self.predicted = list(predicted)
        self.callsites = {k: k for k, _ in predicted}
        self.unresolved_models = set()

    def branch_candidates(self, key):
        keys = [k for k, _ in self.predicted]
        if key not in keys:
            return [(keys[0], 1.0)] if keys else []
        i = keys.index(key)
        return [(keys[i + 1], 1.0)] if i + 1 < len(keys) else []

    def model_of(self, site, fallback):
        return dict(self.predicted)[site]


class FakeInstance:
    def __init__(self, predicted, calls=()):
        self.predicted = list(predicted)                     # [(callsite_key, model)]
        self.calls = {i: c for i, c in enumerate(calls)}
        self.program = FakeProgram(predicted)


def handle(inst, key, model):
    return RequestHandle(inst, key, model, [{"role": "user", "content": "x"}])


class Planning(unittest.TestCase):
    def setUp(self):
        self.c = Controller(gpu_slots=1, host_slots=3, speculate=False)
        for m, secs in [("big-4B", 8.0), ("small-0.5B", 1.0), ("mid-1.5B", 3.0)]:
            self.c.engine_pool[m] = FakeEngine(m, "host")
            self.c.cost.register(m)
            self.c.cost.observe_load(m, "host", secs)
            self.c.cost.observe_load(m, "ssd", 3 * secs)
        self.a = FakeInstance([("a2", "small-0.5B"), ("a3", "big-4B")])
        self.b = FakeInstance([])
        self.c.instances[("p", 1)], self.c.instances[("p", 2)] = self.a, self.b
        for inst, key, model in [(self.a, "a1", "big-4B"), (self.b, "b1", "small-0.5B")]:
            h = handle(inst, key, model)
            asyncio.run(self.c.submit(h))

    def test_fifo_unchanged(self):
        self.assertEqual(self.c.sequences(), [["big-4B", "small-0.5B", "big-4B"], ["small-0.5B"]])
        self.assertEqual(self.c.plan(), ["big-4B", "small-0.5B"])

    def test_cost_aware_planners_reorder(self):
        for planner in ("greedy", "dp"):
            asyncio.run(self.c.control({"set": {"planner": planner}}))
            order = self.c.plan()
            self.assertEqual(order[0], "small-0.5B", planner)
            self.assertEqual(self.c.plan_stats["deviations"], self.c.plan_stats["plans"])
        # same state -> cached, no new plan
        plans = self.c.plan_stats["plans"]
        self.c.plan()
        self.assertEqual(self.c.plan_stats["plans"], plans)

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
        inst = FakeInstance([("z2", "mid-1.5B")], calls=[FakeCall("big-4B", SimpleNamespace(callsite_key="z1"))])
        self.c.instances[("p", 3)] = inst
        self.assertIn("big-4B", self.c._pinned())
        self.assertEqual(self.c.sequences()[-1], ["big-4B", "mid-1.5B"])
        pi = self.c.plan_input()
        self.assertEqual(pi.chains[-1].head, "open")
        self.assertIn("big-4B", pi.pinned)

    def test_stats_and_reset(self):
        h = next(iter(self.c.pending["big-4B"]))
        h.dispatched_at, h.first_token_at, h.done_at = h.created_at + 1.0, h.created_at + 1.5, h.created_at + 2.0
        self.c.requests.append({"callsite": "a1", "kind": "call", "model": "big-4B", "created_at": h.created_at,
                                "dispatched_at": h.dispatched_at, "first_token_at": h.first_token_at,
                                "done_at": h.done_at, "output_tokens": 3, "error": None})
        out = asyncio.run(self.c.control({"stats": True, "detail": True, "cost": True}))
        self.assertAlmostEqual(out["stats"]["queue_wait_ms"]["mean"], 1000.0)
        self.assertAlmostEqual(out["stats"]["ttft_submit_ms"]["mean"], 1500.0)
        self.assertEqual(len(out["requests"]), 1)
        self.assertIn("big-4B|host", out["cost"]["load_s"])
        with self.assertRaises(RuntimeError):
            asyncio.run(self.c.control({"reset": True}))          # requests pending
        self.c.pending.clear()
        for e in self.c.engine_pool.values():
            e.engine = SimpleNamespace(reset_prefix_cache=self._noop)
        asyncio.run(self.c.control({"reset": True, "cold": True}))
        self.assertTrue(all(e.tier == "ssd" for e in self.c.engine_pool.values()))
        self.assertEqual(self.c.requests, [])
        self.assertIn("big-4B|host", self.c.cost.export()["load_s"])   # knowledge survives reset

    @staticmethod
    async def _noop():
        return None


if __name__ == "__main__":
    unittest.main()
