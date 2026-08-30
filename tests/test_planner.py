import random
import unittest

from serve.planner import (Chain, PlanInput, Step, choose, dp_order, evaluate, fifo_order, first_nonresident,
                           greedy_order, keep_value, next_use, root, step)
from serve.sim import Scenario, simulate

# b = a 4B-class model, s = a 0.5B-class model, m = something in between (seconds)
LOAD = {"b": {"host": 8.0, "ssd": 20.0}, "s": {"host": 1.0, "ssd": 3.0}, "s2": {"host": 1.0, "ssd": 3.0},
        "m": {"host": 3.0, "ssd": 8.0}}


def fifo(pi: PlanInput):
    return fifo_order([[s.model for s in c.steps] for c in pi.chains])


def load_s(model: str, tier: str) -> float:
    return 0.0 if tier == "gpu" else LOAD[model][tier]


def run_all(scn: Scenario):
    return {k: simulate(scn, f) for k, f in [("fifo", fifo), ("greedy", greedy_order), ("dp", dp_order)]}


def plan_input(chains, resident=(), gpu_slots=1, busy=(), rate=None, tier=None):
    """Chains as [(head_kind, [(model, exec, weight), ...]), ...]; the first step is the head."""
    cs = [Chain(i, tuple(Step(m, ex, w, predicted=k > 0) for k, (m, ex, w) in enumerate(steps)), head)
          for i, (head, steps) in enumerate(chains)]
    return PlanInput(cs, frozenset(resident), tier or {m: "host" for m in LOAD}, load_s, rate or {}, gpu_slots,
                     busy=frozenset(busy))


class Ordering(unittest.TestCase):
    def test_cost_heterogeneity(self):
        """A (first) waits for the big model, B for the small one: cost-aware planners serve B first."""
        r = run_all(Scenario(LOAD, [[("b", 2.0)], [("s", 0.5)]]))
        self.assertEqual(r["fifo"].actions, ["b", "s"])
        self.assertEqual(r["greedy"].actions, ["s", "b"])
        self.assertEqual(r["dp"].actions, ["s", "b"])
        self.assertAlmostEqual(r["fifo"].wait, 21.5)
        self.assertAlmostEqual(r["greedy"].wait, 13.0)
        self.assertAlmostEqual(r["dp"].wait, 13.0)

    def test_demand_aggregation(self):
        """One early waiter on b, three later waiters on s: serve the three first."""
        r = run_all(Scenario(LOAD, [[("b", 2.0)], [("s", 0.5)], [("s", 0.5)], [("s", 0.5)]]))
        self.assertEqual(r["fifo"].actions[0], "b")
        self.assertEqual(r["greedy"].actions[0], "s")
        self.assertEqual(r["dp"].actions[0], "s")
        self.assertLessEqual(r["dp"].wait, r["greedy"].wait)
        self.assertLess(r["greedy"].wait, r["fifo"].wait)

    def test_lookahead_beats_greedy(self):
        """Greedy takes the cheap s now and pays for it later; the DP sees the m-revisit."""
        scn = Scenario(LOAD, [[("s", 1.0), ("m", 2.0), ("m", 2.0)], [("m", 0.5)], [("s", 2.0), ("s", 1.0)]],
                       arrivals=[0.0, 0.0, 5.0])
        r = run_all(scn)
        self.assertEqual(r["greedy"].actions, ["s", "m", "s"])
        self.assertEqual(r["dp"].actions, ["m", "s", "m"])
        self.assertLess(r["dp"].wait, r["greedy"].wait)
        self.assertLessEqual(r["greedy"].wait, r["fifo"].wait)

    def test_random_scenarios_on_average(self):
        """Across seeded random scenarios: dp <= greedy <= fifo in total, and dp rarely loses to greedy."""
        rng = random.Random(20260829)
        tot = {"fifo": 0.0, "greedy": 0.0, "dp": 0.0}
        dp_loses = 0
        for _ in range(200):
            n = rng.randint(2, 5)
            chains = [[(rng.choice(["b", "s", "m"]), rng.choice([0.5, 1.0, 2.0])) for _ in range(rng.randint(1, 4))]
                      for _ in range(n)]
            arrivals = [0.0] + sorted(rng.choice([0.0, 2.0, 5.0]) for _ in range(n - 1))
            r = run_all(Scenario(LOAD, chains, arrivals=arrivals, gpu_slots=rng.choice([1, 1, 2])))
            for k in tot:
                tot[k] += r[k].wait
            dp_loses += r["dp"].wait > r["greedy"].wait + 1e-9
        self.assertLess(tot["dp"], tot["greedy"])
        self.assertLess(tot["greedy"], tot["fifo"])
        self.assertLess(dp_loses, 20)


class Semantics(unittest.TestCase):
    def test_running_head_is_not_a_load_target(self):
        """A runs on resident s (remaining 0.3 s); B waits for b: the next load is b, and s is priced once."""
        pi = plan_input([("running", [("s", 0.3, 1.0)]), ("waiting", [("b", 2.0, 1.0)])], resident=["s"], busy=["s"])
        for order in (greedy_order(pi), dp_order(pi)):
            self.assertEqual(first_nonresident(order, pi.resident), "b")
        cost, wall = evaluate(pi, ["s", "b"])
        self.assertAlmostEqual(wall, 0.3 + 8.0 + 2.0)
        self.assertAlmostEqual(cost, 0.3 * 2 + 10.0 * 1)

    def test_busy_resident_is_not_evicted_at_root(self):
        pi = plan_input([("running", [("s", 0.3, 1.0)]), ("waiting", [("b", 2.0, 1.0)])], resident=["s"], busy=["s"])
        self.assertIsNone(step(pi, root(pi), "b"))          # K=1: nothing can make room while s is busy
        self.assertEqual(dp_order(pi)[0], "s")               # so the plan is: wait for s, then b

    def test_consecutive_same_model_steps_share_one_residency(self):
        pi = plan_input([("waiting", [("b", 1.0, 1.0), ("b", 1.0, 0.7), ("s", 0.5, 0.49)])])
        st, cost, wall = step(pi, root(pi), "b")
        self.assertEqual(st.prog, (2,))
        self.assertAlmostEqual(wall, 8.0 + 2.0)

    def test_k2_eviction_prefers_the_model_needed_latest(self):
        """Resident {b, s} idle; a new chain [s2, b] arrives: s goes (b is needed right after s2)."""
        pi = plan_input([("waiting", [("s2", 0.5, 1.0), ("b", 2.0, 0.7)])], resident=["b", "s"], gpu_slots=2)
        nu = next_use(pi)
        self.assertLess(keep_value(pi, "s", nu), keep_value(pi, "b", nu))
        st, _, _ = step(pi, root(pi), "s2")
        self.assertEqual(st.resident, frozenset({"b", "s2"}))

    def test_arrival_rate_keeps_a_hot_model(self):
        """No chain needs b or s, but b is requested every 5 s: evict s."""
        pi = plan_input([("waiting", [("s2", 0.5, 1.0)])], resident=["b", "s"], gpu_slots=2, rate={"b": 0.2})
        st, _, _ = step(pi, root(pi), "s2")
        self.assertEqual(st.resident, frozenset({"b", "s2"}))

    def test_k2_prefetch(self):
        r = run_all(Scenario(LOAD, [[("b", 2.0)], [("s", 0.5)], [("s2", 0.5)]], gpu_slots=2))
        self.assertLessEqual(r["dp"].wait, r["greedy"].wait)
        self.assertLessEqual(r["greedy"].wait, r["fifo"].wait)
        self.assertLess(r["greedy"].wall, 8.0 + 2.0 + 1.0 + 0.5 + 1.0 + 0.5)   # some load overlapped a batch

    def test_starvation_cap(self):
        """A waiter aged past 6 tau (weight >= 7) goes first even though s is far cheaper per unit."""
        pi = plan_input([("waiting", [("b", 2.0, 11.0)]), ("waiting", [("s", 0.5, 1.0)])])
        self.assertEqual(greedy_order(pi)[0], "b")
        pi = plan_input([("waiting", [("b", 2.0, 2.0)]), ("waiting", [("s", 0.5, 1.0)])])
        self.assertEqual(greedy_order(pi)[0], "s")

    def test_hysteresis(self):
        """Two orders within 10% of each other: keep FIFO; a clear win: deviate."""
        pi = plan_input([("waiting", [("m", 1.0, 1.0)]), ("waiting", [("s", 0.5, 1.0)])])   # 5.5 vs 4 => 27% saving
        order, dev = choose(pi, fifo(pi), greedy_order(pi))
        self.assertTrue(dev)
        self.assertEqual(order[0], "s")
        pi = plan_input([("waiting", [("s2", 0.5, 1.0)]), ("waiting", [("s", 0.4, 1.0)])])  # 4 vs 3.9
        order, dev = choose(pi, fifo(pi), greedy_order(pi))
        self.assertFalse(dev)
        self.assertEqual(order, fifo(pi))

    def test_budget_fallback(self):
        chains = [("waiting", [(m, 1.0, 1.0) for m in ["b", "s", "m", "s2", "b", "m"]]) for _ in range(10)]
        pi = plan_input(chains)
        info = {}
        self.assertEqual(dp_order(pi, budget=3, info=info), greedy_order(pi))
        self.assertTrue(info["fallback"])
        info = {}
        dp_order(pi, info=info)
        self.assertFalse(info["fallback"])


if __name__ == "__main__":
    unittest.main()
