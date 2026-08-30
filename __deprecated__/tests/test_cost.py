import unittest

from serve.cost import CREATE_PENALTY_S, PRIOR_SECS_PER_B, CostModel, param_count

MODELS = {"Qwen/Qwen2.5-0.5B-Instruct": 0.5, "Qwen/Qwen3-0.6B": 0.6, "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
          "Qwen/Qwen3-1.7B": 1.7, "Qwen/Qwen2.5-3B-Instruct": 3.0, "Qwen/Qwen3-4B": 4.0}


class ParamCount(unittest.TestCase):
    def test_all_six(self):
        for name, b in MODELS.items():
            self.assertEqual(param_count(name), b, name)

    def test_unparseable(self):
        self.assertEqual(param_count("org/mystery-model"), 1.0)


class Load(unittest.TestCase):
    def test_prior_scales_with_size(self):
        c = CostModel()
        self.assertEqual(c.load_s("Qwen/Qwen3-4B", "gpu"), 0.0)
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-4B", "host"), 4.0 * PRIOR_SECS_PER_B["host"])
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-4B", "ssd"), 4.0 * PRIOR_SECS_PER_B["ssd"])

    def test_observation_recalibrates_prior(self):
        c = CostModel()
        c.observe_load("Qwen/Qwen3-4B", "host", 8.0)          # 2 s/B observed at host tier
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-4B", "host"), 8.0)
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-0.6B", "host"), 0.6 * 2.0)   # unseen model, observed tier
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-0.6B", "ssd"), 0.6 * PRIOR_SECS_PER_B["ssd"])

    def test_ema(self):
        c = CostModel(alpha=0.5)
        c.observe_load("m-1B", "host", 2.0)
        c.observe_load("m-1B", "host", 4.0)
        self.assertAlmostEqual(c.load_s("m-1B", "host"), 3.0)

    def test_create_penalty_for_unknown_model(self):
        c = CostModel()
        c.register("Qwen/Qwen3-4B")
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-1.7B", "host") - 1.7 * PRIOR_SECS_PER_B["host"], CREATE_PENALTY_S)
        self.assertAlmostEqual(c.load_s("Qwen/Qwen3-4B", "host"), 4.0 * PRIOR_SECS_PER_B["host"])


class Exec(unittest.TestCase):
    def test_fallback_chain(self):
        c = CostModel()
        self.assertAlmostEqual(c.exec_s("k", "Qwen/Qwen3-4B"), 0.5 + 0.4 * 4.0)
        c.observe_exec("other", "Qwen/Qwen3-4B", 3.0)
        self.assertAlmostEqual(c.exec_s("k", "Qwen/Qwen3-4B"), 3.0)
        c.observe_exec("k", "Qwen/Qwen3-4B", 1.0)
        self.assertAlmostEqual(c.exec_s("k", "Qwen/Qwen3-4B"), 1.0)


class Turns(unittest.TestCase):
    def test_tool_loop_turns_multiply(self):
        c = CostModel(alpha=1.0)
        c.observe_exec("k", "m-1B", 1.0, first_turn=True)
        c.observe_exec("k", "m-1B", 1.0, first_turn=False)
        c.observe_exec("k", "m-1B", 1.0, first_turn=False)
        self.assertAlmostEqual(c.exec_s("k", "m-1B"), 3.0)      # call in progress: 3 turns so far
        c.observe_exec("k", "m-1B", 2.0, first_turn=True)        # next call: previous one counted (3 turns)
        self.assertAlmostEqual(c.exec_s("k", "m-1B"), 6.0)
        self.assertEqual(c.export()["turns"], {"k": 3.0})


class Rate(unittest.TestCase):
    def test_rate_and_decay(self):
        c = CostModel()
        self.assertEqual(c.rate("m", 10.0), 0.0)
        for t in (0.0, 1.0, 2.0, 3.0, 4.0):
            c.observe_arrival("m", t)
        self.assertAlmostEqual(c.rate("m", 4.0), 1.0)
        self.assertLess(c.rate("m", 100.0), 0.06)          # nothing for a long time: rate decays


class Freeze(unittest.TestCase):
    def test_frozen_ignores_observations(self):
        c = CostModel(learn=False)
        c.observe_load("m-1B", "host", 100.0)
        c.observe_exec("k", "m-1B", 100.0)
        c.observe_arrival("m-1B", 0.0)
        self.assertAlmostEqual(c.load_s("m-1B", "host"), PRIOR_SECS_PER_B["host"])
        self.assertAlmostEqual(c.exec_s("k", "m-1B"), 0.9)
        self.assertEqual(c.rate("m-1B", 1.0), 0.0)

    def test_export(self):
        c = CostModel()
        c.observe_load("m-1B", "ssd", 5.0)
        c.observe_arrival("m-1B", 0.0); c.observe_arrival("m-1B", 2.0)
        e = c.export(now=2.0)
        self.assertEqual(e["load_s"], {"m-1B|ssd": 5.0})
        self.assertEqual(e["arrivals"], {"m-1B": 2})
        self.assertAlmostEqual(e["rate"]["m-1B"], 0.5)


if __name__ == "__main__":
    unittest.main()
