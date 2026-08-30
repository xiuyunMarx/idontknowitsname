import unittest

from serve.predict import predicted_chain


class FakeProgram:
    """A hover-like loop: plan -> retrieve -> reason -> assess -> {plan (p), verify (1-p)} -> synth."""

    def __init__(self, p_loop: float):
        self.callsites = {k: k for k in ("plan", "retrieve", "reason", "assess", "verify", "synth")}
        self.edges = {"plan": [("retrieve", 1.0)], "retrieve": [("reason", 1.0)], "reason": [("assess", 1.0)],
                      "assess": sorted([("plan", p_loop), ("verify", 1 - p_loop)], key=lambda t: -t[1]),
                      "verify": [("synth", 1.0)], "synth": []}
        self.unresolved_models = set()

    def branch_candidates(self, key):
        return self.edges[key]

    def model_of(self, site, fallback):
        return {"plan": "s", "retrieve": "m", "reason": "b", "assess": "s", "verify": "B", "synth": "s"}[site]


class Predict(unittest.TestCase):
    def test_one_extra_lap_at_even_odds(self):
        chain = [k for k, _ in predicted_chain(FakeProgram(0.5), "plan", "x", max_steps=20)]
        self.assertEqual(chain, ["retrieve", "reason", "assess", "plan", "retrieve", "reason", "assess", "verify", "synth"])

    def test_no_lap_when_loop_is_rare(self):
        chain = [k for k, _ in predicted_chain(FakeProgram(0.2), "plan", "x", max_steps=20)]
        self.assertEqual(chain, ["retrieve", "reason", "assess", "verify", "synth"])

    def test_three_laps_when_loop_is_likely(self):
        chain = [k for k, _ in predicted_chain(FakeProgram(0.75), "plan", "x", max_steps=40)]
        self.assertEqual(chain.count("plan"), 3)
        self.assertEqual(chain[-2:], ["verify", "synth"])

    def test_cap_and_models(self):
        chain = predicted_chain(FakeProgram(0.5), "plan", "x", max_steps=3)
        self.assertEqual(chain, [("retrieve", "m"), ("reason", "b"), ("assess", "s")])


if __name__ == "__main__":
    unittest.main()
