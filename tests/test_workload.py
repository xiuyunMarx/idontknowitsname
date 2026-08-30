import importlib.util
import json
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gen_schedule", os.path.join(ROOT, "benchmark/workload/gen_schedule.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

Q = "Qwen/"
CHAINS = {  # program -> per-case chains of (model, exec_s); mirrors the static call graphs
    "benchmark/applications/hover.jac": [[("Qwen2.5-1.5B-Instruct", 1.0), ("Qwen3-1.7B", 1.5), ("Qwen2.5-3B-Instruct", 2.0),
                                          ("Qwen2.5-1.5B-Instruct", 1.0), ("Qwen3-4B", 3.0), ("Qwen2.5-0.5B-Instruct", 0.5)]] * 3,
    "benchmark/applications/rag_qa.jac": [[("Qwen2.5-0.5B-Instruct", 0.5), ("Qwen3-0.6B", 0.5), ("Qwen3-4B", 2.5)]] * 4,
    "benchmark/applications/text2sql.jac": [[("Qwen2.5-3B-Instruct", 2.0), ("Qwen3-1.7B", 1.5), ("Qwen2.5-3B-Instruct", 2.0),
                                             ("Qwen2.5-0.5B-Instruct", 0.5)]] * 4,
}
ENV = {"hover.jac": "HOVER_CLAIM_INDEX", "rag_qa.jac": "RAG_INDEX", "text2sql.jac": "SQL_TASK_INDEX"}


def write_profiles(path):
    cases = {}
    for p, cs in CHAINS.items():
        name = os.path.basename(p)
        for i, ch in enumerate(cs):
            cases[f"{name}:{i}"] = {"program": p, "env": ENV[name], "case_index": i, "rc": 0, "wall_s": 1.0,
                                    "chain": [{"callsite": f"c{j}", "kind": "call", "model": Q + m, "exec_s": e,
                                               "queue_s": 0, "gap_s": 0, "output_tokens": 1} for j, (m, e) in enumerate(ch)]}
    with open(path, "w") as f:
        json.dump({"meta": {}, "cost": {"load_s": {Q + "Qwen3-4B|ssd": 20.0, Q + "Qwen3-4B|host": 8.0}}, "cases": cases}, f)


class Generator(unittest.TestCase):
    W = {"hover.jac": 1, "rag_qa.jac": 1, "text2sql.jac": 1}

    def test_prefix_stability(self):
        with tempfile.TemporaryDirectory() as d:
            prof = os.path.join(d, "profiles.json")
            write_profiles(prof)
            for pattern, kw in [("poisson", {}), ("burst", {}), ("adversarial", {"profiles": prof, "candidates": 20})]:
                kw = dict(kw, oracle=True) if pattern == "adversarial" else kw
                small = gen.generate(1, pattern, 6, self.W, burst_size=3, **kw)["entries"]
                big = gen.generate(1, pattern, 12, self.W, burst_size=3, **kw)["entries"]
                self.assertEqual(small, big[:6], pattern)
                self.assertNotEqual(gen.generate(2, pattern, 6, self.W, burst_size=3, **kw)["entries"], small, pattern)

    def test_adversarial_bursts_favour_dp(self):
        with tempfile.TemporaryDirectory() as d:
            prof = os.path.join(d, "profiles.json")
            write_profiles(prof)
            sched = gen.generate(3, "adversarial", 9, self.W, burst_size=3, profiles=prof, candidates=40, oracle=True)
            self.assertEqual(len(sched["entries"]), 9)
            for b in sched["meta"]["bursts"]:
                self.assertLessEqual(b["sim_wait"]["dp"], b["sim_wait"]["greedy"], b)
            self.assertEqual(sched["meta"]["profiles_sha1"], gen.Profiles(prof, oracle=True).sha1)
            offs = [e["start_offset_s"] for e in sched["entries"]]
            self.assertTrue(all(25.0 <= o <= 26.0 for o in offs[3:6]))


if __name__ == "__main__":
    unittest.main()
