"""Compile the real agents, without executing their entrypoints or calling LLMs.

Keep prompt layout and loop/lifetime contracts stable when changing KV policy.
"""
import contextlib
import io
import unittest
from pathlib import Path

from benchmark.mixed_workload import PROGRAMS
from static_analysis.agent_launcher import analyze_program
from static_analysis.primitives import Heterogeneity, program_from_dict


class ApplicationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.programs = {}
        apps = Path(__file__).resolve().parents[1] / "benchmark" / "applications"
        for name in ("fact_check", "coding_agent", "doc_analysis", "BFCL_agent"):
            with contextlib.redirect_stderr(io.StringIO()):
                payload = analyze_program(str(apps / (name + ".jac")))
            p = program_from_dict(payload["program"])
            p.decide_layouts()
            p.decide_loops()
            cls.programs[name] = p

    def sites(self, name):
        return {key.signature: t for key, t in self.programs[name].sites.items()}

    def test_fact_check_keeps_header_first_in_agent_and_benchmark(self):
        self.assertTrue(PROGRAMS["fact_check"]["no_header_last"])
        expected = {
            "ClaimAnalyzer.decompose_claim": ["claim", "evidences", "maximum_questions"],
            "EvidenceScout.assess_evidence": ["claim", "evidences", "search_results", "question"],
            "EvidenceAnalyzer.verify_claim": ["claim", "evidences"],
        }
        for name, t in self.sites("fact_check").items():
            self.assertTrue(t.no_header_last)
            self.assertFalse(t.header_last)
            self.assertEqual(t.served_order(), expected[name])
            values = {b.name: b.literal or "'value'" for b in t.bindings}
            self.assertTrue(t.render(values).startswith(t.header + "\n"))

    def test_coding_keeps_shared_prefix_and_nested_loops(self):
        p = self.programs["coding_agent"]
        ts = self.sites("coding_agent")
        for name in ("Planner.plan", "Coder.implement", "Analyzer.diagnose"):
            self.assertEqual(ts[name].served_order()[:3], ["module", "spec", "tries"])
            self.assertTrue(ts[name].header_last)
        self.assertEqual(len(p.loops), 2)
        reset = p.edges[(ts["Coder.implement"].key, ts["Planner.plan"].key)]
        self.assertTrue({"spec", "tries"}.issubset(reset.invalidates))
        retry = p.edges[(ts["Coder.implement"].key, ts["Analyzer.diagnose"].key)]
        self.assertNotIn("tries", retry.invalidates)

    def test_document_stays_leading_until_its_last_consumer(self):
        # question and document are the same static class; the declaration puts
        # question first, the observed sizes put the document first at every site
        p = self.programs["doc_analysis"]
        ts = self.sites("doc_analysis")
        self.assertEqual(ts["Extractor.extract"].served_order(), ["question", "document"])
        moved = p.observe_sizes({"question": 60, "document": 10000})
        self.assertEqual({k.signature for k in moved}, {"Extractor.extract", "Analyst.compute", "Auditor.check"})
        for name in ("Extractor.extract", "Analyst.compute", "Auditor.check"):
            t = ts[name]
            self.assertEqual(t.served_order()[:2], ["document", "question"])
            self.assertTrue(t.header_last)
        self.assertEqual(ts["Analyst.compute"].served_order(), ["document", "question", "facts"])
        self.assertEqual(ts["Auditor.check"].served_order(), ["document", "question", "facts", "analysis"])
        self.assertNotIn("document", ts["Writer.answer"].served_order())
        # sizes are kept from the first value: a later, different count changes nothing
        self.assertEqual(p.observe_sizes({"question": 20000, "facts": 300}), [])

    def test_bfcl_keeps_append_only_history_and_self_loop(self):
        p = self.programs["BFCL_agent"]
        t = self.sites("BFCL_agent")["Agent.step"]
        self.assertEqual(t.served_order(), ["tools", "question", "history", "observation"])
        self.assertEqual(t.binding("observation").heterogeneity, Heterogeneity.VOLATILE)
        self.assertIn("observation", p.edges[(t.key, t.key)].invalidates)
        self.assertEqual(t.binding("history").heterogeneity, Heterogeneity.EXTEND)
        self.assertEqual(t.binding("tools").heterogeneity, Heterogeneity.CONST)
        self.assertIn((t.key, t.key), p.edges)
        self.assertNotIn("history", p.edges[(t.key, t.key)].invalidates)
        self.assertFalse(t.header_last)
        self.assertTrue(t.fixed_head().startswith(t.header + "\n"))
        self.assertIn(t.binding("tools").literal, t.fixed_head())


if __name__ == "__main__":
    unittest.main()
