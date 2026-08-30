"""Parsing of every wire shape, session folding, workflow statistics, and offline == online."""
import json
import os
import tempfile
import unittest

from tests import wire
from trace_extractor.parser import SessionBuilder, build_programs, is_continuation, load_trace, parse_request
from trace_extractor.primitives import CallsiteType, END, Program, TraceRecord, VisitByCallsite


def _records():
    rows = wire.trace_rows()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        path = f.name
    wire.write_jsonl(path, rows)
    try:
        return rows, load_trace(path)
    finally:
        os.unlink(path)


class Parsing(unittest.TestCase):
    def setUp(self):
        self.rows, self.records = _records()

    def test_visit(self):
        site, call = parse_request(self.records[0])
        self.assertIsInstance(site, VisitByCallsite)
        self.assertEqual(site.select, "exactly one")
        self.assertEqual(site.candidates, ["BillingAgent", "ShippingAgent", "Orphan"])
        self.assertEqual(len(site.edges), 2)
        self.assertTrue(site.edges[0].startswith("the class of request"))
        self.assertTrue(call.walker.startswith("the triage walker"))
        self.assertTrue(call.here.startswith("the front desk"))

    def test_plain_by_llm(self):
        site, call = parse_request(self.records[4])  # summarize
        self.assertIs(site.kind, CallsiteType.BYLLM)
        self.assertEqual(site.label, "Agent.summarize")
        self.assertEqual([p[0] for p in site.params], ["request", "desk", "answer"])
        self.assertEqual(site.params[1], ("desk", "str", "the desk that handled the request"))
        self.assertEqual(site.params[0], ("request", "str", ""))
        self.assertEqual(site.return_type, "str")
        self.assertTrue(site.sem.startswith("Compress"))
        self.assertEqual(call.bindings, {"request": repr("The tracking has not moved in nine days."),
                                         "desk": "'shipping'", "answer": "'reshipped'"})
        self.assertTrue(call.self_view.startswith("Agent(request=") and call.self_view.endswith("summary='')"))
        self.assertNotIn("Schema requirements", site.context_desc)

    def test_structured_and_typed_retry(self):
        site, call = parse_request(self.records[6])
        self.assertEqual(site.label, "ResearchSupervisor.decompose_task")
        self.assertEqual(site.return_type, "list[str]")
        self.assertEqual(call.bindings, {"task": "'why cold starts'", "web": "''"})
        retry_site, retry_call = parse_request(self.records[7])
        self.assertEqual(retry_site.key, site.key)               # the feedback turn is not a new callsite
        self.assertEqual(retry_call.bindings, call.bindings)
        self.assertTrue(is_continuation(self.records[6].request, self.records[7].request))

    def test_react_native_and_text(self):
        site, call = parse_request(self.records[9])   # native turn 2
        self.assertIs(site.kind, CallsiteType.TOOL)
        self.assertEqual(site.tools, ["finish_tool", "open_page", "web_search"])
        self.assertEqual(call.turn, 1)
        self.assertTrue(is_continuation(self.records[8].request, self.records[9].request))
        self.assertFalse(is_continuation(self.records[9].request, self.records[8].request))
        site, call = parse_request(self.records[13])  # text-protocol turn 2
        self.assertIs(site.kind, CallsiteType.TOOL)
        self.assertEqual(site.tools, ["finish_tool", "open_page", "web_search"])
        self.assertEqual(call.turn, 1)
        self.assertEqual(call.bindings["task"], "'why cold starts'")
        final_site, final_call = parse_request(self.records[14])  # forced final pass: no tool block
        self.assertEqual(final_site.key, site.key)
        self.assertEqual(final_call.turn, 2)
        self.assertTrue(is_continuation(self.records[13].request, self.records[14].request))

    def test_same_callsite_same_key_across_sessions(self):
        a, _ = parse_request(self.records[4])
        b, _ = parse_request(self.records[5])
        self.assertEqual(a.key, b.key)


class Workflow(unittest.TestCase):
    def setUp(self):
        self.rows, self.records = _records()
        self.programs = build_programs(self.records)

    def test_programs_and_sequences(self):
        self.assertEqual(set(self.programs), {"triage", "deep_research"})
        tri = self.programs["triage"]
        self.assertEqual(len(tri.callsites), 4)
        seqs = {sid: [tri.label(k) for k in s.sequence] for sid, s in tri.sessions.items()}
        self.assertEqual(seqs["triage:1"][1:], ["ShippingAgent.resolve_shipping", "Agent.summarize"])
        self.assertEqual(seqs["triage:2"][1:], ["BillingAgent.resolve_billing", "Agent.summarize"])

    def test_branches_gaps_end(self):
        tri = self.programs["triage"]
        visit = tri.sessions["triage:1"].sequence[0]
        cands = tri.branch_candidates(visit)
        self.assertEqual(len(cands), 2)
        for _, p in cands:
            self.assertAlmostEqual(p, 0.5)
        self.assertEqual(tri.end_prob(tri.sessions["triage:1"].sequence[-1]), 1.0)
        self.assertEqual(tri.entry[visit], 2)
        ship = tri.sessions["triage:1"].sequence[1]
        self.assertAlmostEqual(tri.gap_s(visit, ship), 0.8)

    def test_react_and_retry_folding_and_loop(self):
        dr = self.programs["deep_research"]
        s7 = dr.sessions["deep_research:7"]
        self.assertEqual([dr.label(k) for k in s7.sequence],
                         ["ResearchSupervisor.decompose_task", "WebSearchAgent.web_research",
                          "ResearchSupervisor.decompose_task", "WebSearchAgent.web_research"])
        self.assertEqual([len(c.turns) for c in s7.calls], [2, 3, 1, 3])   # retry folded, 3 native turns, 2 text turns + forced final
        self.assertEqual(len(dr.callsites), 2)   # native/text tool variants and the forced final pass share one key
        self.assertAlmostEqual(s7.calls[1].exec_s, 1.2)
        self.assertEqual([round(g, 2) for g in s7.calls[1].tool_gaps], [0.6, 0.6])
        dk, rk = s7.sequence[0], s7.sequence[1]
        self.assertEqual(dr.branch_candidates(dk)[0][0], rk)
        self.assertEqual(dr.succ[rk][dk], 1)
        pred = dr.predict(dk)
        self.assertEqual(pred[0][0], rk)
        self.assertAlmostEqual(pred[0][2], 1.0)   # gap: research arrives 1.0 s after decompose finishes

    def test_persistence_round_trip(self):
        tri = self.programs["triage"]
        back = Program.from_dict(json.loads(json.dumps(tri.to_dict())))
        visit = tri.sessions["triage:1"].sequence[0]
        self.assertEqual(back.branch_candidates(visit), tri.branch_candidates(visit))
        self.assertEqual(back.callsites[visit].candidates, tri.callsites[visit].candidates)
        ship = tri.sessions["triage:1"].sequence[1]
        self.assertEqual(back.callsites[ship].params, tri.callsites[ship].params)

    def test_online_equals_offline(self):
        """Feeding requests one at a time through SessionBuilder + observe_call + close_session
        gives the same statistics as build_programs over the whole trace."""
        online = {}
        builders = {}
        for rec in sorted(self.records, key=lambda r: r.t_arrive):
            prog = online.setdefault(rec.program, Program(rec.program))
            b = builders.setdefault(rec.session, SessionBuilder(rec.session, rec.program))
            site, call = parse_request(rec)
            prog.add_callsite(site)
            before = list(b.session.calls)
            inst, new = b.step(call)
            if new and before:
                prog.observe_call(before[-2] if len(before) > 1 else None, before[-1], inst)
        for sid, b in builders.items():
            online[b.session.program].close_session(b.session)
        for name, prog in self.programs.items():
            a, o = prog.to_dict(), online[name].to_dict()
            a.pop("callsites"), o.pop("callsites")
            self.assertEqual(a, o, name)


if __name__ == "__main__":
    unittest.main()
