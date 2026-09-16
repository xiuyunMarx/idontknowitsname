"""Loop nesting of the callsite graph, iteration counters, and the branch
statistics keyed by them."""
import unittest

from static_analysis.primitives import CallSiteID, Program, PromptTemplate


def site(name: str) -> CallSiteID:
    return CallSiteID(name, 1, "/agent.jac")


def coding() -> Program:
    """Planner -> Coder -> {Analyzer -> Coder | Planner | Committer}: a retry loop
    headed by Coder inside a per-function loop headed by Planner."""
    P, C, A, M = site("Planner"), site("Coder"), site("Analyzer"), site("Committer")
    p = Program(P)
    for k in (P, C, A, M):
        p.add_site(PromptTemplate(key=k, ability=k.signature))
    p.add_edge(P, C)
    p.add_edge(C, A)
    p.add_edge(A, C)
    p.add_edge(C, P)
    p.add_edge(C, M)
    p.exits = frozenset({M})
    p.decide_loops()
    return p


def fact() -> Program:
    """ClaimAnalyzer -> Scout (fan-out: Scout -> Scout) -> EvidenceAnalyzer -> ClaimAnalyzer."""
    CA, S, EA = site("ClaimAnalyzer"), site("Scout"), site("EvidenceAnalyzer")
    p = Program(CA)
    for k in (CA, S, EA):
        p.add_site(PromptTemplate(key=k, ability=k.signature))
    p.add_edge(CA, S)
    p.add_edge(S, S)
    p.add_edge(S, EA)
    p.add_edge(EA, CA)
    p.exits = frozenset({EA})
    p.decide_loops()
    return p


class LoopNestingTests(unittest.TestCase):
    def test_coding_graph_has_a_retry_loop_inside_the_function_loop(self) -> None:
        p = coding()
        P, C, A, M = (site(n) for n in ("Planner", "Coder", "Analyzer", "Committer"))
        self.assertEqual(len(p.loops), 2)
        outer, inner = p.loops
        self.assertEqual((outer.head, outer.body, outer.parent), (P, frozenset({P, C, A}), None))
        self.assertEqual((inner.head, inner.body, inner.parent), (C, frozenset({C, A}), 0))
        self.assertEqual(p.chain[P], (0,))
        self.assertEqual(p.chain[C], (0, 1))
        self.assertEqual(p.chain[A], (0, 1))
        self.assertEqual(p.chain[M], ())

    def test_fan_out_self_edge_is_a_loop_of_its_own(self) -> None:
        p = fact()
        CA, S, EA = (site(n) for n in ("ClaimAnalyzer", "Scout", "EvidenceAnalyzer"))
        heads = {lp.head: lp for lp in p.loops}
        self.assertEqual(set(heads), {CA, S})
        self.assertEqual(heads[CA].body, frozenset({CA, S, EA}))
        self.assertEqual(heads[S].body, frozenset({S}))
        self.assertEqual(heads[S].parent, p.loops.index(heads[CA]))
        self.assertEqual(len(p.chain[S]), 2)
        self.assertEqual(len(p.chain[EA]), 1)

    def test_acyclic_graph_has_no_loops(self) -> None:
        A, B = site("A"), site("B")
        p = Program(A)
        p.add_site(PromptTemplate(key=A))
        p.add_site(PromptTemplate(key=B))
        p.add_edge(A, B)
        p.decide_loops()
        self.assertEqual(p.loops, [])
        self.assertEqual(p.chain, {A: (), B: ()})


class CounterTests(unittest.TestCase):
    def contexts(self, p: Program, seq):
        ctr, prev, out = {}, None, []
        for k in seq:
            ctr = p.step_counters(ctr, prev, k)
            out.append(p.context(k, ctr))
            prev = k
        return out

    def test_coding_session_counts_functions_and_retries(self) -> None:
        p = coding()
        P, C, A, M = (site(n) for n in ("Planner", "Coder", "Analyzer", "Committer"))
        seq = [P, C, A, C, A, C, P, C, M]
        self.assertEqual(self.contexts(p, seq), [
            (1,), (1, 1), (1, 1), (1, 2), (1, 2), (1, 3),   # function 1: three tries
            (2,), (2, 1),                                    # function 2: the retry counter restarts
            ()])                                             # Committer is outside every loop

    def test_fact_session_counts_rounds_and_scouts(self) -> None:
        p = fact()
        CA, S, EA = (site(n) for n in ("ClaimAnalyzer", "Scout", "EvidenceAnalyzer"))
        seq = [CA, S, S, EA, CA, S, S, S, EA]
        self.assertEqual(self.contexts(p, seq), [
            (1,), (1, 1), (1, 2), (1,),
            (2,), (2, 1), (2, 2), (2, 3), (2,)])

    def test_context_is_capped(self) -> None:
        p = fact()
        S = site("Scout")
        ctr = p.entry_counters(S)
        for _ in range(20):
            ctr = p.step_counters(ctr, S, S)
        self.assertEqual(p.context(S, ctr), (1, 8))


class BranchProbabilityTests(unittest.TestCase):
    def feed(self, p: Program, sessions) -> None:
        for seq in sessions:
            ctr, prev = {}, None
            for k in seq:
                if prev is not None:
                    p.observe_branch(prev, p.context(prev, ctr), k)
                ctr = p.step_counters(ctr, prev, k)
                prev = k
            p.observe_branch(prev, p.context(prev, ctr), None)

    def test_cold_start_is_uniform_over_static_successors(self) -> None:
        p = coding()
        C = site("Coder")
        probs = dict(p.branch_probs(C, p.entry_counters(C)))
        self.assertEqual(len(probs), 3)
        for v in probs.values():
            self.assertAlmostEqual(v, 1 / 3)

    def test_exit_probability_depends_on_the_iteration_count(self) -> None:
        p = coding()
        P, C, A, M = (site(n) for n in ("Planner", "Coder", "Analyzer", "Committer"))
        # every function needs three tries and then moves on; the last one commits
        self.feed(p, [[P, C, A, C, A, C, P, C, A, C, A, C, M]] * 6)
        c1 = dict(p.branch_probs(C, {0: 1, 1: 1}))
        c3 = dict(p.branch_probs(C, {0: 1, 1: 3}))
        self.assertGreater(c1[A], 0.8)
        self.assertLess(c1[P], 0.1)
        self.assertGreater(c3[P], 0.8)
        self.assertLess(c3[A], 0.1)
        # the outer count separates "move on" from "commit"
        f2 = dict(p.branch_probs(C, {0: 2, 1: 3}))
        self.assertGreater(f2[M], 0.8)
        self.assertGreater(c3[P], f2[P])

    def test_unseen_context_backs_off_to_the_pooled_counts(self) -> None:
        p = coding()
        P, C, A, M = (site(n) for n in ("Planner", "Coder", "Analyzer", "Committer"))
        self.feed(p, [[P, C, P, C, M]] * 4)
        pooled = dict(p.branch_probs(C))
        unseen = dict(p.branch_probs(C, {0: 7, 1: 5}))
        for k in (P, A, M):
            self.assertAlmostEqual(pooled[k], unseen[k])

    def test_session_end_consumes_mass(self) -> None:
        p = fact()
        CA, S, EA = (site(n) for n in ("ClaimAnalyzer", "Scout", "EvidenceAnalyzer"))
        two_rounds = [CA, S, S, EA, CA, S, S, EA]
        self.feed(p, [two_rounds] * 5)
        r1 = dict(p.branch_probs(EA, {0: 1}))
        r2 = dict(p.branch_probs(EA, {0: 2}))
        self.assertGreater(r1[CA], 0.8)
        self.assertLess(r2[CA], 0.2)          # the end took the mass; no successor for it
        s1 = dict(p.branch_probs(S, {0: 1, 1: 1}))
        s2 = dict(p.branch_probs(S, {0: 1, 1: 2}))
        self.assertGreater(s1[S], 0.8)
        self.assertGreater(s2[EA], 0.8)

    def test_prediction_walks_the_counters_along_the_path(self) -> None:
        p = coding()
        P, C, A, M = (site(n) for n in ("Planner", "Coder", "Analyzer", "Committer"))
        self.feed(p, [[P, C, A, C, A, C, P, C, A, C, A, C, M]] * 6)
        # from the third try of function 1, Planner is next and Analyzer is not a
        # direct successor any more (it is reached only through the next function)
        direct = {c.key: c for c in p.predict_tree(C, {0: 1, 1: 3}) if c.path == (C, c.key)}
        self.assertIn(P, direct)
        self.assertNotIn(A, direct)
        # from function 2's third try the direct successor is Committer
        direct = {c.key: c for c in p.predict_tree(C, {0: 2, 1: 3}) if c.path == (C, c.key)}
        self.assertIn(M, direct)
        self.assertNotIn(A, direct)
        self.assertGreater(direct[M].p, 0.8)
        # without counters the call still works (pooled statistics)
        self.assertTrue(p.predict_tree(C))


if __name__ == "__main__":
    unittest.main()


class WiringTests(unittest.TestCase):
    def test_end_probability_is_learned_per_context(self) -> None:
        p = fact()
        CA, S, EA = (site(n) for n in ("ClaimAnalyzer", "Scout", "EvidenceAnalyzer"))
        # cold start: an exit site with one successor splits the prior with END; a
        # non-exit site has no END mass at all
        self.assertAlmostEqual(p.end_prob(EA, {0: 1}), 0.5)
        self.assertEqual(p.end_prob(S, {0: 1, 1: 1}), 0.0)
        BranchProbabilityTests.feed(self, p, [[CA, S, S, EA, CA, S, S, EA]] * 5)
        self.assertLess(p.end_prob(EA, {0: 1}), 0.2)
        self.assertGreater(p.end_prob(EA, {0: 2}), 0.8)

    def test_reregistering_the_same_analysis_keeps_the_statistics(self) -> None:
        from server.server import Controller
        from static_analysis.primitives import program_to_dict
        p = coding()
        P, C = site("Planner"), site("Coder")
        body = {"file": "/agent.jac", "program": program_to_dict(p)}
        ctrl = Controller.__new__(Controller)
        ctrl.enable_relayout = False
        ctrl.programs, ctrl._by_site, ctrl._resets, ctrl._registered = {}, {}, {}, {}
        ctrl.register(body)
        prog = ctrl.programs["/agent.jac"]
        prog.observe_branch(P, prog.context(P, prog.entry_counters(P)), C)
        self.assertEqual(ctrl.register(body).get("unchanged"), True)
        self.assertIs(ctrl.programs["/agent.jac"], prog)
        self.assertTrue(prog.ctx_counts)
        # a changed analysis replaces the program
        p.add_edge(P, site("Committer"))
        ctrl.register({"file": "/agent.jac", "program": program_to_dict(p)})
        self.assertIsNot(ctrl.programs["/agent.jac"], prog)
