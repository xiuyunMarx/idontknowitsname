import asyncio
import time
import unittest
from types import SimpleNamespace

from server.kv_planner import Job, KVPlanner
from server.server import Controller, LiveSession
from static_analysis.primitives import (
    Binding,
    CallInstance,
    CallSiteID,
    Heterogeneity,
    Program,
    PromptTemplate,
    program_from_dict,
    program_to_dict,
)


def site(name: str, line: int) -> CallSiteID:
    return CallSiteID(name, line, "/agent.jac")


def template(key: CallSiteID, ability: str, bindings=None, system: str = "sys") -> PromptTemplate:
    return PromptTemplate(
        key=key,
        ability=ability,
        system_prompt=system,
        bindings=list(bindings or []),
        order=[b.name for b in (bindings or [])],
        header_last=True,
    )


class StaticFlowTests(unittest.TestCase):
    def test_prediction_counts_only_the_next_visit_to_each_site(self) -> None:
        a, b, end = site("A", 1), site("B", 2), site("End", 3)
        p = Program(a)
        for key in (a, b, end):
            p.add_site(template(key, key.signature))
        p.add_edge(a, b)
        p.add_edge(a, end)
        p.add_edge(b, b)
        p.add_edge(b, end)
        p.decide_loops()

        predicted = {c.key: c for c in p.predict_tree(a)}
        self.assertEqual(predicted[b].paths, [(a, b)])
        self.assertAlmostEqual(predicted[b].p, 0.5)
        # Keep walking loops to find the first arrival at downstream sites.
        self.assertIn((a, b, b, end), predicted[end].paths)

        # The current call does not count as a future visit: predict one return.
        revisit = next(c for c in p.predict_tree(b) if c.key == b)
        self.assertEqual(revisit.paths, [(b, b)])
        self.assertAlmostEqual(revisit.p, 0.5)

    def test_first_visit_keeps_alternative_paths_of_different_lengths(self) -> None:
        a, b, c = site("A", 1), site("B", 2), site("C", 3)
        value = Binding("x", Heterogeneity.COPY, field="x", label="x = ")
        p = Program(a)
        for key in (a, b, c):
            p.add_site(template(key, key.signature, [value]))
        p.add_edge(a, c)
        p.add_edge(a, b)
        p.add_edge(b, c, frozenset({"x"}), {}, frozenset({"x"}))

        predicted = next(call for call in p.predict_tree(a) if call.key == c)
        self.assertEqual(set(predicted.paths), {(a, c), (a, b, c)})
        self.assertAlmostEqual(predicted.p, 1.0)
        effect = Controller._paths_effect(p, predicted.paths)
        self.assertIn("x", effect.invalidates)
        ctrl = Controller.__new__(Controller)
        sess = LiveSession("s", program=p, open=CallInstance(p.sites[a], {"x": "'old'"}))
        self.assertEqual(ctrl._resolve(sess, p.sites[c], effect), ("", False))

    def test_later_iterations_do_not_truncate_the_next_coder_job(self) -> None:
        planner, coder = site("Planner", 1), site("Coder", 2)
        bindings = [
            Binding("module", Heterogeneity.EXTEND, field="module", label="module = "),
            Binding("spec", Heterogeneity.COPY, field="spec", label="spec = "),
        ]
        p = Program(planner)
        source = p.add_site(template(planner, "Planner", bindings))
        target = p.add_site(template(coder, "Coder", bindings))
        p.add_edge(planner, coder)
        p.add_edge(coder, planner, frozenset({"module", "spec"}), {}, frozenset({"spec"}))
        p.decide_loops()
        values = {"module": repr("accepted code\n"), "spec": repr("current specification")}
        sess = LiveSession("s", program=p, open=CallInstance(source, values))
        ctrl = Controller.__new__(Controller)
        ctrl.engine = SimpleNamespace(_stride=8, cost=lambda ids: (len(ids), 0))
        rendered = []

        def tokenize(tpl, text, complete):
            rendered.append((text, complete))
            return list(text.encode())

        ctrl._planned_tokens = tokenize
        ctrl._prefix_tokens = lambda tpl: []
        predicted = next(c for c in p.predict_tree(planner) if c.key == coder)
        jobs = ctrl._plan_call(sess, predicted, time.monotonic())
        expected = target.render(values)
        self.assertEqual(rendered, [(expected, True)])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].toks, list(expected.encode()))

    def test_cfg_path_overrides_must_agree(self) -> None:
        a, b = site("A", 1), site("B", 2)
        x1 = Binding("x", Heterogeneity.CONST, literal="1", field="x")
        x2 = Binding("x", Heterogeneity.CONST, literal="2", field="x")

        p = Program(a)
        p.add_edge(a, b, frozenset({"x"}), {"x": x1}, frozenset({"x"}))
        p.add_edge(a, b, frozenset({"x"}), {"x": x1}, frozenset({"x"}))
        self.assertEqual(p.edges[(a, b)].overrides["x"].literal, "1")

        p.add_edge(a, b, frozenset({"x"}), {"x": x2}, frozenset({"x"}))
        self.assertNotIn("x", p.edges[(a, b)].overrides)

    def test_prediction_retains_the_earliest_callsite_path(self) -> None:
        a, b, c = site("A", 1), site("B", 2), site("C", 3)
        p = Program(a)
        for key in (a, b, c):
            p.add_site(template(key, key.signature))
        p.add_edge(a, b)
        p.add_edge(b, c)

        predicted = {call.key: call for call in p.predict_tree(a)}
        self.assertEqual(predicted[b].path, (a, b))
        self.assertEqual(predicted[c].path, (a, b, c))
        self.assertEqual(predicted[c].paths, [(a, b, c)])

    def test_multihop_effect_carries_the_latest_known_origin(self) -> None:
        a, b, c = site("A", 1), site("B", 2), site("C", 3)
        bx = Binding("middle_x", Heterogeneity.CONST, literal="'new'", field="x")
        cx = Binding("target_x", Heterogeneity.COPY, field="x", label="target_x = ")
        p = Program(a)
        p.add_site(template(a, "A"))
        p.add_site(template(b, "B", [bx]))
        p.add_site(template(c, "C", [cx]))
        p.add_edge(a, b, frozenset({"x"}), {"middle_x": bx}, frozenset({"x"}))
        p.add_edge(b, c)

        effect = Controller._path_effect(p, (a, b, c))
        self.assertIsNotNone(effect)
        self.assertEqual(effect.overrides["target_x"].literal, "'new'")
        self.assertEqual(effect.invalidates, frozenset({"x"}))

    def test_alternative_prediction_paths_keep_only_common_origins(self) -> None:
        a, b, c, d = (site("A", 1), site("B", 2), site("C", 3), site("D", 4))
        dx = Binding("x", Heterogeneity.COPY, field="x", label="x = ")
        p = Program(a)
        for key in (a, b, c):
            p.add_site(template(key, key.signature))
        p.add_site(template(d, "D", [dx]))
        p.add_edge(a, b)
        p.add_edge(a, c)
        one = Binding("x", Heterogeneity.CONST, literal="1", field="x")
        two = Binding("x", Heterogeneity.CONST, literal="2", field="x")
        p.add_edge(b, d, frozenset({"x"}), {"x": one}, frozenset({"x"}))
        p.add_edge(c, d, frozenset({"x"}), {"x": two}, frozenset({"x"}))

        predicted = {call.key: call for call in p.predict_tree(a)}
        self.assertEqual(set(predicted[d].paths), {(a, b, d), (a, c, d)})
        effect = Controller._paths_effect(p, predicted[d].paths)
        self.assertIsNotNone(effect)
        self.assertNotIn("x", effect.overrides)
        self.assertEqual(effect.invalidates, frozenset({"x"}))

    def test_new_edge_fields_survive_wire_roundtrip(self) -> None:
        a, b = site("A", 1), site("B", 2)
        p = Program(a)
        p.add_site(template(a, "A"))
        p.add_site(template(b, "B"))
        p.add_edge(a, b, frozenset({"x", "history"}), {}, frozenset({"x"}))
        restored = program_from_dict(program_to_dict(p))
        edge = restored.edges[(a, b)]
        self.assertEqual(edge.writes, frozenset({"x", "history"}))
        self.assertEqual(edge.invalidates, frozenset({"x"}))


class RuntimeInvalidationTests(unittest.TestCase):
    def test_invalidation_starts_a_new_field_generation(self) -> None:
        a, b = site("A", 1), site("B", 2)
        old_binding = Binding("stable", Heterogeneity.COPY, field="stable", label="stable = ")
        dead_binding = Binding("x", Heterogeneity.COPY, field="x", label="x = ")
        old_tpl = template(a, "A", [old_binding, dead_binding])
        new_tpl = template(b, "B")
        p = Program(a)
        p.add_site(old_tpl)
        p.add_site(new_tpl)
        edge = p.add_edge(a, b, frozenset({"x"}), {}, frozenset({"x"}))
        old = CallInstance(old_tpl, {"stable": "'s'", "x": "'old'"}, served_ids=list(range(32)))
        sess = LiveSession("s", program=p, calls=[old])

        class Planner:
            def __init__(self):
                self.calls = []

            def invalidate(self, sid, ids, keep):
                self.calls.append((sid, ids, keep))

        ctrl = Controller.__new__(Controller)
        ctrl.planner = Planner()
        ctrl.programs = {"/agent.jac": p}
        ctrl._resets = {"/agent.jac": {}}
        ctrl._planned_tokens = lambda tpl, text, complete: list(range(len(text)))
        ctrl._prefix_tokens = lambda tpl: []

        ctrl._invalidate(sess, new_tpl, edge)
        self.assertEqual(sess.field_valid_from["x"], 1)
        self.assertEqual(len(ctrl.planner.calls), 1)

        current = CallInstance(new_tpl, {"x": "'new'"})
        sess.open = current
        lookup = Binding("x", Heterogeneity.COPY, field="x", scope="A")
        self.assertEqual(Controller._in_scope(sess, lookup), [current])

    def test_response_producer_belongs_to_the_new_generation(self) -> None:
        producer_key, consumer_key = site("Producer", 1), site("Consumer", 2)
        producer = template(producer_key, "Producer")
        producer.resp_spec = {"k": "obj", "name": "Answer", "fields": [["text", {"k": "prim", "t": "str"}, None]]}
        value = Binding("answer", Heterogeneity.RESP, source=(producer_key, ""),
                        field="answer", scope="Producer", label="answer = ")
        consumer = template(consumer_key, "Consumer", [value])
        p = Program(producer_key)
        p.add_site(producer)
        p.add_site(consumer)
        override = Binding("answer", Heterogeneity.RESP, source=(producer_key, ""), field="answer")
        edge = p.add_edge(producer_key, consumer_key, frozenset({"answer"}),
                          {"answer": override}, frozenset({"answer"}))
        produced = CallInstance(producer, {}, response='{"text":"new"}', served_ids=list(range(16)))
        sess = LiveSession("s", program=p, calls=[produced])

        class Planner:
            def invalidate(self, sid, ids, keep):
                pass

        ctrl = Controller.__new__(Controller)
        ctrl.planner = Planner()
        ctrl.programs = {"/agent.jac": p}
        ctrl._resets = {"/agent.jac": {}}
        ctrl._planned_tokens = lambda tpl, text, complete: []
        ctrl._prefix_tokens = lambda tpl: []
        ctrl._invalidate(sess, consumer, edge)

        self.assertEqual(sess.field_valid_from["answer"], 0)
        self.assertEqual(ctrl._value_of(sess, consumer, value, edge)[0], "Answer(text='new')")

        # A new producer call with no reply must mask the previous iteration.
        sess.calls.append(CallInstance(producer, {}, response=""))
        sess.field_valid_from["answer"] = 1
        self.assertEqual(ctrl._value_of(sess, consumer, value, edge), (None, None))

    def test_forward_head_requires_identical_leading_bytes(self) -> None:
        a, b = site("A", 1), site("B", 2)
        ax = Binding("x", Heterogeneity.COPY, field="x", label="x = ")
        renamed = Binding("renamed", Heterogeneity.TAKE, field="x", label="renamed = ")
        ta, tb = template(a, "A", [ax]), template(b, "B", [renamed])
        p = Program(a)
        p.add_site(ta)
        p.add_site(tb)
        p.add_edge(a, b)
        inst = CallInstance(ta, {"x": "'value'"})
        sess = LiveSession("s", program=p)
        ctrl = Controller.__new__(Controller)

        self.assertEqual(ctrl._forward_head(sess, inst), "")
        renamed.label = "x = "
        self.assertEqual(ctrl._forward_head(sess, inst), "x = 'value'")
        tb.system_prompt = "other"
        self.assertEqual(ctrl._forward_head(sess, inst), "")


class PlannerBookkeepingTests(unittest.TestCase):
    def test_session_end_retires_history_but_keeps_static_prefix(self) -> None:
        planner = KVPlanner(object())
        ids = list(range(100))
        planner.note_served("s", ids, 80, static_len=10)
        self.assertEqual(planner._demote, [(ids, 80)])
        self.assertEqual(planner._served["s"], [(ids, 10)])
        planner.drop_session("s")
        self.assertEqual(planner._retire, [(ids, 10)])
        self.assertNotIn("s", planner._served)
        planner.drop_session("s")
        self.assertEqual(planner._retire, [(ids, 10)])

    def test_invalidation_does_not_widen_static_retirement_boundary(self) -> None:
        planner = KVPlanner(object())
        ids = list(range(100))
        planner.note_served("s", ids, 80, static_len=10)
        planner.invalidate("s", ids, 40)
        planner.drop_session("s")
        self.assertEqual(planner._demote[-1], (ids, 40))
        self.assertEqual(planner._retire, [(ids, 10)])

    def test_invalidation_updates_the_retirement_boundary(self) -> None:
        planner = KVPlanner(object())
        ids = [1, 2, 3, 4]
        planner._served["s"] = [(ids, 3)]
        planner.invalidate("s", ids, 1)
        self.assertEqual(planner._served["s"], [(ids, 1)])
        self.assertEqual(planner._demote, [(ids, 1)])

    def test_gc_marks_the_priority_map_dirty(self) -> None:
        planner = KVPlanner(object())
        planner._jobs["s"] = {
            "j": Job("j", "s", 1, "site", "hold", [1] * 16,
                     1.0, 1.0, time.monotonic(), 0.0, 0)
        }
        planner._dirty = False
        planner._gc(time.monotonic())
        self.assertEqual(planner._jobs["s"]["j"].state, "void")
        self.assertTrue(planner._dirty)

    def test_protection_is_selective_by_value_density_within_the_budget(self) -> None:
        class Engine:
            def __init__(self):
                self.calls = []
                self.ledger = SimpleNamespace(device_cap_tokens=lambda: 1000)

            def cost(self, toks):
                return (0, 0)          # everything resident

            async def set_kv_priority(self, demote, protect, rid, retire):
                self.calls.append(protect)

        engine = Engine()
        planner = KVPlanner(engine)
        now = time.monotonic()
        big_soon = Job("s1|hold|A", "s1", 1, "A", "hold", [1] * 500, 1.0, 0.0, now, now, 0)
        small_soon = Job("s2|hold|B", "s2", 1, "B", "hold", [2] * 100, 1.0, 0.0, now, now, 0)
        big_late = Job("s3|hold|C", "s3", 1, "C", "hold", [3] * 500, 1.0, 0.0, now + 100, now + 100, 0)
        for j in (big_soon, small_soon, big_late):
            planner._jobs[j.sid] = {j.key: j}
        planner._dirty = True
        asyncio.run(planner.push_priorities(now))
        protected = engine.calls[-1]
        # density: big_soon 500/0.5 > small_soon 100/0.5 > big_late 500/100; the budget
        # (60% of 1000) takes the two soon jobs and stops before the late one
        self.assertEqual([len(ids) for ids, _ in protected], [500, 100])
        self.assertGreater(protected[0][1], protected[1][1])

    def test_retire_session_moves_served_prompts_and_voids_jobs(self) -> None:
        planner = KVPlanner(object())
        planner._served["s"] = [([1, 2, 3, 4], 2)]
        planner._jobs["s"] = {"j": Job("j", "s", 1, "site", "hold", [1] * 16, 1.0, 1.0, 0.0, 0.0, 0)}
        planner.retire_session("s")
        self.assertEqual(planner._retire, [([1, 2, 3, 4], 2)])
        self.assertNotIn("s", planner._served)
        self.assertEqual(planner._jobs["s"]["j"].state, "void")
        self.assertTrue(planner._dirty)

    def test_empty_priority_replacement_clears_the_previous_plan(self) -> None:
        class Engine:
            def __init__(self):
                self.calls = []

            async def set_kv_priority(self, demote, protect, rid, retire):
                self.calls.append((demote, protect, retire))

        engine = Engine()
        planner = KVPlanner(engine)
        planner._dirty = True
        asyncio.run(planner.push_priorities(time.monotonic()))
        self.assertEqual(engine.calls, [([], [], [])])


class StaticPrefixTests(unittest.TestCase):
    def test_header_first_keeps_constant_tools_but_not_private_history(self) -> None:
        t = template(site("Agent", 1), "Agent", [
            Binding("tools", Heterogeneity.CONST, literal="'tools'", label="tools = "),
            Binding("question", Heterogeneity.COPY, field="question", label="question = "),
            Binding("history", Heterogeneity.EXTEND, field="history", label="history = "),
        ])
        t.header, t.header_last, t.no_header_last = "HEADER", False, True
        values = {"tools": "'tools'", "question": "'private'", "history": "['private']"}
        before = t.render(values)
        self.assertEqual(t.fixed_head(), "HEADER\ntools = 'tools'")
        self.assertEqual(t.render(values), before)
        self.assertTrue(before.startswith(t.fixed_head()))
        self.assertFalse(t.header_last)

    def test_header_last_and_empty_header_keep_only_leading_constants(self) -> None:
        t = template(site("Agent", 1), "Agent", [
            Binding("x", Heterogeneity.CONST, literal="1", label="x = "),
            Binding("y", Heterogeneity.COPY, field="y", label="y = "),
        ])
        t.header = "HEADER"
        self.assertEqual(t.fixed_head(), "x = 1")
        t.header, t.header_last = "", False
        self.assertEqual(t.fixed_head(), "x = 1")


if __name__ == "__main__":
    unittest.main()
