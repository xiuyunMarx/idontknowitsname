import asyncio
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
