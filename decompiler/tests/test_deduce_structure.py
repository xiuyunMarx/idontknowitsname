"""Template deduction and re-layout, validated on recorded traffic. No GPU, no model.

logs/requests.jsonl holds real request bodies: fact_check.jac and coding_agent.jac
(byllm, InterceptorLLM) and the DSPy 3.3.1 programs in synth/dspy_programs.py, all run
against synth/recorder.py, a mock chat-completions server that answers with
schema-valid replies. synth/make_logs.sh regenerates the file.

byllm traffic is checked against decompiler/parser.py, the hand-written rules:
same values per binding, same re-layout bytes for any order and for header-last.
DSPy traffic has no hand parser; it is checked for byte-exact round trips, the
expected slot names, and the reorder rules the design promises.

    python -m pytest decompiler/tests -q
"""
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import pytest

from decompiler import parser
from decompiler.deduce_structure import (StructureDeducer, call_message, detect_tail, induce, key_of,
                                         system_text)
from decompiler.structure_template import PromptTemplate

LOG = os.path.join(os.path.dirname(__file__), "logs", "requests.jsonl")


@pytest.fixture(scope="module")
def records() -> List[Dict[str, Any]]:
    with open(LOG) as f:
        return [json.loads(line) for line in f]


def bodies(records, framework):
    return [r["body"] for r in records if r["framework"] == framework]


def turn0(body) -> bool:
    """A fresh call. byllm's typed retry re-sends the call with a feedback turn (a
    continuation); a DSPy History turn is a new call with the past turns prepended."""
    msgs = body["messages"]
    return len(msgs) == 2 or "[[ ## " in call_message(body)


def replay(bodies_: List[Dict[str, Any]]) -> StructureDeducer:
    """Feed the calls; pair every request with the previous request of the same key
    and session so hopping tails are learned."""
    d = StructureDeducer()
    last: Dict[Any, Dict[str, Any]] = {}
    for b in bodies_:
        sid = b.get("user", "")
        if sid in last:
            d.observe_pair(last[sid], b)
        if turn0(b):
            d.observe(b)
        last[sid] = b
    return d


# ============================================================================ byllm vs parser
@pytest.fixture(scope="module")
def byllm(records):
    return bodies(records, "byllm")


@pytest.fixture(scope="module")
def byllm_deducer(byllm):
    return replay(byllm)


def test_byllm_every_site_gets_a_template(byllm, byllm_deducer):
    keys = {key_of(b) for b in byllm if turn0(b)}
    assert len(keys) == 7, keys                         # 3 fact_check + 4 coding_agent callsites
    for k in keys:
        assert byllm_deducer.template(k) is not None, k


def test_byllm_values_match_hand_parser(byllm, byllm_deducer):
    """Every binding the hand parser finds, the deduced template finds with the same
    bytes, plus the self view; every message round-trips."""
    checked = 0
    for b in byllm:
        if not turn0(b):
            continue
        tpl = byllm_deducer.template(key_of(b))
        text = call_message(b)
        vals = tpl.split(text)
        assert vals is not None, tpl.describe()
        assert tpl.render(vals) == text
        _, names, bindings, self_view, _ = parser.split_byllm_user(text)
        for n in names:
            assert vals.get(n) == bindings[n], (n, tpl.describe())
        if self_view is not None:
            assert vals.get("self") == self_view, tpl.describe()
        checked += 1
    assert checked >= 40


def test_byllm_movable_slots_are_exactly_the_bindings(byllm, byllm_deducer):
    for b in byllm:
        if not turn0(b):
            continue
        tpl = byllm_deducer.template(key_of(b))
        _, names, _, self_view, _ = parser.split_byllm_user(call_message(b))
        if len(names) >= 2:
            assert tpl.movable_names() == names, tpl.describe()
        else:
            assert tpl.movable_names() == [], tpl.describe()  # a single binding has nothing to permute
        assert "self" not in tpl.movable_names()
        # the header+schema block leads the message and may trail it
        assert tpl.statics_can_trail() == (len(names) >= 2), tpl.describe()


def test_byllm_relayout_matches_hand_parser(byllm, byllm_deducer):
    """Any order, with and without header-last, gives the parser's bytes; the result
    splits back to the same values and re-laying it out again changes nothing."""
    for b in byllm:
        if not turn0(b):
            continue
        tpl = byllm_deducer.template(key_of(b))
        text = call_message(b)
        _, names, _, _, _ = parser.split_byllm_user(text)
        if len(names) < 2:
            continue
        vals = tpl.split(text)
        for order in (list(reversed(names)), names[1:] + names[:1], [names[-1]]):
            for header_last in (False, True):
                ours = tpl.relayout(text, order, statics_last=header_last)
                theirs = parser.relayout_user(text, order, header_last=header_last)
                assert ours == theirs, (order, header_last, tpl.describe())
                assert tpl.split(ours) == vals
                assert tpl.relayout(ours, order, statics_last=header_last) == ours


def test_byllm_fixed_head_is_header_and_schema(byllm, byllm_deducer):
    for b in byllm:
        if not turn0(b):
            continue
        tpl = byllm_deducer.template(key_of(b))
        text = call_message(b)
        ctx, names, _, _, _ = parser.split_byllm_user(text)
        head = tpl.fixed_head()
        # the parser calls every schema row fixed; a row that byllm re-renders with the
        # value's shape is an unkeyed slot here, so the head may stop before it
        assert (ctx + "\n" + (f"{names[0]} = " if names else "")).startswith(head), tpl.describe()
        assert head.startswith(ctx.split("\n", 1)[0] + "\n")
        if len(names) >= 2:
            assert tpl.fixed_head(statics_last=True) == f"{names[0]} = "


def test_byllm_typed_retry_teaches_the_hint_tail(byllm, byllm_deducer):
    """The schema hint hops to the feedback turn on a typed retry: the pair reveals
    it, and it is the hint the hand parser strips."""
    retries = [b for b in byllm if not turn0(b)]
    assert retries, "the log has no typed retry"
    keys = {key_of(prev) for prev, b in zip(byllm, byllm[1:]) if not turn0(b) and turn0(prev)}
    assert keys and keys <= set(byllm_deducer.tails), (keys, list(byllm_deducer.tails))
    for k in keys:
        tpl = byllm_deducer.template(k)
        assert tpl is not None and tpl.tail == byllm_deducer.tails[k]
        assert tpl.tail.lstrip("\n").startswith(parser.HINT_HEADER + "\n"), repr(tpl.tail[:60])   # byllm sends the hint as its own text part: one line break
    # hint-carrying sites: the value split ignores the tail and the tail is re-emitted
    for b in byllm:
        if turn0(b) and key_of(b) in byllm_deducer.tails:
            tpl = byllm_deducer.template(key_of(b))
            text = call_message(b)
            assert text.endswith(tpl.tail)
            assert tpl.render(tpl.split(text)) == text


def test_byllm_label_family_learned(byllm_deducer):
    fam = byllm_deducer.families.get(("", " = ", False), set())
    assert {"claim", "evidences", "spec", "attempts"} <= fam, fam


# ============================================================================ DSPy
@pytest.fixture(scope="module")
def dspy(records):
    return bodies(records, "dspy")


@pytest.fixture(scope="module")
def dspy_deducer(dspy):
    return replay(dspy)


def _dspy_sites(dspy, dspy_deducer):
    by_key = defaultdict(list)
    for b in dspy:
        if turn0(b):
            by_key[key_of(b)].append(b)
    return {k: (dspy_deducer.template(k), bs) for k, bs in by_key.items()}


def test_dspy_every_site_gets_a_template(dspy, dspy_deducer):
    sites = _dspy_sites(dspy, dspy_deducer)
    assert len(sites) == 6, list(sites)           # Decompose, Assess, Verify, ReAct(Verify), ReAct extract, Chat
    for k, (tpl, _) in sites.items():
        assert tpl is not None, k
    assert all(tpl.static_pre.startswith("Your input fields are:") for tpl, _ in sites.values())


def test_dspy_slots_are_the_input_fields_and_round_trip(dspy, dspy_deducer):
    for k, (tpl, bs) in _dspy_sites(dspy, dspy_deducer).items():
        fields = [f for f in _dspy_input_fields(bs[0]) if f != "history"]
        assert tpl.slot_names() == fields, (tpl.describe(), fields)
        assert tpl.movable_names() == (fields if len(fields) >= 2 else []), tpl.describe()
        for b in bs:
            text = call_message(b)
            vals = tpl.split(text)
            assert vals is not None, tpl.describe()
            assert tpl.render(vals) == text
            for f in fields:
                assert f"[[ ## {f} ## ]]\n{vals[f]}" in text or vals[f] == "", (f, vals[f])


def test_dspy_only_the_user_section_moves(dspy, dspy_deducer):
    """Headers live in the system prompt, so no static may trail; the labelled blocks
    of the user message permute, keep their bytes, and split back unchanged."""
    for k, (tpl, bs) in _dspy_sites(dspy, dspy_deducer).items():
        assert not tpl.statics_can_trail(), tpl.describe()
        fields = tpl.movable_names()
        if len(fields) < 2:
            continue
        for b in bs:
            text = call_message(b)
            vals = tpl.split(text)
            order = list(reversed(fields))
            out = tpl.relayout(text, order)
            assert out != text
            assert tpl.relayout(text, order, statics_last=True) == out   # header-last is a no-op here
            assert tpl.split(out) == vals
            assert tpl.relayout(out, order) == out
            # blocks intact: each label+value appears verbatim, in the requested order
            pos = [out.index(f"[[ ## {f} ## ]]\n{vals[f]}") for f in order]
            assert pos == sorted(pos)
            # static text unchanged: strip the blocks and compare
            strip = lambda s, o: s if not o else strip(s.replace(f"[[ ## {o[0]} ## ]]\n{vals[o[0]]}", "", 1), o[1:])
            assert strip(out, order) == strip(text, fields)


def test_dspy_react_trajectory_is_one_growing_slot(dspy, dspy_deducer):
    sites = _dspy_sites(dspy, dspy_deducer)
    react = [(tpl, bs) for tpl, bs in sites.values() if "trajectory" in tpl.slot_names() and "next_tool_name" in tpl.static_pre]
    assert len(react) == 1
    tpl, bs = react[0]
    # inner [[ ## thought_k ## ]] labels are value, not structure
    assert not any(n.startswith("thought") or n.startswith("observation") for n in tpl.slot_names())
    by_session = defaultdict(list)
    for b in bs:
        vals = tpl.split(call_message(b))
        by_session[vals["claim"]].append(vals["trajectory"])
    for traj in by_session.values():
        assert traj[0] == ""                                    # turn 0: empty trajectory
        for a, b in zip(traj, traj[1:]):
            assert b.startswith(a) and len(b) > len(a)          # every turn extends the last


def test_dspy_history_turn_teaches_the_reminder_tail(dspy, dspy_deducer):
    """The output reminder is dropped from a message once it becomes history: the
    pair reveals it, and afterwards split/render treat it as the per-request tail."""
    hist = [b for b in dspy if len(b["messages"]) == 4]
    assert hist
    k = key_of(hist[0])
    assert k in dspy_deducer.tails
    tail = dspy_deducer.tails[k]
    assert tail.startswith("\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`")
    tpl = dspy_deducer.template(k)
    assert tpl is not None and tpl.tail == tail
    for b in dspy:
        if turn0(b) and key_of(b) == k:
            text = call_message(b)
            assert tpl.split(text) == {"question": text[len("[[ ## question ## ]]\n"):-len(tail)]}
    assert dspy_deducer.families.get(("[[ ## ", " ## ]]", True), set()) >= {"claim", "evidence", "question"}


def _dspy_input_fields(body) -> List[str]:
    system = system_text(body)
    block = system.split("Your output fields are:")[0]
    return [line.split("`")[1] for line in block.split("\n") if line[:1].isdigit() and "`" in line]


# ============================================================================ synthetic: visit prompts, weak lines
def test_visit_zones_round_trip_and_stay_put():
    """byllm's route_visit layout: fixed zone headers, no identifier hole, so nothing
    is movable, and every zone comes back as a named slot."""
    def visit(goal, walker, here, cands):
        return (f"Goal: {goal}\n\nWalker:\n{walker}\n\nCurrent node:\n{here}\n\n"
                "Candidates (choose by handle):\n" + "\n".join(cands))
    texts = [visit("reach the number node", "W(step=1)", "Root()", ["a) here --(E())--> Num(v=1)", "b) here --(E())--> Txt(s='x')"]),
             visit("reach the number node", "W(step=2)", "Num(v=1)", ["a) here --(E())--> Num(v=2)", "b) here --(E())--> Txt(s='y')"]),
             visit("reach the number node", "W(step=3)", "Txt(s='y')", ["a) here --(E())--> Num(v=3)", "b) here --(E())--> Txt(s='z')", "c) here --(F())--> End()"])]
    tpl = induce("visit", texts)
    assert tpl is not None, "no template"
    for t in texts:
        assert tpl.render(tpl.split(t)) == t
    vals = tpl.split(texts[0])
    assert vals["walker"] == "W(step=1)" and vals["current_node"] == "Root()"
    assert vals["candidates"].startswith("a) here")
    assert tpl.movable_names() == []
    assert tpl.relayout(texts[0], ["candidates", "walker"]) == texts[0]


def test_weak_lines_stay_in_values():
    """`{` and `}` lines of a pretty-printed value are value bytes, not skeleton."""
    def msg(a, b):
        return f"[[ ## a ## ]]\n{{\n  \"k\": {a}\n}}\n\n[[ ## b ## ]]\n{{\n  \"k\": {b}\n}}\n\nRespond now."
    texts = [msg(1, 2), msg(3, 4), msg(5, 6)]
    tpl = induce("weak", texts, known={("[[ ## ", " ## ]]", True): {"x", "y"}})
    assert tpl is not None
    vals = tpl.split(texts[0])
    assert vals == {"a": '{\n  "k": 1\n}', "b": '{\n  "k": 2\n}'}, tpl.describe()
    assert tpl.movable_names() == ["a", "b"]
    assert tpl.render(tpl.split(texts[2])) == texts[2]


def test_detect_tail_requires_a_reappearance():
    a = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "call TAIL"}]}
    b = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "call"},
                      {"role": "user", "content": "feedback TAIL"}]}
    assert detect_tail(a, b) == " TAIL"
    assert detect_tail(a, a) is None
