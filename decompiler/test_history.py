"""Self-check for accumulating-history learning: `python -m decompiler.test_history`.

A byllm site declares (a, q, history) with the history LAST. Over one session the
learner must classify a as session-constant (copy), q as fresh, history as
accumulating (extend); freeze the layout a | history | q; rebuild the next call's
prompt up to the history so far; and relayout a request byte-exactly and
idempotently."""
from decompiler.parser import decompose, relayout_body, relayout_user, split_byllm_user
from decompiler.primitives import CallObservation, Program

HEAD = ("Agent.step(a: str, q: str, history: list[str]) -> str --- one step of the loop\n"
        "      a: str ---- the anchor\n"
        "      q: str ---- the fresh question\n"
        "      history: list[str] ---- everything answered so far")
HINT = "\n\nSchema requirements:\n- return a str"


def body(a, q, hist):
    user = f"{HEAD}\na = {a!r}\nq = {q!r}\nhistory = {hist!r}" + HINT
    return {"model": "m", "messages": [{"role": "system", "content": "sys"},
                                       {"role": "user", "content": user}]}


def session(prog, sid, n=5, serve=lambda b: None):
    """n calls of the one site; the history grows by the previous reply each call."""
    obs, walked, hist = [], [], []
    for i in range(n):
        b = body("x", f"fresh-{sid}-{i}", list(hist))
        serve(b)
        site, ex = decompose(b)
        site = prog.add_callsite(site)
        obs.append(CallObservation(key=site.key, t_arrive=float(i), t_done=i + 0.5,
                                   bindings=ex.bindings, user_text=ex.user_text, response=f"r{i}"))
        walked.append(site.key)
        hist.append(f"r{i}")
    return walked, obs


if __name__ == "__main__":
    prog = Program()
    walked, obs = session(prog, "s1")
    key = walked[0]
    site = prog.sites[key]

    # session 1: the rules settle and the layout freezes
    assert prog.update_graph(walked, obs) == [key], prog.flow
    assert prog._dominant(key, "history") == "extend", dict(prog.flow[(key, "history")])
    assert prog._dominant(key, "a") == "copy", dict(prog.flow[(key, "a")])
    assert prog._dominant(key, "q") is None, dict(prog.flow[(key, "q")])
    assert site.layout == ["a", "history", "q"], site.layout
    assert prog.proto[key]["order"] == site.layout

    # relayout: binding blocks permuted, header/schema/hint untouched, idempotent
    raw = body("x", "fresh", ["r0", "r1"])["messages"][1]["content"]
    laid = relayout_user(raw, site.layout)
    assert laid == f"{HEAD}\na = 'x'\nhistory = ['r0', 'r1']\nq = 'fresh'" + HINT, laid
    assert relayout_user(laid, site.layout) == laid
    assert split_byllm_user(laid)[2] == {"a": "'x'", "history": "['r0', 'r1']", "q": "'fresh'"}
    assert split_byllm_user(raw)[2] == {"a": "'x'", "q": "'fresh'", "history": "['r0', 'r1']"}
    b = body("x", "fresh", ["r0", "r1"])
    assert relayout_body(b, site.layout) and b["messages"][1]["content"] == laid
    assert not relayout_body(b, site.layout)
    assert decompose(b)[0].key == key                    # order does not change identity

    # session 2, served in the frozen order: the rebuild of call 3 knows header | a |
    # history-so-far and stops there (the tail is open); the rules stay settled
    walked2, obs2 = session(prog, "s2", serve=lambda b: relayout_body(b, site.layout))
    text, complete = prog.resolve_user(key, obs2[:2])
    assert not complete
    assert text == f"{HEAD}\na = 'x'\nhistory = ['r0'", text
    assert prog.update_graph(walked2, obs2) == []       # frozen once, never again
    assert site.layout == ["a", "history", "q"]
    assert prog._dominant(key, "history") == "extend"
    print("ok")
