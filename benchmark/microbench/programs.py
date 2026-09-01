"""Synthetic byllm-shaped programs the benchmark replays against the server.

The messages reproduce byllm's wire formats byte-exactly, so decompiler.parser
identifies them as real callsites: a byllm call is a `Qual(name: type) -> ret
--- sem` header plus `name = repr` bindings (mtir.impl.jac), a routing call is
route_visit's zone layout. Two programs:

flow    2 calls per session; the second call's argument carries the first
        call's reply verbatim, so the server can learn the resp value-flow
        rule and rebuild the whole prompt ahead of arrival.
visit   3 calls: announce -> route (a visit callsite — probe-able once its
        reply head has converged) -> act on the chosen node's type.
"""

SYS = ("This is a task you must complete by returning only the output.\n"
       "Do not include explanations.")
ROUTE_SYS = ("You are routing a graph walker. Choose which candidate node(s) the walker "
             "should visit next, by handle. Return only valid handles. Choose exactly one.")
VISIT_USER = """Goal: reach the number node

Walker:
W(step=1)

Current node:
Root()

Candidates (choose by handle):
a) here --(E())--> Num(v=1)
b) here --(E())--> Txt(s='x')"""

N_RULES = 110  # ~2.2k tokens of distinct static prefix per service


def service_system(w: int) -> str:
    """Service `w`'s distinct static prefix. Sixteen of these (~43k tokens)
    overflow a 31830-token device pool: steady-state eviction churn."""
    rules = " ".join(f"rule{w}-{i}: always keep requirement {i} of service {w} satisfied "
                     f"before responding to any downstream request." for i in range(N_RULES))
    return f"You are service worker {w}. Return only the output, no explanations.\n{rules}"


def step_user(w: int, s: int) -> str:
    return f"Svc{w}.step(x: str) -> str --- repeat x back exactly\nx = 'job{s}'"


def report_user(w: int, reply: str) -> str:
    return f"Svc{w}.report(y: str) -> str --- say done\ny = " + repr(reply)


def scout_user() -> str:
    return "Scout.start(g: str) -> str --- announce the goal\ng = 'go'"


def act_user() -> str:
    return "Num.act(v: int) -> str --- act on the node\nv = 1"
