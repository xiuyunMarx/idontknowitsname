"""The planner's view of an instance's future: a loop-aware walk over the branch predictor.

`Program.predict_path` stops at the first callsite already on the path, so a program
with a retry or multi-hop loop is only predicted up to its first lap. The cost-aware
planners need the laps too (they are where models get revisited), so this walk follows
the most likely successor at every step and lets a callsite repeat as often as its
loop-edge probability says it usually does: an edge taken with probability p is
expected to be taken p/(1-p) more times, so a successor may be revisited
round(p/(1-p)) times beyond its first visit before the next-best successor is taken."""
from typing import Dict, List, Tuple


def predicted_chain(program, callsite_key: str, fallback_model: str, max_steps: int = 8) -> List[Tuple[str, str]]:
    """[(callsite_key, model), ...] after `callsite_key`, at most `max_steps` long."""
    out: List[Tuple[str, str]] = []
    visits: Dict[str, int] = {callsite_key: 1}
    key = callsite_key
    while len(out) < max_steps:
        nxt = None
        for succ, p in program.branch_candidates(key):
            allowed = 1 + (int(round(p / (1.0 - p))) if p < 1.0 else max_steps)
            if visits.get(succ, 0) < allowed:
                nxt = succ
                break
        if nxt is None:
            break
        visits[nxt] = visits.get(nxt, 0) + 1
        out.append((nxt, program.model_of(program.callsites[nxt], fallback_model)))
        key = nxt
    return out
