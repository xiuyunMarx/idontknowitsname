"""Ordering: which model goes on the GPU next.

Every live request contributes the model sequence it still needs (its current
model, then what the compiler predicts). Order within a sequence is fixed, across
sequences free, and one load of a model serves every sequence waiting for it. The
cheapest load order is a weighted shortest common supersequence: DP over the
tuple of per-sequence positions (few models, short windows, so exact is fine)."""
from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple


def scs_order(seqs: Sequence[Sequence[str]], cost: Callable[[str], float],
              resident: "set[str]" = frozenset(), max_seqs: int = 6, depth: int = 4) -> List[str]: #type: ignore[no-untyped-def]
    """Load order minimising total load cost; models already resident are free
    while they stay at the front (one of them can run before any load)."""
    seqs = [tuple(s[:depth]) for s in seqs if s]
    seqs = seqs[:max_seqs] + [s[:1] for s in seqs[max_seqs:]]  # beyond the window only the immediate need counts
    if not seqs:
        return []

    @lru_cache(maxsize=None)
    def best(pos: Tuple[int, ...], head: bool) -> Tuple[float, Tuple[str, ...]]:
        heads = {seqs[i][p] for i, p in enumerate(pos) if p < len(seqs[i])}
        if not heads:
            return 0.0, ()
        top = None
        for m in sorted(heads):
            nxt = tuple(p + 1 if p < len(seqs[i]) and seqs[i][p] == m else p for i, p in enumerate(pos))
            c, rest = best(nxt, head and m in resident)
            c += 0.0 if head and m in resident else cost(m)
            if top is None or c < top[0]:
                top = (c, (m,) + rest)
        return top  # type: ignore[return-value]

    return list(best(tuple(0 for _ in seqs), True)[1])


def fifo_order(seqs: Sequence[Sequence[str]]) -> List[str]:
    """Reactive baseline: first model of each sequence, in arrival order."""
    out: List[str] = []
    for s in seqs:
        if s and s[0] not in out:
            out.append(s[0])
    return out
