"""Ordering: which model goes on the GPU next.

Every live request contributes the sequence of models it still needs (its current
model, then what the compiler predicts), each tagged with the static interval in
which that need arrives: [earliest, latest] seconds from now, from the max_tokens
of the callsites before it and the model's profiled step time. Order within a
sequence is fixed, across sequences free, and one load of a model serves every
sequence whose arrival interval that load can cover. The cheapest load order is a
weighted shortest common supersequence over the tuple of per-sequence positions,
with the GPU-resident set carried in the DP state so a model stays free while it
stays resident and evictions are chosen by the DP itself (few models, short
windows, small slot counts, so exact is fine)."""
from functools import lru_cache
from typing import Callable, List, Sequence, Tuple

Need = Tuple[str, float, float]  # (model, earliest, latest) seconds from now


def scs_order(seqs: Sequence[Sequence[Need]], cost: Callable[[str, str], float], tier: Callable[[str], str],
              resident: "set[str]" = frozenset(), slots: int = 1, hold: float = 0.0,
              max_seqs: int = 6, depth: int = 4) -> List[str]:
    """Load order minimising total load cost.

    `cost(model, tier)` is the load time from `tier` ("host" / "ssd"); `tier(model)`
    is where a model not on the GPU currently lives. A model in `resident` costs
    nothing until the plan evicts it; a model evicted within the plan is reloaded
    from host (offload parks it there; `_trim_host` demotions to SSD are not
    modelled). When the GPU holds `slots` models a load must evict one, and the DP
    tries every candidate.

    One load step of `m` placed at time t serves every sequence whose head is `m`
    with earliest <= t <= latest + hold; a need outside that window is served by a
    later step, free if `m` is still resident then, so `hold` only says how far a
    single step may reach (0: overlapping needs; inf: no eviction pressure, the
    plain SCS). Candidate t's are each head's latest + hold, which is enough: any
    window is dominated by one of them."""
    seqs = [tuple(s[:depth]) for s in seqs if s]
    seqs = seqs[:max_seqs] + [s[:1] for s in seqs[max_seqs:]]  # beyond the window only the immediate need counts
    if not seqs:
        return []
    initial = frozenset(resident)
    slots = max(1, slots)

    def load_cost(m: str) -> float:
        return cost(m, "host" if m in initial else tier(m))

    def served(pos: Tuple[int, ...], m: str) -> List[Tuple[int, ...]]:
        """Distinct sets of sequences one load of `m` can serve now, each as the next pos tuple."""
        heads = [(i, seqs[i][p][1], seqs[i][p][2]) for i, p in enumerate(pos) if p < len(seqs[i]) and seqs[i][p][0] == m]
        outs = set()
        for _, _, l_i in heads:
            t = l_i + hold
            cover = {j for j, e_j, l_j in heads if e_j <= t <= l_j + hold}
            outs.add(tuple(p + 1 if i in cover else p for i, p in enumerate(pos)))
        return sorted(outs)

    @lru_cache(maxsize=None)
    def best(pos: Tuple[int, ...], res: frozenset) -> Tuple[float, Tuple[str, ...]]:
        heads = {seqs[i][p][0] for i, p in enumerate(pos) if p < len(seqs[i])}
        if not heads:
            return 0.0, ()
        top = None
        for m in sorted(heads):
            if m in res:
                cands = [(0.0, res)]
            elif len(res) < slots:
                cands = [(load_cost(m), res | {m})]
            else:
                cands = [(load_cost(m), (res - {v}) | {m}) for v in sorted(res)]
            for nxt in served(pos, m):
                for c, nres in cands:
                    c2, rest = best(nxt, nres)
                    if top is None or c + c2 < top[0]:
                        top = (c + c2, (m,) + rest)
        return top  # type: ignore[return-value]

    order: List[str] = []
    for m in best(tuple(0 for _ in seqs), initial)[1]:  # a model kept resident may serve a later need "again"
        if not order or order[-1] != m:
            order.append(m)
    return order


def fifo_order(seqs: Sequence[Sequence[str]]) -> List[str]:
    """Reactive baseline: first model of each sequence, in arrival order."""
    out: List[str] = []
    for s in seqs:
        if s and s[0] not in out:
            out.append(s[0])
    return out
