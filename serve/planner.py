"""Ordering: which model goes on the GPU next.

Every live instance contributes the sequence of models it still needs: its current
model, then what branch prediction says follows, to the end of the program. Loads
are ordered first-in-first-out over those chains: an instance that arrived earlier
gets all of its predicted models ahead of a later one's."""
from typing import List, Sequence


def fifo_order(seqs: Sequence[Sequence[str]]) -> List[str]:
    """FIFO over predicted chains: every model of each sequence, in arrival order, once."""
    out: List[str] = []
    for s in seqs:
        for m in s:
            if m not in out:
                out.append(m)
    return out
