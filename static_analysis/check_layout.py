"""Is the chosen re-layout optimal for prefix reuse, under the static model?

    python -m static_analysis.check_layout static_analysis/.cache/<program>.json [--trace P,C,A,C,P,C,Cm]

Every program-wide field order is tried (all sites lay shared fields out in
that order, unshared ones keep their relative placement), and the score of the
server's own layout (Program.decide_layouts) is compared with the best. Weights:
EXTEND fields 4, others 1 (a growing history is the bulk of a prompt); the
unweighted count is reported too.
"""
from __future__ import annotations

import itertools
import json
import sys
from typing import Dict, List, Optional, Tuple

from static_analysis.primitives import Binding, BindingKind, CallSiteID, Heterogeneity, Program, program_from_dict

H = Heterogeneity


def default_trace(p: Program) -> List[CallSiteID]:
    """A representative path: from the entry, prefer the successor not yet
    visited, revisit loops once, stop at an exit after ~2 loop rounds."""
    order: List[CallSiteID] = []
    cur = p.entry
    seen: Dict[CallSiteID, int] = {}
    for _ in range(12):
        order.append(cur)
        seen[cur] = seen.get(cur, 0) + 1
        succ = [e.dst for e in p.successors(cur)]
        if not succ:
            break
        succ.sort(key=lambda k: (seen.get(k, 0), repr(k)))
        cur = succ[0]
        if cur in p.exits and len(order) >= 6:
            order.append(cur)
            break
    return order


def replay(p: Program, orders: Dict[CallSiteID, List[str]], trace: List[CallSiteID], weighted: bool) -> float:
    """Total reusable prefix over the trace under the given per-site orders."""
    version: Dict[str, int] = {}          # field -> version (bumped on a write)
    extended: Dict[str, bool] = {}        # field -> last change was an extension
    cache: List[List[Tuple[str, str]]] = []   # earlier prompts as [(field-or-name, symbolic bytes)]
    total = 0.0
    prev: Optional[CallSiteID] = None
    for key in trace:
        t = p.sites[key]
        e = p.edges.get((prev, key)) if prev is not None else None
        if e is not None:
            for f in e.writes:
                version[f] = version.get(f, 0) + 1
                kinds = [b.heterogeneity for s in p.sites.values() for b in s.params if b.field == f]
                extended[f] = H.EXTEND in kinds
        # the prompt's bindings as symbolic bytes
        prompt: List[Tuple[str, str]] = []
        for name in orders[key]:
            b = t.binding(name)
            if b.heterogeneity is H.CONST:
                prompt.append((f"const:{b.literal}", "k"))
            elif b.field:
                base = b.field[:-2] if b.field.endswith("[]") else b.field
                v = version.get(base, 0)
                if b.field.endswith("[]") or b.heterogeneity is H.VOLATILE:
                    v = len(cache) * 1000 + v      # fresh every call
                prompt.append((b.field, f"v{v}"))
            else:
                prompt.append((name, f"fresh{len(cache)}"))
        # longest reusable prefix against any cached prompt
        best = 0.0
        for old in cache:
            score, n = 0.0, 0
            for (f1, v1), (f2, v2) in zip(prompt, old):
                if f1 != f2:
                    break
                w = 4.0 if (weighted and extended.get(f1, False)) else 1.0
                if v1 == v2:
                    score += w
                    n += 1
                    continue
                if f1 in extended and extended[f1] and int(v1[1:]) > int(v2[1:]):
                    score += w / 2      # the old bytes are a prefix of the new
                break
            best = max(best, score)
        total += best
        cache.append(prompt)
        prev = key
    return total


def orders_for(p: Program, field_order: List[str]) -> Dict[CallSiteID, List[str]]:
    """Every site lays its params out by the global field order; params without a
    field keep their relative position after the ordered ones."""
    rank = {f: i for i, f in enumerate(field_order)}
    out = {}
    for key, t in p.sites.items():
        names = [b.name for b in t.params]
        out[key] = sorted(names, key=lambda n: (rank.get(t.binding(n).field, len(rank)), names.index(n)))
    return out


def main(argv: List[str]) -> int:
    path = argv[0]
    d = json.load(open(path))
    p = program_from_dict(d["program"] if "program" in d else d)
    p.decide_layouts(header_last=True)
    trace = default_trace(p)
    if "--trace" in argv:
        names = argv[argv.index("--trace") + 1].split(",")
        by = {k.signature.split(".")[0]: k for k in p.sites}
        trace = [next(k for k in p.sites if k.signature.startswith(n)) for n in names]
    print("trace:", " -> ".join(repr(k) for k in trace))
    chosen = {k: list(t.order) for k, t in p.sites.items()}
    fields = sorted({b.field for t in p.sites.values() for b in t.params if b.field})
    results = []
    for perm in itertools.permutations(fields):
        o = orders_for(p, list(perm))
        results.append((replay(p, o, trace, True), replay(p, o, trace, False), perm, o))
    for weighted_idx, label in ((0, "weighted"), (1, "unweighted")):
        best = max(r[weighted_idx] for r in results)
        mine = (replay(p, chosen, trace, weighted_idx == 0))
        print(f"[{label}] server layout = {mine:.1f}, best over {len(results)} global orders = {best:.1f}"
              + ("  OPTIMAL" if mine >= best - 1e-9 else "  SUBOPTIMAL"))
        if mine < best - 1e-9:
            for r in sorted(results, key=lambda r: -r[weighted_idx])[:3]:
                print("   better:", r[2], {repr(k): v for k, v in r[3].items()})
    for k, t in p.sites.items():
        print(f"  {k!r:36} {t.order}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
