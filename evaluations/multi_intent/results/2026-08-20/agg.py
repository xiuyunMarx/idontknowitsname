#!/usr/bin/env python3
"""Aggregate a cold-start sweep; A/B on matched (case, callsite) pairs."""
import json, pathlib, statistics as S, sys

def load(d):
    out = {}
    for f in sorted(pathlib.Path(d).glob("*.json")):
        j = json.loads(f.read_text())
        out[j["case"]["sid"]] = j
    return out

def rows(j):
    return [c for c in j["stats"]["stats"]["calls"] if c.get("ttft") is not None]

def summarize(name, runs):
    print(f"\n===== {name}  ({len(runs)} cases) =====")
    per_key = {}
    sums = []
    for sid, j in runs.items():
        cs = rows(j)
        sums.append(sum(c["ttft"] for c in cs) * 1000)
        for c in cs:
            per_key.setdefault(c["key"], []).append((c["ttft"] * 1000, c["cached_tokens"] or 0))
    print(f"{'callsite':<24}{'n':>3}{'medTTFT':>10}{'meanTTFT':>10}{'medCached':>11}")
    for k, v in sorted(per_key.items(), key=lambda kv: -len(kv[1])):
        t = [x[0] for x in v]; c = [x[1] for x in v]
        print(f"{k:<24}{len(v):>3}{S.median(t):>9.1f}ms{S.mean(t):>9.1f}ms{S.median(c):>11.0f}")
    print(f"per-case SumTTFT: median {S.median(sums):.1f}ms  mean {S.mean(sums):.1f}ms")
    return per_key, sums

def ab(a, b):
    ra, rb = load(a), load(b)
    ka, sa = summarize("BASELINE (--no-speculate)", ra)
    kb, sb = summarize("SPECULATION", rb)
    shared = sorted(set(ra) & set(rb))
    print(f"\n===== matched (case, callsite) A/B  [{len(shared)} shared cases] =====")
    pair = {}
    for sid in shared:
        ia = {}; ib = {}
        for c in rows(ra[sid]): ia.setdefault(c["key"], []).append(c)
        for c in rows(rb[sid]): ib.setdefault(c["key"], []).append(c)
        for k in set(ia) & set(ib):
            for x, y in zip(ia[k], ib[k]):
                pair.setdefault(k, []).append((x["ttft"] * 1000, y["ttft"] * 1000,
                                               x["cached_tokens"] or 0, y["cached_tokens"] or 0))
    print(f"{'callsite':<24}{'n':>3}{'base':>10}{'spec':>10}{'gain':>8}{'cached b->s':>16}")
    tb = ts = 0.0
    for k, v in sorted(pair.items(), key=lambda kv: -len(kv[1])):
        b_ = S.median([x[0] for x in v]); s_ = S.median([x[1] for x in v])
        cb = S.median([x[2] for x in v]); cs_ = S.median([x[3] for x in v])
        tb += sum(x[0] for x in v); ts += sum(x[1] for x in v)
        print(f"{k:<24}{len(v):>3}{b_:>9.1f}ms{s_:>9.1f}ms{b_/s_ if s_ else 0:>7.2f}x{f'{cb:.0f} -> {cs_:.0f}':>16}")
    print(f"{'TOTAL (matched calls)':<24}{'':>3}{tb:>9.1f}ms{ts:>9.1f}ms{tb/ts if ts else 0:>7.2f}x")
    ca = {s: sum(c['ttft'] for c in rows(ra[s]))*1000 for s in shared}
    cbb = {s: sum(c['ttft'] for c in rows(rb[s]))*1000 for s in shared}
    print(f"per-case SumTTFT median: {S.median(list(ca.values())):.1f} -> {S.median(list(cbb.values())):.1f}ms "
          f"({S.median(list(ca.values()))/S.median(list(cbb.values())):.2f}x)")

if __name__ == "__main__":
    if len(sys.argv) == 3: ab(sys.argv[1], sys.argv[2])
    else: summarize(sys.argv[1], load(sys.argv[1]))
