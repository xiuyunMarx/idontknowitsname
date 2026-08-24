"""Aggregate a coldstart_experiment.py results directory.

Validity first: a (trial, tenant) pair between the baseline cell and a
comparison cell is used only when both clients exited 0 and both served the
same callsite sequence (same work). Everything else is dropped and counted.

Primary metric is uncached prompt tokens at serve (prompt_tokens -
cached_tokens): deterministic, unaffected by queueing. TTFT is reported next
to it. Per tenant and pooled, paired differences with a bootstrap 95% CI.

Usage: python experiments/coldstart_aggregate.py <results_dir> [--base off]
"""

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SERVE = re.compile(r"\[serve\] (\S+) duration_ms=([\d.]+) cached_tokens=(\d+) prompt_tokens=(\d+)")
DECODE = re.compile(r"\[decode\] \S+ decode_ms=([\d.]+) output_tokens=(\d+)")
PREFILL = re.compile(r"\[prefill\] \S+ cost_tokens=(\d+) duration_ms=([\d.]+)")


def parse(root):
    """-> serves[(mode, cell, trial)] = [serve dicts in log order], tbts[(mode, cell)],
    spec[(mode, cell, trial)] = (prefill count, tokens, ms)"""
    serves, tbts, spec = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0, 0.0])
    for mode in sorted(os.listdir(root)):
        mdir = os.path.join(root, mode)
        if not os.path.isdir(mdir) or mode not in ("single", "multi"):
            continue
        for cell in sorted(os.listdir(mdir)):
            for tdir in sorted(os.listdir(os.path.join(mdir, cell))):
                log = os.path.join(mdir, cell, tdir, "server.log")
                if not os.path.exists(log):
                    continue
                trial = int(tdir.split("-")[1])
                key = (mode, cell, trial)
                for line in open(log, errors="replace"):
                    line = ANSI.sub("", line)
                    if m := SERVE.search(line):
                        rid = m.group(1)
                        tenant, site = (rid.split(":", 2) + ["?", "?"])[:2]
                        serves[key].append({"tenant": tenant, "site": site.split("@")[0],
                                            "ttft": float(m.group(2)), "cached": int(m.group(3)),
                                            "prompt": int(m.group(4))})
                    elif m := DECODE.search(line):
                        if (n := int(m.group(2))) > 1:
                            tbts[(mode, cell)].append(float(m.group(1)) / (n - 1))
                    elif m := PREFILL.search(line):
                        spec[key][0] += 1
                        spec[key][1] += int(m.group(1))
                        spec[key][2] += float(m.group(2))
    return serves, tbts, spec


def load_rc(root):
    rc = {}
    path = os.path.join(root, "clients.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            rc[(r["mode"], r["cell"], r["trial"], r["tenant"])] = r["rc"]
    return rc


def per_tenant(serves_for_trial):
    out = defaultdict(list)
    for s in serves_for_trial:
        out[s["tenant"]].append(s)
    return out


def boot_ci(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def med(v):
    return statistics.median(v) if v else float("nan")


def fmt_ci(diffs):
    if len(diffs) < 3:
        return f"n={len(diffs)} (too few)"
    lo, hi = boot_ci(diffs)
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{statistics.mean(diffs):+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"


def compare(mode, base, cell, serves, rc, out):
    """Paired comparison of `cell` against `base` in `mode`."""
    trials = sorted({t for (m, c, t) in serves if m == mode and c in (base, cell)})
    tenants = sorted({s["tenant"] for (m, c, t), ss in serves.items() if m == mode for s in ss})
    pairs = defaultdict(list)  # tenant -> [(base serves, cell serves)]
    dropped = defaultdict(int)
    for t in trials:
        b, c = per_tenant(serves.get((mode, base, t), [])), per_tenant(serves.get((mode, cell, t), []))
        for tenant in tenants:
            if tenant not in b or tenant not in c:
                dropped["missing"] += 1
                continue
            if rc.get((mode, base, t, tenant), 0) != 0 or rc.get((mode, cell, t, tenant), 0) != 0:
                dropped["client failed"] += 1
                continue
            if [s["site"] for s in b[tenant]] != [s["site"] for s in c[tenant]]:
                dropped["different callsite sequence"] += 1
                continue
            pairs[tenant].append((b[tenant], c[tenant]))

    out.append(f"\n### {mode}: {cell} vs {base}\n")
    out.append(f"trials={len(trials)}; pairs used={sum(len(v) for v in pairs.values())}; "
               f"dropped: {dict(dropped) or 'none'}\n")
    out.append("| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | "
               "ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |")
    out.append("|---|---|---|---|---|---|---|")
    pooled_unc, pooled_ttft, pooled_b, pooled_c = [], [], [], []
    for tenant in tenants:
        ps = pairs.get(tenant, [])
        if not ps:
            out.append(f"| {tenant} | 0 | – | – | – | – | – |")
            continue
        unc_b = [sum(s["prompt"] - s["cached"] for s in b) for b, _ in ps]
        unc_c = [sum(s["prompt"] - s["cached"] for s in c) for _, c in ps]
        tt_b = [sum(s["ttft"] for s in b) for b, _ in ps]
        tt_c = [sum(s["ttft"] for s in c) for _, c in ps]
        d_unc = [x - y for x, y in zip(unc_b, unc_c)]
        d_tt = [x - y for x, y in zip(tt_b, tt_c)]
        pooled_unc += d_unc
        pooled_ttft += d_tt
        pooled_b += tt_b
        pooled_c += tt_c
        out.append(f"| {tenant} | {len(ps)} | {med(unc_b):.0f}→{med(unc_c):.0f} | {fmt_ci(d_unc)} "
                   f"| {med(tt_b):.0f}→{med(tt_c):.0f} | {fmt_ci(d_tt)} | {med(tt_b) / med(tt_c):.2f}x |")
    if pooled_unc:
        out.append(f"| **all** | {len(pooled_unc)} | | {fmt_ci(pooled_unc)} | {med(pooled_b):.0f}→{med(pooled_c):.0f} "
                   f"| {fmt_ci(pooled_ttft)} | {med(pooled_b) / med(pooled_c):.2f}x |")
    out.append("\n`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.")
    return pairs


def callsite_table(mode, cells, serves, rc, out):
    """First serve per callsite: median uncached tokens and TTFT, per cell."""
    first = defaultdict(lambda: defaultdict(list))  # (tenant, site) -> cell -> [(uncached, ttft)]
    for (m, c, t), ss in serves.items():
        if m != mode:
            continue
        seen = set()
        for s in ss:
            if rc.get((m, c, t, s["tenant"]), 0) != 0:
                continue
            k = (s["tenant"], s["site"])
            if k in seen:
                continue
            seen.add(k)
            first[k][c].append((s["prompt"] - s["cached"], s["ttft"], s["prompt"]))
    out.append(f"\n### {mode}: first serve per callsite — median uncached tokens / TTFT ms\n")
    out.append("| tenant:site | prompt | " + " | ".join(cells) + " |")
    out.append("|---|---|" + "---|" * len(cells))
    for (tenant, site) in sorted(first):
        row = []
        prompt = med([p for c in cells for _, _, p in first[(tenant, site)].get(c, [])])
        for c in cells:
            v = first[(tenant, site)].get(c, [])
            row.append(f"{med([u for u, _, _ in v]):.0f} / {med([t for _, t, _ in v]):.0f}" if v else "–")
        out.append(f"| {tenant}:{site} | {prompt:.0f} | " + " | ".join(row) + " |")


def cell_summary(mode, cells, serves, tbts, spec, rc, out):
    out.append(f"\n### {mode}: per-cell totals\n")
    out.append("| cell | trials | serves | warm fraction (cached/prompt) | mean TBT ms | spec prefills/trial | "
               "spec tokens/trial | spec ms/trial |")
    out.append("|---|---|---|---|---|---|---|---|")
    for c in cells:
        keys = [k for k in serves if k[0] == mode and k[1] == c]
        ss = [s for k in keys for s in serves[k] if rc.get((k[0], k[1], k[2], s["tenant"]), 0) == 0]
        cached, prompt = sum(s["cached"] for s in ss), sum(s["prompt"] for s in ss)
        sp = [spec[k] for k in keys]
        n = len(keys) or 1
        tbt = statistics.mean(tbts[(mode, c)]) if tbts.get((mode, c)) else float("nan")
        out.append(f"| {c} | {len(keys)} | {len(ss)} | {cached / prompt if prompt else float('nan'):.1%} | {tbt:.2f} "
                   f"| {sum(x[0] for x in sp) / n:.0f} | {sum(x[1] for x in sp) / n:.0f} | {sum(x[2] for x in sp) / n:.0f} |")


def waste(mode, base, cell, pairs, serves, spec, out):
    """Speculative tokens spent vs uncached tokens actually removed at serve."""
    spent = sum(spec[(mode, cell, t)][1] for (m, c, t) in serves if m == mode and c == cell)
    saved = sum(sum(s["prompt"] - s["cached"] for s in b) - sum(s["prompt"] - s["cached"] for s in c)
                for ps in pairs.values() for b, c in ps)
    if spent:
        out.append(f"\nSpeculation efficiency ({mode}, {cell}): {spent} speculative tokens prefilled, "
                   f"{saved} uncached tokens removed from paired serves → {saved / spent:.2f} useful per spent "
                   f"(pairs only; unpaired trials' spend is counted, their savings are not).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="off")
    args = ap.parse_args()
    serves, tbts, spec = parse(args.root)
    rc = load_rc(args.root)
    cfg_path = os.path.join(args.root, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    out = [f"# Cold-start results: `{os.path.basename(os.path.abspath(args.root))}`\n",
           f"config: {json.dumps({k: v for k, v in cfg.items() if k != 'cell_flags'})}\n"]
    failed = sum(1 for v in rc.values() if v != 0)
    out.append(f"client rows: {len(rc)}, failed (rc≠0): {failed}\n")

    for mode in ("single", "multi"):
        cells = sorted({c for (m, c, t) in serves if m == mode}, key=lambda c: (c != args.base, c))
        if not cells:
            continue
        out.append(f"\n## {mode}-task cold start\n")
        cell_summary(mode, cells, serves, tbts, spec, rc, out)
        for cell in cells:
            if cell == args.base:
                continue
            pairs = compare(mode, args.base, cell, serves, rc, out)
            waste(mode, args.base, cell, pairs, serves, spec, out)
        callsite_table(mode, cells, serves, rc, out)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.root, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
