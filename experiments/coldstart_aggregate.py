"""Aggregate a coldstart_experiment.py results directory.

Per cell (multi/single x off/spec), per trial: every [serve] line is a cold-path
measurement. Reports per-tenant per-trial ΣTTFT distributions, per-callsite
first-serve TTFT medians, cache-hit totals, mean TBT, and off->spec speedups.

Usage: python experiments/coldstart_aggregate.py <results_dir>
"""

import json
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SERVE = re.compile(r"\[serve\] (\S+) duration_ms=([\d.]+) cached_tokens=(\d+) prompt_tokens=(\d+)")
DECODE = re.compile(r"\[decode\] \S+ decode_ms=([\d.]+) output_tokens=(\d+)")

CELLS = ["multi-off", "multi-spec", "single-off", "single-spec"]


def parse(root):
    serves, tbts = [], {}
    for cell in CELLS:
        for trial_dir in sorted(os.listdir(os.path.join(root, cell))):
            log = os.path.join(root, cell, trial_dir, "server.log")
            if not os.path.exists(log):
                continue
            trial = int(trial_dir.split("-")[1])
            for line in open(log):
                line = ANSI.sub("", line)
                if m := SERVE.search(line):
                    rid = m.group(1)
                    tenant, site = (rid.split(":", 2) + ["?", "?"])[:2]
                    serves.append({"cell": cell, "trial": trial, "tenant": tenant,
                                   "site": site.split("@")[0], "ttft": float(m.group(2)),
                                   "cached": int(m.group(3)), "prompt": int(m.group(4))})
                elif m := DECODE.search(line):
                    if (n := int(m.group(2))) > 1:
                        tbts.setdefault(cell, []).append(float(m.group(1)) / (n - 1))
    return serves, tbts


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def main():
    root = sys.argv[1]
    serves, tbts = parse(root)
    tenants = sorted({s["tenant"] for s in serves})

    def sum_ttft(cell, tenant):
        per_trial = {}
        for s in serves:
            if s["cell"] == cell and s["tenant"] == tenant:
                per_trial[s["trial"]] = per_trial.get(s["trial"], 0.0) + s["ttft"]
        return list(per_trial.values())

    for mode in ("multi", "single"):
        print(f"\n## {mode}-tenant cold start: per-trial ΣTTFT per tenant (ms)\n")
        print("| tenant | off p25/p50/p75 | spec p25/p50/p75 | speedup (p50) |")
        print("|---|---|---|---|")
        overall = {"off": [], "spec": []}
        for t in tenants:
            row = {}
            for cond in ("off", "spec"):
                vals = sorted(sum_ttft(f"{mode}-{cond}", t))
                overall[cond] += vals
                if vals:
                    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
                    row[cond] = (q(0.25), med(vals), q(0.75))
            if "off" in row and "spec" in row:
                print(f"| {t} | {row['off'][0]:.0f}/{row['off'][1]:.0f}/{row['off'][2]:.0f} "
                      f"| {row['spec'][0]:.0f}/{row['spec'][1]:.0f}/{row['spec'][2]:.0f} "
                      f"| {row['off'][1] / row['spec'][1]:.2f}x |")
        if overall["off"] and overall["spec"]:
            print(f"| **all** | p50 {med(overall['off']):.0f} | p50 {med(overall['spec']):.0f} "
                  f"| {med(overall['off']) / med(overall['spec']):.2f}x |")

    print("\n## Per-callsite first-serve TTFT median (ms) / median cached tokens\n")
    sites = sorted({(s["tenant"], s["site"]) for s in serves})
    print("| tenant:site | " + " | ".join(CELLS) + " |")
    print("|---|" + "---|" * len(CELLS))
    for tenant, site in sites:
        cells_out = []
        for cell in CELLS:
            first = {}
            for s in serves:
                if s["cell"] == cell and s["tenant"] == tenant and s["site"] == site:
                    if s["trial"] not in first:
                        first[s["trial"]] = s
            if first:
                cells_out.append(f"{med([s['ttft'] for s in first.values()]):.0f} / "
                                 f"{med([s['cached'] for s in first.values()]):.0f}")
            else:
                cells_out.append("-")
        print(f"| {tenant}:{site} | " + " | ".join(cells_out) + " |")

    print("\n## Speedup: single-task vs multi-tenant (p50 ΣTTFT, off/spec)\n")
    print("| tenant | single speedup | multi speedup | retained under concurrency |")
    print("|---|---|---|---|")
    for t in tenants:
        sp = {}
        for mode in ("single", "multi"):
            off = med(sum_ttft(f"{mode}-off", t))
            spec = med(sum_ttft(f"{mode}-spec", t))
            sp[mode] = off / spec if spec else float("nan")
        gain = lambda x: x - 1.0
        retained = gain(sp["multi"]) / gain(sp["single"]) if gain(sp["single"]) > 0 else float("nan")
        print(f"| {t} | {sp['single']:.2f}x | {sp['multi']:.2f}x | {retained:.0%} |")

    print("\n## Cache hit and TBT\n")
    for cell in CELLS:
        ss = [s for s in serves if s["cell"] == cell]
        cached, prompt = sum(s["cached"] for s in ss), sum(s["prompt"] for s in ss)
        tbt = statistics.mean(tbts[cell]) if tbts.get(cell) else float("nan")
        print(f"- {cell}: serves={len(ss)} hit={cached}/{prompt} ({cached / prompt:.1%}) mean_TBT={tbt:.2f}ms")


if __name__ == "__main__":
    main()
