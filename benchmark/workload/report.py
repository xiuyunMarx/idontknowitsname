#!/usr/bin/env python
"""Summarise a run_schedule.py result directory as markdown (printed and written to summary.md).

    python benchmark/workload/report.py results/burst-k3-g25 [--baseline greedy --candidate dp]

Per cell: server-side loads and waiting (from server.jsonl) and client wall times (from
clients.jsonl). Paired differences between cells are keyed by (trial, order) for wall
time and by trial for the server totals, with a percentile-bootstrap 95% CI."""
import argparse
import json
import os
import random
import statistics
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def boot_ci(diffs, n=10000, seed=0):
    if len(diffs) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--baseline", default="greedy")
    ap.add_argument("--candidate", default="dp")
    args = ap.parse_args()
    clients = load_jsonl(os.path.join(args.dir, "clients.jsonl"))
    server = load_jsonl(os.path.join(args.dir, "server.jsonl"))
    config = json.load(open(os.path.join(args.dir, "config.json")))
    cells = config["cells"]
    lines = [f"# {os.path.basename(os.path.normpath(args.dir))}", "",
             f"schedule `{config['schedule']}` — {config['meta']}", "",
             f"server {config['server']} — git {config['git']} — trials {config['trials']}", ""]

    # ---- per cell ----
    lines += ["## Per cell (mean over trials)", "",
              "| cell | trials | loads | load_s | queue_wait mean/p95 (ms) | ttft_submit mean/p95 (ms) | e2e_srv mean (ms) | sum_wait_s | wall mean/p95 (s) | span (s) | failed |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        srv = [s for s in server if s["cell"] == cell]
        cl = [c for c in clients if c["cell"] == cell and c["rc"] == 0]
        if not srv:
            continue
        m = lambda k, sub=None: statistics.mean((s[k][sub] if sub else s[k]) for s in srv if (s[k][sub] if sub else s[k]) is not None)
        walls = [c["wall_s"] for c in cl]
        lines.append(f"| {cell} | {len(srv)} | {fmt(m('loads'))} | {fmt(m('load_s'))} | "
                     f"{fmt(m('queue_wait_ms', 'mean'))} / {fmt(m('queue_wait_ms', 'p95'))} | "
                     f"{fmt(m('ttft_submit_ms', 'mean'))} / {fmt(m('ttft_submit_ms', 'p95'))} | {fmt(m('e2e_ms', 'mean'))} | "
                     f"{fmt(m('sum_wait_s'))} | {fmt(statistics.mean(walls) if walls else None)} / {fmt(pct(walls, 0.95))} | "
                     f"{fmt(m('wall_s'))} | {sum(s['client_failed'] + s['failed'] for s in srv)} |")
    lines.append("")

    # ---- per program ----
    programs = sorted({c["program"] for c in clients})
    lines += ["## Client wall time per program (mean s)", "", "| program | " + " | ".join(cells) + " |",
              "|---|" + "---|" * len(cells)]
    for p in programs:
        row = []
        for cell in cells:
            ws = [c["wall_s"] for c in clients if c["cell"] == cell and c["program"] == p and c["rc"] == 0]
            row.append(f"{fmt(statistics.mean(ws))} (n={len(ws)})" if ws else "-")
        lines.append(f"| {p} | " + " | ".join(row) + " |")
    lines.append("")

    # ---- paired ----
    def paired(a, b):
        by = defaultdict(dict)
        for c in clients:
            if c["rc"] == 0:
                by[(c["trial"], c["order"])][c["cell"]] = c["wall_s"]
        diffs = [v[b] - v[a] for v in by.values() if a in v and b in v]
        srv = defaultdict(dict)
        for s in server:
            srv[s["trial"]][s["cell"]] = s
        sw = [srv[t][b]["sum_wait_s"] - srv[t][a]["sum_wait_s"] for t in srv if a in srv[t] and b in srv[t]]
        ld = [srv[t][b]["load_s"] - srv[t][a]["load_s"] for t in srv if a in srv[t] and b in srv[t]]
        return diffs, sw, ld

    lines += ["## Paired differences (candidate − baseline; negative = candidate better)", "",
              "| pair | n | wall mean diff (s) | 95% CI | sum_wait_s diff (per trial) | load_s diff (per trial) |", "|---|---|---|---|---|---|"]
    for a, b in [(args.baseline, args.candidate)] + [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)
                                                     if (cells[i], cells[i + 1]) != (args.baseline, args.candidate)]:
        diffs, sw, ld = paired(a, b)
        if not diffs:
            continue
        lo, hi = boot_ci(diffs)
        lines.append(f"| {b} − {a} | {len(diffs)} | {fmt(statistics.mean(diffs), 2)} | [{fmt(lo, 2)}, {fmt(hi, 2)}] | "
                     f"{', '.join(fmt(x) for x in sw)} | {', '.join(fmt(x) for x in ld)} |")
    lines.append("")

    # ---- load orders ----
    lines += ["## Load order per cell (first trial)", ""]
    for cell in cells:
        s = next((s for s in server if s["cell"] == cell), None)
        if s:
            lines.append(f"- **{cell}**: " + " → ".join(f"{m}({t[0]})" for m, t in s["load_order"]))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
