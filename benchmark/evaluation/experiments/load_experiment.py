"""Run a blend from `benchmark/evaluation/synthesis_data.py` against the guard
server and compare speculation with prefix caching alone on the same schedule.

The blend is the schedule: which tenant runs, which case index it takes and when
it starts (`start_offset_s`; None = sequential). Every trial of the blend boots a
fresh server that registers exactly `blend.server.programs`, then launches one
`jac run` per entry with `entry.env`. `off` and `spec` execute the same blend,
so workflows pair by (trial, order). The server log's
`Registered ... (pid P) as prog#S` line maps each session back to the client
process the harness launched, which is how a workflow's serves are found.

Usage (from the repository root):
    python benchmark/evaluation/synthesis_data.py --size 12 --trials 3 --arrival poisson --rate 0.5 \
        --out benchmark/evaluation/blends/poisson.json
    python -m benchmark.evaluation.experiments.load_experiment --blend benchmark/evaluation/blends/poisson.json
"""

import argparse
import glob
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict

from benchmark.evaluation.experiments.smoke_experiment import (
    REPO, JAC_LIB, CELLS, ANSI, SERVE, DECODE, PREFILL, stop_server,
)

REGISTERED = re.compile(r"Registered '(\S+)' \(pid (\d+),.*\) as (\S+)#(\d+)")
SERVE_ID = re.compile(r"^(\S+?):(.+?):s(\d+)c(\d+)-")

# Environment every tenant client needs besides the blend entry's own selector.
CLIENT_ENV = {"JAC_ROUTE_CACHE_LAYOUT": "1", "SQL_FORCE_TURNS": "1", "LD_LIBRARY_PATH": JAC_LIB}


# ------------------------------------------------------------------ running

def start_server(programs, extra, log_path, model, workers):
    cmd = [sys.executable, os.path.join(REPO, "start_server.py"), "--model", model, "--workers", str(workers)]
    for program in programs:
        cmd += ["--program", program]
    cmd += extra
    log = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)


def wait_ready(log_path, proc, expected, timeout=900.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            with open(log_path) as f:
                if f.read().count("listening on") >= expected:
                    return
        except OSError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server not ready within {timeout}s, see {log_path}")


def run_entry(entry, args, out_dir, t0, results, lock, trial, cell):
    env = {**os.environ, "MODEL": args.model, "TOOL_DELAY_SCALE": str(args.tool_scale), **CLIENT_ENV, **entry["env"]}
    started = time.time()
    with open(os.path.join(out_dir, f"client-{entry['order']:02d}-{entry['tenant']}.log"), "w") as log:
        proc = subprocess.Popen(["jac", "run", entry["jac_file"]], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
    with lock:
        results.append({"trial": trial, "cell": cell, "order": entry["order"], "tenant": entry["tenant"],
                        "program": entry["program_name"], "case_index": entry["case_index"],
                        "case_id": entry.get("case_id"), "strata": entry.get("strata", {}),
                        "pid": proc.pid, "arrival": entry["start_offset_s"], "start": started - t0,
                        "end": time.time() - t0, "wall": time.time() - started, "rc": rc})


def run_trial(blend, trial_spec, cell, args, out_root, results, status):
    trial = trial_spec["trial"]
    entries = sorted(trial_spec["entries"], key=lambda e: e["order"])
    out_dir = os.path.join(out_root, cell, f"trial-{trial}")
    os.makedirs(out_dir, exist_ok=True)
    for db in glob.glob(os.path.join(REPO, ".jac", "data", "*.db")) + glob.glob(os.path.join(REPO, "benchmark", "evaluation", "programs", ".jac", "data", "*.db")):
        os.remove(db)  # graph DBs accumulate across runs
    programs = blend["server"]["programs"]
    log_path = os.path.join(out_dir, "server.log")
    extra = CELLS[cell] + (["--cache-isolation", "instance"] if args.isolation == "instance" else [])
    server = start_server(programs, extra, log_path, args.model, args.workers)
    lock = threading.Lock()
    t_trial = time.time()
    try:
        wait_ready(log_path, server, len(programs))
        t0 = time.time()
        sequential = any(e["start_offset_s"] is None for e in entries)
        if sequential:
            for entry in entries:
                run_entry(entry, args, out_dir, t0, results, lock, trial, cell)
        else:
            def go(entry):
                time.sleep(max(0.0, t0 + entry["start_offset_s"] - time.time()))
                run_entry(entry, args, out_dir, t0, results, lock, trial, cell)
            threads = [threading.Thread(target=go, args=(entry,)) for entry in entries]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
    finally:
        stop_server(server)
    rows = [r for r in results if r["trial"] == trial and r["cell"] == cell]
    failed = sum(1 for r in rows if r["rc"] != 0)
    status.append({"trial": trial, "cell": cell, "workflows": len(rows), "failed_clients": failed,
                   "makespan": max(r["end"] for r in rows), "seconds": time.time() - t_trial})
    print(f"[{cell} trial {trial}] workflows={len(rows)} failed={failed} makespan={max(r['end'] for r in rows):.0f}s "
          f"{time.time() - t_trial:.0f}s", flush=True)


# ------------------------------------------------------------- aggregating

def parse_server(log):
    """-> sessions {(program, session) -> pid}, serves {(program, session) -> [serve dicts in order]}, tbts, spec"""
    sessions, serves, tbts, spec = {}, defaultdict(list), [], [0, 0, 0.0, 0]
    for line in open(log, errors="replace"):
        line = ANSI.sub("", line)
        if m := REGISTERED.search(line):
            sessions[(m.group(3), int(m.group(4)))] = int(m.group(2))
        elif m := SERVE.search(line):
            sid = SERVE_ID.match(m.group(1))
            if sid is None:
                continue
            serves[(sid.group(1), int(sid.group(3)))].append({
                "site": sid.group(2).split("@")[0], "ttft": float(m.group(2)),
                "cached": int(m.group(3)), "prompt": int(m.group(4))})
        elif m := DECODE.search(line):
            if (n := int(m.group(2))) > 1:
                tbts.append(float(m.group(1)) / (n - 1))
        elif m := PREFILL.search(line):
            spec[0] += 1; spec[1] += int(m.group(1)); spec[2] += float(m.group(2)); spec[3] += 1 if m.group(3) else 0
    return sessions, serves, tbts, spec


def mean(v):
    return statistics.mean(v) if v else float("nan")


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def boot_ci(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def concurrency(rows, step=0.5):
    """Mean and peak number of workflows in flight, sampled every `step` s."""
    end = max(r["end"] for r in rows)
    samples, t = [], 0.0
    while t <= end:
        samples.append(sum(1 for r in rows if r["start"] <= t < r["end"]))
        t += step
    return mean(samples), max(samples)


def aggregate(root, base="off"):
    clients = [json.loads(l) for l in open(os.path.join(root, "clients.jsonl"))]
    config = json.load(open(os.path.join(root, "config.json")))
    meta = config["blend_meta"]
    cells = [c for c in CELLS if any(r["cell"] == c for r in clients)]
    trials = sorted({r["trial"] for r in clients})
    tenants = list(meta["per_trial_counts"])
    wf, tbts, spec, calls = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    sites = defaultdict(lambda: defaultdict(list))
    unmatched = 0
    for cell in cells:
        for trial in trials:
            log = os.path.join(root, cell, f"trial-{trial}", "server.log")
            if not os.path.exists(log):
                continue
            sessions, serves, tb, sp = parse_server(log)
            tbts[cell] += tb; spec[cell].append(sp)
            by_pid = {r["pid"]: r for r in clients if r["cell"] == cell and r["trial"] == trial}
            for (program, sess), pid in sessions.items():
                r = by_pid.get(pid)
                ss = serves.get((program, sess), [])
                if r is None:
                    unmatched += 1
                    continue
                if not ss:
                    continue
                wf[(cell, trial, r["order"])] = {**r, "calls": len(ss), "uncached": sum(s["prompt"] - s["cached"] for s in ss),
                                                 "sum_ttft": sum(s["ttft"] for s in ss), "seq": [s["site"] for s in ss],
                                                 "p95": pct([s["ttft"] for s in ss], 0.95)}
                calls[cell] += [s["ttft"] for s in ss]
                for s in ss:
                    sites[f"{program}:{s['site']}"][cell].append(s)
    desc = {"uniform": f"uniform over {meta.get('window_s')}s", "burst": f"burst within {meta.get('jitter_s')}s",
            "poisson": f"poisson {meta.get('rate_per_s')}/s"}.get(meta.get("arrival"), "sequential")
    out = [f"# Multi-tenant blend: `{os.path.basename(root)}`", "",
           f"blend: {config['blend']} — {meta['size']} workflows/trial, mode={meta['mode']} ({desc}), "
           f"ratio {dict(zip(meta['tenants'], meta['blend_ratio']))}, seed {meta['seed']}; "
           f"workers={config['workers']}, model={config['model']}, cache isolation={config.get('isolation', 'none')}"
           + (f"; **{unmatched} sessions could not be mapped to a client**" if unmatched else ""), ""]
    out += ["## per cell", "", "| cell | trials | workflows (rc=0) | mean / peak concurrency | makespan s | warm fraction | call TTFT p50 / p95 / p99 ms | mean TBT ms | spec prefills/trial | aborted/trial |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        rows = [r for r in clients if r["cell"] == cell]
        ok = [r for r in rows if r["rc"] == 0]
        conc = [concurrency([r for r in rows if r["trial"] == t]) for t in trials if any(r["trial"] == t for r in rows)]
        prompt = sum(s["prompt"] for site in sites.values() for s in site[cell]); cached = sum(s["cached"] for site in sites.values() for s in site[cell])
        mk = [max(r["end"] for r in rows if r["trial"] == t) for t in trials if any(r["trial"] == t for r in rows)]
        out.append(f"| {cell} | {len(trials)} | {len(ok)}/{len(rows)} | {mean([c[0] for c in conc]):.1f} / {max(c[1] for c in conc)} | {mean(mk):.0f} | "
                   f"{100 * cached / max(prompt, 1):.1f}% | {pct(calls[cell], .5):.0f} / {pct(calls[cell], .95):.0f} / {pct(calls[cell], .99):.0f} | "
                   f"{mean(tbts[cell]):.2f} | {mean([s[0] for s in spec[cell]]):.0f} | {mean([s[3] for s in spec[cell]]):.0f} |")
    out.append("")
    for cell in cells:
        if cell == base:
            continue
        out += [f"## {cell} vs {base}: paired per workflow (same trial, same order)", "",
                "| tenant | pairs (same seq / all) | uncached tok/workflow base→cell | Δuncached | ΣTTFT ms base→cell | ΔΣTTFT [95% CI] | ratio | per-call p95 base→cell | wall s base→cell |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name in tenants + ["all"]:
            pairs = []
            for (c, t, order), b in wf.items():
                if c != base or b["rc"] != 0 or (name != "all" and b["tenant"] != name):
                    continue
                s = wf.get((cell, t, order))
                if s is None or s["rc"] != 0:
                    continue
                pairs.append((b, s))
            if not pairs:
                continue
            same = sum(1 for b, s in pairs if b["seq"] == s["seq"])
            dt = [b["sum_ttft"] - s["sum_ttft"] for b, s in pairs]
            du = [b["uncached"] - s["uncached"] for b, s in pairs]
            lo, hi = boot_ci(dt) if len(dt) > 1 else (float("nan"), float("nan"))
            ratio = sum(b["sum_ttft"] for b, _ in pairs) / max(sum(s["sum_ttft"] for _, s in pairs), 1e-9)
            bold = "**" if name == "all" else ""
            out.append(f"| {bold}{name}{bold} | {same} / {len(pairs)} | {mean([b['uncached'] for b, _ in pairs]):.0f}→{mean([s['uncached'] for _, s in pairs]):.0f} | "
                       f"{bold}{mean(du):+.0f}{bold} | {mean([b['sum_ttft'] for b, _ in pairs]):.0f}→{mean([s['sum_ttft'] for _, s in pairs]):.0f} | "
                       f"{bold}{mean(dt):+.1f} [{lo:+.1f}, {hi:+.1f}]{bold} | {bold}{ratio:.2f}x{bold} | "
                       f"{mean([b['p95'] for b, _ in pairs]):.0f}→{mean([s['p95'] for _, s in pairs]):.0f} | "
                       f"{mean([b['wall'] for b, _ in pairs]):.1f}→{mean([s['wall'] for _, s in pairs]):.1f} |")
        out.append("")
    out += ["## per callsite (mean uncached tokens / mean TTFT ms)", "", "| callsite | n | " + " | ".join(cells) + " |", "|---|---|" + "---|" * len(cells)]
    for site in sorted(sites):
        n = sum(len(v) for v in sites[site].values())
        cols = [f"{mean([s['prompt'] - s['cached'] for s in sites[site][c]]):.0f} / {mean([s['ttft'] for s in sites[site][c]]):.0f}" if sites[site][c] else "—" for c in cells]
        out.append(f"| {site} | {n} | " + " | ".join(cols) + " |")
    out.append("")
    text = "\n".join(out)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", help="blend JSON from benchmark/evaluation/synthesis_data.py")
    parser.add_argument("--cells", default="off,spec")
    parser.add_argument("--workers", type=int, default=16, help="server generation slots")
    parser.add_argument("--isolation", choices=["none", "instance"], default="none",
                        help="instance: server runs with --cache-isolation instance in every cell "
                             "(no prefix-cache sharing across workflow instances)")
    parser.add_argument("--tool-scale", type=float, default=1.0)
    parser.add_argument("--model", default=os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        print(aggregate(args.out))
        return
    if not args.blend:
        parser.error("--blend is required")
    blend = json.load(open(args.blend))
    out = args.out or os.path.join(REPO, "benchmark", "evaluation", "experiments", "results",
                                   "load-" + os.path.splitext(os.path.basename(args.blend))[0]
                                   + ("-iso" if args.isolation == "instance" else ""))
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "cells": cells, "blend_meta": blend["meta"], "programs": blend["server"]["programs"]}, f, indent=1)
    results, status = [], []
    for k, trial_spec in enumerate(blend["trials"]):
        rot = k % len(cells)
        for cell in cells[rot:] + cells[:rot]:
            run_trial(blend, trial_spec, cell, args, out, results, status)
            with open(os.path.join(out, "clients.jsonl"), "w") as f:
                for row in results:
                    f.write(json.dumps(row) + "\n")
            with open(os.path.join(out, "cells.jsonl"), "w") as f:
                for row in status:
                    f.write(json.dumps(row) + "\n")
    print(aggregate(out))
    print(f"results in {out}")


if __name__ == "__main__":
    main()
