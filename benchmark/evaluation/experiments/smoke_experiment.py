"""Smoke evaluation of the three evaluation tenants (HoVer, deep_search,
intercode_sql): what speculation recovers when each program starts alone and
when the three start concurrently, on a fresh server every trial.

    single   the three tenants run one after another
    multi    the three start within a 2 s window

Cells are server configurations (`off` = prefix caching only, `spec` = full
system, `spec-idle` = speculate only while the engine is empty). Trial k feeds
task k of each tenant's dataset in every (mode, cell); cell order rotates per
trial. Every tenant has its own input index so the served callsite sequence is
whatever the model's own choices produce; the aggregate reports pooled per-
callsite numbers and marks how many (trial, tenant) pairs served identical
sequences.

Usage (from the repository root):
    python -m benchmark.evaluation.experiments.smoke_experiment --trials 3 --cells off,spec --modes single,multi
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROGRAMS = os.path.join("benchmark", "evaluation", "programs")
JAC_LIB = "/home/xiaoyu/miniconda3/envs/jaseci/lib"

# name, source, port, index env var, dataset size, extra client env
TENANTS = [
    ("hover", "HoVer.jac", 8971, "HOVER_CLAIM_INDEX", 30, {}),
    ("deep_search", "deep_search.jac", 8972, "DEEP_TASK_INDEX", 30, {}),
    ("intercode_sql", "intercode_sql.jac", 8973, "SQL_TASK_INDEX", 30, {"SQL_FORCE_TURNS": "1"}),
]
CELLS = {"off": ["--no-prefill"], "spec": [], "spec-idle": ["--no-budget"]}
MODES = ("single", "multi")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SERVE = re.compile(r"\[serve\] (\S+) duration_ms=([\d.]+) cached_tokens=(\d+) prompt_tokens=(\d+)")
DECODE = re.compile(r"\[decode\] \S+ decode_ms=([\d.]+) output_tokens=(\d+)")
PREFILL = re.compile(r"\[prefill\] \S+ cost_tokens=(\d+) duration_ms=([\d.]+)( aborted=1)?")


# ------------------------------------------------------------------ running

def start_server(extra, log_path, model):
    cmd = [sys.executable, os.path.join(REPO, "start_server.py"), "--model", model, "--workers", "8"]
    for name, src, port, *_ in TENANTS:
        cmd += ["--program", f"{name}:{PROGRAMS}/{src}:{port}"]
    cmd += extra
    log = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)


def stop_server(server):
    import signal
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=30)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)


def wait_ready(log_path, proc, timeout=900.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            with open(log_path) as f:
                if f.read().count("listening on") >= len(TENANTS):
                    return
        except OSError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server not ready within {timeout}s, see {log_path}")


def run_client(name, src, env_var, index, extra_env, model, tool_scale, log_dir, results, lock, row):
    env = {**os.environ, "MODEL": model, "JAC_ROUTE_CACHE_LAYOUT": "1",
           "TOOL_DELAY_SCALE": str(tool_scale), env_var: str(index),
           "LD_LIBRARY_PATH": JAC_LIB, **extra_env}
    started = time.time()
    with open(os.path.join(log_dir, f"client-{name}.log"), "w") as log:
        try:
            rc = subprocess.run(["jac", "run", f"{PROGRAMS}/{src}"], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    with lock:
        results.append({**row, "tenant": name, "index": index, "wall": time.time() - started, "rc": rc})


def run_trial(mode, cell, trial, model, tool_scale, out_root, results, status):
    out_dir = os.path.join(out_root, mode, cell, f"trial-{trial}")
    os.makedirs(out_dir, exist_ok=True)
    for db in glob.glob(os.path.join(REPO, PROGRAMS, ".jac", "data", "*.db")):
        os.remove(db)  # graph DBs accumulate across runs
    log_path = os.path.join(out_dir, "server.log")
    server = start_server(CELLS[cell], log_path, model)
    lock = threading.Lock()
    row = {"trial": trial, "mode": mode, "cell": cell, "tool_scale": tool_scale}
    t0 = time.time()
    try:
        wait_ready(log_path, server)
        rng = random.Random(20260824 + trial)
        staggers = [rng.uniform(0.0, 2.0) for _ in TENANTS]
        jobs = [(name, src, var, trial % size, extra) for name, src, _, var, size, extra in TENANTS]
        if mode == "multi":
            threads = []
            for (name, src, var, index, extra), delay in zip(jobs, staggers):
                def go(name=name, src=src, var=var, index=index, extra=extra, delay=delay):
                    time.sleep(delay)
                    run_client(name, src, var, index, extra, model, tool_scale, out_dir, results, lock, row)
                threads.append(threading.Thread(target=go))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for name, src, var, index, extra in jobs:
                run_client(name, src, var, index, extra, model, tool_scale, out_dir, results, lock, row)
    finally:
        stop_server(server)
    failed = sum(1 for r in results if r["trial"] == trial and r["mode"] == mode and r["cell"] == cell and r["rc"] != 0)
    status.append({**row, "failed_clients": failed, "seconds": time.time() - t0})
    print(f"[{mode}/{cell} trial {trial}] failed={failed} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------- aggregating

def parse(root):
    serves, tbts, spec = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0, 0.0, 0])
    for mode in MODES:
        for cell in CELLS:
            for tdir in sorted(glob.glob(os.path.join(root, mode, cell, "trial-*"))):
                log = os.path.join(tdir, "server.log")
                if not os.path.exists(log):
                    continue
                key = (mode, cell, int(tdir.rsplit("-", 1)[1]))
                for line in open(log, errors="replace"):
                    line = ANSI.sub("", line)
                    if m := SERVE.search(line):
                        tenant, site = (m.group(1).split(":", 2) + ["?", "?"])[:2]
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
                        spec[key][3] += 1 if m.group(3) else 0
    return serves, tbts, spec


def load_rc(root):
    rc = {}
    path = os.path.join(root, "clients.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            rc[(r["mode"], r["cell"], r["trial"], r["tenant"])] = r["rc"]
    return rc


def mean(v):
    return statistics.mean(v) if v else float("nan")


def boot_ci(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def aggregate(root, base="off"):
    serves, tbts, spec = parse(root)
    rc = load_rc(root)
    cells = [c for c in CELLS if any(k[1] == c for k in serves)]
    trials = sorted({k[2] for k in serves})
    out = [f"# Evaluation-tenant smoke: `{os.path.basename(root)}`", "",
           f"trials={trials}; cells={cells}", ""]
    for mode in MODES:
        if not any(k[0] == mode for k in serves):
            continue
        out += [f"## {mode}", "", "### per-cell totals", "",
                "| cell | trials | serves | warm fraction | mean TBT ms | spec prefills/trial | aborted/trial | spec tokens/trial |",
                "|---|---|---|---|---|---|---|---|"]
        for cell in cells:
            keys = [k for k in serves if k[0] == mode and k[1] == cell]
            ss = [s for k in keys for s in serves[k]]
            prompt = sum(s["prompt"] for s in ss); cached = sum(s["cached"] for s in ss)
            sp = [spec[k] for k in keys]
            out.append(f"| {cell} | {len(keys)} | {len(ss)} | {100 * cached / max(prompt, 1):.1f}% | "
                       f"{mean(tbts[(mode, cell)]):.2f} | {mean([s[0] for s in sp]):.0f} | "
                       f"{mean([s[3] for s in sp]):.0f} | {mean([s[1] for s in sp]):.0f} |")
        out.append("")
        # per tenant per cell: workflows that exited 0
        out += ["### per tenant (workflows with rc=0; uncached = prompt − cached, summed over the workflow)", "",
                "| tenant | cell | n | calls/workflow | uncached tok/workflow | ΣTTFT ms | wall s |", "|---|---|---|---|---|---|---|"]
        per = {}
        for name, *_ in TENANTS:
            for cell in cells:
                rows = []
                for k in serves:
                    if k[0] != mode or k[1] != cell:
                        continue
                    if rc.get((mode, cell, k[2], name)) != 0:
                        continue
                    ss = [s for s in serves[k] if s["tenant"] == name]
                    if ss:
                        rows.append((k[2], len(ss), sum(s["prompt"] - s["cached"] for s in ss),
                                     sum(s["ttft"] for s in ss), [s["site"] for s in ss]))
                per[(name, cell)] = rows
                walls = [json.loads(l)["wall"] for l in open(os.path.join(root, "clients.jsonl"))
                         if (r := json.loads(l))["mode"] == mode and r["cell"] == cell and r["tenant"] == name and r["rc"] == 0] \
                    if os.path.exists(os.path.join(root, "clients.jsonl")) else []
                out.append(f"| {name} | {cell} | {len(rows)} | {mean([r[1] for r in rows]):.1f} | "
                           f"{mean([r[2] for r in rows]):.0f} | {mean([r[3] for r in rows]):.0f} | {mean(walls):.1f} |")
        out.append("")
        # paired deltas vs base
        for cell in cells:
            if cell == base:
                continue
            out += [f"### {mode}: {cell} vs {base}, paired per (trial, tenant)", "",
                    "| tenant | pairs (same sequence / all) | Δuncached tok/workflow (base−cell) | ΔΣTTFT ms (base−cell) | ΣTTFT ratio |",
                    "|---|---|---|---|---|"]
            all_tok, all_ttft, all_b, all_c = [], [], [], []
            for name, *_ in TENANTS:
                b = {r[0]: r for r in per.get((name, base), [])}
                c = {r[0]: r for r in per.get((name, cell), [])}
                common = sorted(set(b) & set(c))
                same = [t for t in common if b[t][4] == c[t][4]]
                dt = [b[t][2] - c[t][2] for t in common]
                dl = [b[t][3] - c[t][3] for t in common]
                all_tok += dt; all_ttft += dl
                all_b += [b[t][3] for t in common]; all_c += [c[t][3] for t in common]
                if common:
                    ratio = sum(b[t][3] for t in common) / max(sum(c[t][3] for t in common), 1e-9)
                    out.append(f"| {name} | {len(same)} / {len(common)} | {mean(dt):+.0f} | {mean(dl):+.1f} | {ratio:.2f}x |")
            if all_tok:
                lo, hi = boot_ci(all_ttft) if len(all_ttft) > 1 else (float('nan'), float('nan'))
                out.append(f"| **all** | | **{mean(all_tok):+.0f}** | **{mean(all_ttft):+.1f}** [{lo:+.1f}, {hi:+.1f}] | "
                           f"**{sum(all_b) / max(sum(all_c), 1e-9):.2f}x** |")
            out.append("")
        # per callsite
        out += [f"### {mode}: per callsite (mean uncached tokens / mean TTFT ms)", "",
                "| callsite | n | " + " | ".join(cells) + " |", "|---|---|" + "---|" * len(cells)]
        sites = defaultdict(lambda: defaultdict(list))
        for k, ss in serves.items():
            if k[0] != mode:
                continue
            for s in ss:
                sites[f"{s['tenant']}:{s['site']}"][k[1]].append(s)
        for site in sorted(sites):
            n = sum(len(v) for v in sites[site].values())
            cols = [f"{mean([s['prompt'] - s['cached'] for s in sites[site][c]]):.0f} / {mean([s['ttft'] for s in sites[site][c]]):.0f}"
                    if sites[site][c] else "—" for c in cells]
            out.append(f"| {site} | {n} | " + " | ".join(cols) + " |")
        out.append("")
    text = "\n".join(out)
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cells", default="off,spec")
    parser.add_argument("--modes", default="single,multi")
    parser.add_argument("--tool-scale", type=float, default=1.0)
    parser.add_argument("--model", default=os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default=os.path.join(REPO, "benchmark", "evaluation", "experiments", "results",
                                                      time.strftime("smoke-%Y%m%d-%H%M%S")))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        print(aggregate(args.out))
        return
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"trials": args.trials, "cells": cells, "modes": modes, "tool_scale": args.tool_scale,
                   "model": args.model, "tenants": [t[0] for t in TENANTS]}, f, indent=1)
    results, status = [], []
    units = [(m, c) for m in modes for c in cells]
    for trial in range(args.trials):
        k = trial % len(units)
        for mode, cell in units[k:] + units[:k]:
            run_trial(mode, cell, trial, args.model, args.tool_scale, args.out, results, status)
            with open(os.path.join(args.out, "clients.jsonl"), "w") as f:
                for row in results:
                    f.write(json.dumps(row) + "\n")
            with open(os.path.join(args.out, "cells.jsonl"), "w") as f:
                for row in status:
                    f.write(json.dumps(row) + "\n")
    print(aggregate(args.out))
    print(f"results in {args.out}")


if __name__ == "__main__":
    main()
