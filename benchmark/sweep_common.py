"""Concurrency sweep of one application against an already running server; see sweep_<program>.py.

Each level runs c lanes of the program through benchmark.mixed_workload with tag
<prefix>_<arm>_c<c>. Sessions per lane default to ceil(inputs / c); the workload
driver stops when the input set is exhausted, so every input is measured once.
The first level warms the program up with one sequential session.

The server is started outside (start_server.bash <arm>; benchmark/run_sweeps.bash runs
the four sweeps of one arm on one server). `arm` here only names the outputs:
mixed_results/sweeps/<prefix>_sweep_<arm>.csv, one row per level, and
mixed_results/sweeps/logs/<tag>.out per level. The driver's own files stay under
mixed_results/cache/<tag>/.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

from benchmark.mixed_workload import RESULTS as DRIVER_RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.environ.get("SWEEP_OUT") or os.path.join(HERE, "mixed_results", "sweeps")   # SWEEP_OUT: alternate output dir
COLUMNS = ["c", "sessions", "failed", "jct_p50_s", "jct_p95_s", "jct_mean_s", "ttft_p50_ms", "ttft_p95_ms",
           "ttft_mean_ms", "device_pct", "host_pct", "miss_pct", "predicted_hit", "accuracy", "calls",
           "prompt_tokens_per_call"]


def run_level(program: str, tag: str, c: int, sessions: int, warmup: int, server_log: str) -> dict:
    cmd = [sys.executable, "-m", "benchmark.mixed_workload", "--tag", tag, "--lanes", f"{program}={c}",
           "--sessions", str(sessions), "--warmup", str(warmup), "--server-log", server_log]
    with open(os.path.join(RESULTS, "logs", f"{tag}.out"), "w") as out:
        subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=True)
    with open(os.path.join(DRIVER_RESULTS, tag, "summary.json")) as f:
        row = json.load(f)["programs"][program]
    row["c"] = c
    print(f"[sweep] {tag}: " + " ".join(f"{k}={row.get(k)}" for k in COLUMNS), flush=True)
    return row


def sweep(program: str, prefix: str, inputs: int, levels: str) -> int:
    ap = argparse.ArgumentParser(description=f"concurrency sweep of {program} on a running server")
    ap.add_argument("arm", nargs="?", default="ours", choices=["ours", "relayout", "vanilla", "kvonly", "cachescout", "continuum", "kvflow", "kvflow_relayout", "continuum_relayout", "cachescout_relayout"],
                    help="the arm the running server was started as; names the outputs")
    ap.add_argument("--c", default=levels, help="comma-separated concurrency levels")
    ap.add_argument("--sessions", default="auto",
                    help=f"maximum sessions per lane; auto = ceil({inputs} / c), covering each input once")
    ap.add_argument("--server-log", default="", help="the server's stdout, for cache and prediction stats")
    ap.add_argument("--dry-run", action="store_true", help="print the sweep plan without running it")
    a = ap.parse_args()
    try:
        cs = [int(x.strip()) for x in a.c.split(",") if x.strip()]
    except ValueError as e:
        ap.error(f"--c must be a comma-separated list of integers: {e}")
    if not cs or any(c <= 0 for c in cs):
        ap.error("--c values must be positive integers")
    if a.sessions != "auto":
        try:
            fixed_sessions = int(a.sessions)
        except ValueError:
            ap.error("--sessions must be 'auto' or a positive integer")
        if fixed_sessions <= 0:
            ap.error("--sessions must be positive for a concurrency sweep")
    else:
        fixed_sessions = 0

    plan = [(c, math.ceil(inputs / c) if a.sessions == "auto" else fixed_sessions) for c in cs]
    if a.dry_run:
        for c, sessions in plan:
            print(f"[sweep-plan] program={program} arm={a.arm} c={c} "
                  f"sessions_per_lane={sessions} max_sessions={c * sessions} inputs={inputs}")
        return 0

    os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
    csv_path = os.path.join(RESULTS, f"{prefix}_sweep_{a.arm}.csv")
    server_log = a.server_log or os.path.join(RESULTS, "logs", f"sweep_{a.arm}_server.log")

    rows = []
    for i, (c, sessions) in enumerate(plan):
        tag = f"{prefix}_{a.arm}_c{c}"
        rows.append(run_level(program, tag, c, sessions, warmup=1 if i == 0 else 0,
                              server_log=server_log))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"[sweep] {prefix} {a.arm} done: {csv_path}", flush=True)
    return 0
