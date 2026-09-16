"""Concurrency sweep of one application on one server per arm; see sweep_<program>.py.

Each level runs c lanes of the program through benchmark.mixed_workload with tag
<prefix>_<arm>_c<c>. Sessions per lane default to inputs // c (every input once).
--fresh restarts the server before every level; otherwise one server serves the
whole sweep after a single warmup. Results: mixed_results/<tag>/ plus
mixed_results/<prefix>_sweep_<arm>.csv with one row per level.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "mixed_results")
COLUMNS = ["c", "sessions", "failed", "jct_p50_s", "jct_p95_s", "jct_mean_s", "ttft_p50_ms", "ttft_p95_ms",
           "ttft_mean_ms", "device_pct", "host_pct", "miss_pct", "predicted_hit", "accuracy", "calls",
           "prompt_tokens_per_call"]


def start_server(arm: str, log: str, host_gb: int) -> None:
    env = dict(os.environ, HOST=str(host_gb), LOG=log)
    subprocess.run(["bash", os.path.join(REPO, "start_server.bash"), arm], cwd=REPO, env=env, check=True)


def stop_server() -> None:
    subprocess.run(["pkill", "-9", "-f", "[s]glang::|[s]erver\\.server"])


def run_level(program: str, tag: str, c: int, sessions: int, warmup: int, log: str) -> dict:
    cmd = [sys.executable, "-m", "benchmark.mixed_workload", "--tag", tag, "--lanes", f"{program}={c}",
           "--sessions", str(sessions), "--warmup", str(warmup), "--server-log", log]
    with open(os.path.join(RESULTS, f"{tag}.out"), "w") as out:
        subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=True)
    with open(os.path.join(RESULTS, tag, "summary.json")) as f:
        row = json.load(f)["programs"][program]
    row["c"] = c
    print(f"[sweep] {tag}: " + " ".join(f"{k}={row.get(k)}" for k in COLUMNS), flush=True)
    return row


def sweep(program: str, prefix: str, inputs: int, levels: str) -> int:
    ap = argparse.ArgumentParser(description=f"concurrency sweep of {program}")
    ap.add_argument("arm", nargs="?", default="ours", choices=["ours", "lru", "lruraw", "kvonly"])
    ap.add_argument("--c", default=levels, help="comma-separated concurrency levels")
    ap.add_argument("--sessions", default="auto", help=f"per lane; auto = {inputs} // c")
    ap.add_argument("--host", type=int, default=8, help="host KV tier, GB")
    ap.add_argument("--fresh", action="store_true", help="fresh server per level")
    a = ap.parse_args()
    cs = [int(x) for x in a.c.split(",")]
    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, f"{prefix}_sweep_{a.arm}.csv")

    log = os.path.join(RESULTS, f"{prefix}_{a.arm}_server.log")
    if not a.fresh:
        start_server(a.arm, log, a.host)
    rows = []
    for c in cs:
        tag = f"{prefix}_{a.arm}_c{c}"
        sessions = max(1, inputs // c) if a.sessions == "auto" else int(a.sessions)
        if a.fresh:
            log = os.path.join(RESULTS, tag, "server.log")
            start_server(a.arm, log, a.host)
        rows.append(run_level(program, tag, c, sessions, warmup=1 if (a.fresh or c == cs[0]) else 0, log=log))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    stop_server()
    print(f"[sweep] {prefix} {a.arm} done: {csv_path}", flush=True)
    return 0
