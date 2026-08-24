"""Cold-start validation: compile-time-guided speculation at its target regime.

Two modes x two conditions, N trials each. Every trial boots a fresh server
(empty KV cache), each tenant runs exactly one round, then the server dies:

    multi-off / multi-spec     all 7 tenants started concurrently (staggered 0-3s)
    single-off / single-spec   the same 7 tenants run strictly one at a time

spec = global-idle speculation (--no-budget), off = --no-prefill. Trial k uses
the same inputs and the same stagger in all four cells, so comparisons pair.
Graph DBs are wiped before every trial (route graphs accumulate).

Usage: python experiments/coldstart_experiment.py [--trials 8] [--model M] [--out DIR]
Aggregate with: python experiments/coldstart_aggregate.py <out>
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import threading
import time

from run_experiment import ROOT, TENANTS, start_server, stop_server, wait_ready

CELLS = {
    "multi-off": ("multi", ["--no-prefill"]),
    "multi-spec": ("multi", ["--no-budget"]),
    "single-off": ("single", ["--no-prefill"]),
    "single-spec": ("single", ["--no-budget"]),
}


def run_client(name: str, jac_file: str, env_var: str, case: str, model: str,
               log_dir: str, results: list, lock: threading.Lock, trial: int) -> None:
    env = {**os.environ, "MODEL": model, "JAC_ROUTE_CACHE_LAYOUT": "1",
           env_var: case,
           "LD_LIBRARY_PATH": "/home/xiaoyu/miniconda3/envs/jaseci/lib"}
    started = time.time()
    with open(os.path.join(log_dir, f"client-{name}.log"), "w") as log:
        try:
            proc = subprocess.run(["jac", "run", f"jac_programs/{jac_file}"], cwd=ROOT,
                                  env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    with lock:
        results.append({"trial": trial, "tenant": name, "wall": time.time() - started, "rc": rc})


def run_trial(cell: str, trial: int, model: str, out_root: str, results: list) -> None:
    mode, extra = CELLS[cell]
    out_dir = os.path.join(out_root, cell, f"trial-{trial}")
    os.makedirs(out_dir, exist_ok=True)
    for db in glob.glob(os.path.join(ROOT, ".jac", "data", "*.db")):
        os.remove(db)

    log_path = os.path.join(out_dir, "server.log")
    server = start_server(extra, log_path, model=model)
    lock = threading.Lock()
    try:
        wait_ready(log_path, server)
        rng = random.Random(20260823 + trial)  # same schedule in every cell of this trial
        staggers = [rng.uniform(0.0, 3.0) for _ in TENANTS]
        jobs = [(name, f, var, cases[trial % len(cases)])
                for name, f, _, var, cases in TENANTS]
        if mode == "multi":
            threads = []
            for (name, f, var, case), delay in zip(jobs, staggers):
                def go(name=name, f=f, var=var, case=case, delay=delay):
                    time.sleep(delay)
                    run_client(name, f, var, case, model, out_dir, results, lock, trial)
                threads.append(threading.Thread(target=go))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for name, f, var, case in jobs:
                run_client(name, f, var, case, model, out_dir, results, lock, trial)
    finally:
        stop_server(server)

    with open(log_path) as f:
        registered = len(re.findall(r"Registered", f.read()))
    ok = registered == len(TENANTS)
    print(f"[{cell} trial {trial}] registered={registered}/{len(TENANTS)}"
          + ("" if ok else "  WARNING: silent-fallback runs, trial is void"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--model", default=os.environ.get("MODEL") or "Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results",
                                                      time.strftime("coldstart-%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results: list = []
    for trial in range(args.trials):
        for cell in CELLS:
            run_trial(cell, trial, args.model, args.out, results)
        with open(os.path.join(args.out, "clients.jsonl"), "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
    print(f"results in {args.out}; aggregate with: python experiments/coldstart_aggregate.py {args.out}")


if __name__ == "__main__":
    main()
