"""Cold-start evaluation of compile-time-guided speculation.

Every trial boots a fresh server (empty KV cache, no history of any kind), each
tenant runs exactly one workflow, then the server dies. Two modes:

    single   the 7 tenants run strictly one at a time (single-task cold start)
    multi    all 7 tenants start concurrently, staggered 0-3 s (multi-tenant cold start)

Cells are server configurations (see CELLS). `off` is the baseline: vLLM prefix
caching on, speculation off. `spec` is the full system. The rest are ablations.

Pairing: trial k uses the same inputs and the same stagger schedule in every
(mode, cell), so per-trial comparisons pair. Cell order is rotated per trial so
no cell always runs first (counterbalanced). Graph DBs are wiped before every
trial (route graphs accumulate).

Every client result row carries trial/mode/cell/tenant/rc so the aggregator can
drop a whole pair when either side failed or did different work.

Usage:
  python experiments/coldstart_experiment.py --trials 20 --cells off,spec --modes single,multi
  python experiments/coldstart_experiment.py --trials 10 --cells off,spec --modes single --tool-scale 0.5
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

from benchmark.microbench.experiments.run_experiment import ROOT, TENANTS, start_server, stop_server, wait_ready

# name -> extra server flags. Budget profile is loaded whenever --no-budget is absent.
CELLS = {
    "off": ["--no-prefill"],
    "spec": [],                                                # full system
    "spec-idle": ["--no-budget"],                              # speculate only when the engine is empty
    "anchor": ["--spec-features", "none"],                     # successors' invariants only (what a trace could warm)
    "+const": ["--spec-features", "const"],
    "+ret": ["--spec-features", "const,ret"],
    "+toolturn": ["--spec-features", "const,ret,toolturn"],    # = spec minus route probe
    "random": ["--spec-order", "random"],                      # successor order shuffled
    "chunk32": ["--spec-chunk", "32"],                         # smaller idle-time speculative requests:
    "chunk64": ["--spec-chunk", "64"],                         #   a real request arriving waits behind a shorter step
}
MODES = ("single", "multi")


def run_client(name: str, jac_file: str, env_var: str, case: str, model: str, tool_scale: float,
               log_dir: str, results: list, lock: threading.Lock, row: dict) -> None:
    env = {**os.environ, "MODEL": model, "JAC_ROUTE_CACHE_LAYOUT": "1",
           "TOOL_DELAY_SCALE": str(tool_scale), env_var: case,
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
        results.append({**row, "tenant": name, "wall": time.time() - started, "rc": rc})


def run_trial(mode: str, cell: str, trial: int, model: str, tool_scale: float,
              out_root: str, results: list, status: list) -> None:
    out_dir = os.path.join(out_root, mode, cell, f"trial-{trial}")
    os.makedirs(out_dir, exist_ok=True)
    # jac keeps a program's graph DB in a .jac/ directory next to the source.
    for db in glob.glob(os.path.join(ROOT, "jac_programs", ".jac", "data", "*.db")) \
            + glob.glob(os.path.join(ROOT, ".jac", "data", "*.db")):
        os.remove(db)

    log_path = os.path.join(out_dir, "server.log")
    server = start_server(CELLS[cell], log_path, model=model)
    lock = threading.Lock()
    row = {"trial": trial, "mode": mode, "cell": cell, "tool_scale": tool_scale}
    t0 = time.time()
    try:
        wait_ready(log_path, server)
        rng = random.Random(20260823 + trial)  # same schedule in every cell of this trial
        staggers = [rng.uniform(0.0, 3.0) for _ in TENANTS]
        jobs = [(name, f, var, cases[trial % len(cases)]) for name, f, _, var, cases in TENANTS]
        if mode == "multi":
            threads = []
            for (name, f, var, case), delay in zip(jobs, staggers):
                def go(name=name, f=f, var=var, case=case, delay=delay):
                    time.sleep(delay)
                    run_client(name, f, var, case, model, tool_scale, out_dir, results, lock, row)
                threads.append(threading.Thread(target=go))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for name, f, var, case in jobs:
                run_client(name, f, var, case, model, tool_scale, out_dir, results, lock, row)
    finally:
        stop_server(server)

    with open(log_path) as f:
        registered = len(re.findall(r"Registered", f.read()))
    failed = sum(1 for r in results if r["trial"] == trial and r["mode"] == mode
                 and r["cell"] == cell and r["rc"] != 0)
    status.append({**row, "registered": registered, "expected": len(TENANTS),
                   "failed_clients": failed, "seconds": time.time() - t0})
    note = "" if registered == len(TENANTS) else "  WARNING: silent-fallback runs, trial is void"
    print(f"[{mode}/{cell} trial {trial}] registered={registered}/{len(TENANTS)} "
          f"failed={failed} {time.time() - t0:.0f}s{note}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cells", default="off,spec")
    parser.add_argument("--modes", default="single,multi")
    parser.add_argument("--tool-scale", type=float, default=1.0,
                        help="multiplier on every fixed tool/inter-call sleep in the Jac programs")
    parser.add_argument("--model", default=os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results",
                                                      time.strftime("coldstart-%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := [c for c in cells if c not in CELLS]:
        raise SystemExit(f"unknown cells {unknown}; known: {list(CELLS)}")
    if unknown := [m for m in modes if m not in MODES]:
        raise SystemExit(f"unknown modes {unknown}")
    if not os.environ.get("MODEL"):
        os.environ["MODEL"] = args.model  # run_experiment.start_server reads it too
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"trials": args.trials, "cells": cells, "modes": modes, "tool_scale": args.tool_scale,
                   "model": args.model, "cell_flags": {c: CELLS[c] for c in cells}}, f, indent=1)

    results: list = []
    status: list = []
    units = [(m, c) for m in modes for c in cells]
    for trial in range(args.trials):
        k = trial % len(units)  # counterbalance: rotate which (mode, cell) goes first
        for mode, cell in units[k:] + units[:k]:
            run_trial(mode, cell, trial, args.model, args.tool_scale, args.out, results, status)
        with open(os.path.join(args.out, "clients.jsonl"), "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(args.out, "cells.jsonl"), "w") as f:
            for row in status:
                f.write(json.dumps(row) + "\n")
    print(f"results in {args.out}; aggregate with: python experiments/coldstart_aggregate.py {args.out}")


if __name__ == "__main__":
    main()
