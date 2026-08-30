#!/usr/bin/env python
"""Replay a schedule (gen_schedule.py) against a running start_server.py under each
planner, so the planners are compared on identical arrivals from identical cold state.

    python benchmark/workload/run_schedule.py --schedule benchmark/workload/schedules/burst-k3-g25.json \
        --cells fifo,greedy,dp --trials 2 --out results/burst-k3-g25

Once, before the first cell: a sequential warmup run of every program (jac compile cache;
also primes the server's branch predictor), then `calibrate` (every engine loaded once
from SSD and once from host so the cost table is measured, not guessed) and `learn_costs
= false` so every cell plans on the same table. Per cell: `reset` cold (every engine to
SSD), then one `jac run` per entry at its offset. Cells alternate direction across trials
to cancel drift. Writes clients.jsonl (one line per run), server.jsonl (the server's
stats per cell, with per-request timings) and config.json."""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from send_requests import JAC_ENV, _setting, control  # noqa: E402


async def run_entry(entry: dict, t0: float, trial: int, cell: str, out, verbose: bool) -> dict:
    delay = t0 + entry["start_offset_s"] - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    env = {**os.environ, **JAC_ENV, entry["env"]: str(entry["case_index"])}
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec("jac", "run", entry["program"], env=env, cwd=ROOT,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    raw, _ = await proc.communicate()
    end = time.perf_counter()
    rec = {"trial": trial, "cell": cell, "order": entry["order"], "program": os.path.basename(entry["program"]),
           "case_index": entry["case_index"], "start_offset_s": entry["start_offset_s"],
           "actual_start_s": round(start - t0, 3), "end_s": round(end - t0, 3), "wall_s": round(end - start, 3),
           "rc": proc.returncode}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"[{cell} t{trial}] #{entry['order']:<3} {rec['program']:<18} idx={entry['case_index']:<2} "
          f"start={rec['actual_start_s']:7.1f}s wall={rec['wall_s']:6.1f}s rc={proc.returncode}", flush=True)
    if proc.returncode != 0 or verbose:
        sys.stderr.write(raw.decode(errors="replace")[-1500:] + "\n")
    return rec


async def reset(cell: str, retries: int = 30) -> dict:
    """The server refuses to reset while speculation is still draining; wait it out."""
    for i in range(retries):
        try:
            return await control({"set": {"planner": cell}, "reset": True, "cold": True})
        except RuntimeError as e:
            if "in flight" not in str(e) or i == retries - 1:
                raise
            await asyncio.sleep(1.0)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--cells", default="fifo,greedy,dp")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true", help="keep the server's current cost table")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="server settings for every cell")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.schedule) as f:
        sched = json.load(f)
    entries, cells = sched["entries"], args.cells.split(",")
    os.makedirs(args.out, exist_ok=True)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    config = {"schedule": args.schedule, "meta": sched["meta"], "cells": cells, "trials": args.trials,
              "set": args.set, "git": rev, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not args.no_warmup:
        for program in sorted({e["program"] for e in entries}):
            print(f"[warmup] {program}", flush=True)
            env = {**os.environ, **JAC_ENV, next(e["env"] for e in entries if e["program"] == program): "0"}
            subprocess.run(["jac", "run", program], env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    settings = dict(_setting(kv) for kv in args.set)
    if settings:
        await control({"set": settings})
    if not args.no_calibrate:
        print("[calibrate] measuring load times", flush=True)
        await control({"calibrate": True})
    cfg = await control({"set": {"learn_costs": False}, "freeze_branches": True, "cost": True})
    config["server"] = {k: v for k, v in cfg.items() if k != "cost"}
    config["cost"] = cfg["cost"]
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"[config] {config['server']}", flush=True)

    with open(os.path.join(args.out, "clients.jsonl"), "a") as clients, open(os.path.join(args.out, "server.jsonl"), "a") as server:
        for trial in range(args.trials):
            for cell in (cells if trial % 2 == 0 else cells[::-1]):
                cfg = await reset(cell)
                print(f"[cell] trial={trial} planner={cell} {cfg}", flush=True)
                t0 = time.perf_counter()
                recs = await asyncio.gather(*(run_entry(e, t0, trial, cell, clients, args.verbose) for e in entries))
                stats = await control({"stats": True, "detail": True})
                line = {"trial": trial, "cell": cell, "wall_s": round(time.perf_counter() - t0, 3),
                        "client_failed": sum(r["rc"] != 0 for r in recs), **stats["stats"], "requests": stats["requests"]}
                server.write(json.dumps(line) + "\n"); server.flush()
                s = stats["stats"]
                print(f"[done] trial={trial} planner={cell} wall={line['wall_s']:.1f}s loads={s['loads']} load_s={s['load_s']} "
                      f"queue_wait_ms={s['queue_wait_ms']} sum_wait_s={s['sum_wait_s']} deviations={s['deviations']} "
                      f"fallbacks={s['fallbacks']} failed={line['client_failed']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
