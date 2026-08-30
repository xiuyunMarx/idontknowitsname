#!/usr/bin/env python
"""Profile every (program, case) once against a running server and write profiles.json:
the callsite/model chain each case actually executes with per-step engine seconds, plus
the server's measured load times. gen_schedule.py's adversarial pattern simulates candidate
bursts on these chains to pick the ones where lookahead pays.

    python benchmark/workload/profile.py --out benchmark/workload/profiles.json [--programs hover.jac,...]

Runs sequentially (one program at a time, so per-step times are uncontended), calibrates
the server's cost table first unless --no-calibrate, and leaves learn_costs as it was."""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from send_requests import JAC_ENV, PROGRAMS, control  # noqa: E402

EXCLUDE = {"cascade.jac"}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark/workload/profiles.json"))
    ap.add_argument("--programs", default="", help="subset by file name (default: all but cascade)")
    ap.add_argument("--cases", type=int, default=0, help="at most this many cases per program (0 = all)")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep the cases already in --out, profile the rest")
    args = ap.parse_args()
    names = set(args.programs.split(",")) if args.programs else None
    paths = [p for p in PROGRAMS if os.path.basename(p) not in EXCLUDE and os.path.exists(os.path.join(ROOT, p))
             and (names is None or os.path.basename(p) in names)]

    if not args.no_calibrate:
        print("[calibrate] measuring load times", flush=True)
        await control({"calibrate": True})
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            profiles = json.load(f)
        cfg = await control({"cost": True})
        profiles["cost"] = cfg["cost"]
        seen = len((await control({"stats": True, "detail": True}))["requests"])
    else:
        cfg = await control({"reset": True, "cold": True, "cost": True})
        profiles = {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "server": {k: v for k, v in cfg.items() if k != "cost"}},
                    "cost": cfg["cost"], "cases": {}}
        seen = 0
    for path in paths:
        var, n = PROGRAMS[path]
        name = os.path.basename(path)
        for idx in range(n if not args.cases else min(n, args.cases)):
            if f"{name}:{idx}" in profiles["cases"] and profiles["cases"][f"{name}:{idx}"]["rc"] == 0:
                continue
            env = {**os.environ, **JAC_ENV, var: str(idx)}
            t = time.perf_counter()
            rc = subprocess.run(["jac", "run", path], env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
            wall = time.perf_counter() - t
            for _ in range(30):   # speculation may still be draining; per-request records land in `finally`
                out = await control({"stats": True, "detail": True})
                reqs = out["requests"][seen:]
                if not any(r["error"] is None and r["done_at"] == 0 for r in reqs):
                    break
                await asyncio.sleep(0.5)
            seen = len(out["requests"])
            chain = [{"callsite": r["callsite"], "kind": r["kind"], "model": r["model"],
                      "exec_s": round(r["done_at"] - r["dispatched_at"], 3),
                      "queue_s": round(r["dispatched_at"] - r["created_at"], 3),
                      "gap_s": round(r["created_at"] - reqs[i - 1]["done_at"], 3) if i else 0.0,
                      "output_tokens": r["output_tokens"]}
                     for i, r in enumerate(reqs)]
            profiles["cases"][f"{name}:{idx}"] = {"program": path, "env": var, "case_index": idx, "rc": rc,
                                                  "wall_s": round(wall, 2), "chain": chain}
            models = " ".join(c["model"].split("/")[-1] for c in chain)
            print(f"[profile] {name:<18} idx={idx:<2} rc={rc} wall={wall:5.1f}s steps={len(chain)} {models}", flush=True)
            with open(args.out, "w") as f:
                json.dump(profiles, f, indent=1)
    print(f"[profile] wrote {args.out}: {len(profiles['cases'])} cases", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
