"""Poisson request generator against start_server.py.

Each request is one `jac run` of a random application (its input picked by the
per-program index variable), started at exponential inter-arrival gaps. Reports
per-program latency and overall throughput plus the server's load/cache counters,
so server configurations can be compared without restarting it:

    python send_requests.py --rate 0.5 --count 30 --seed 1
    python send_requests.py --set speculate=false --rate 0.5 --count 30 --seed 1

--set talks to the control port (start_server.CONTROL_PORT); every run starts from
a reset server (all engines parked on host, caches cleared) unless --no-reset.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

JAC_ENV = {"LD_LIBRARY_PATH": "/home/xiaoyu/miniconda3/envs/jaseci/lib", "JAC_ROUTE_CACHE_LAYOUT": "1"}
CONTROL_PORT = 8965

# program -> (env var selecting its input, number of inputs)
PROGRAMS = {
    "benchmark/applications/cascade.jac": ("CASCADE_INDEX", 16),
    "benchmark/applications/deep_research.jac": ("DEEP_TASK_INDEX", 4),
    "benchmark/applications/hover.jac": ("HOVER_CLAIM_INDEX", 5),
    "benchmark/applications/rag_qa.jac": ("RAG_INDEX", 12),
    "benchmark/applications/text2sql.jac": ("SQL_TASK_INDEX", 11),
    "benchmark/applications/triage.jac": ("TRIAGE_INDEX", 13),
}


async def run_one(path: str, index: int, log: list, verbose: bool) -> None:
    var, _ = PROGRAMS[path]
    env = {**os.environ, **JAC_ENV, var: str(index)}
    t = time.perf_counter()
    proc = await asyncio.create_subprocess_exec("jac", "run", path, env=env,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    lat = time.perf_counter() - t
    log.append((path, index, lat, proc.returncode))
    name = os.path.basename(path)
    print(f"[req] {name:<18} idx={index} {lat:6.1f}s rc={proc.returncode}", flush=True)
    if proc.returncode != 0 or verbose:
        sys.stderr.write(out.decode(errors="replace")[-1500:] + "\n")


async def control(req: dict) -> dict:
    reader, writer = await asyncio.open_connection("localhost", CONTROL_PORT)
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()
    resp = json.loads(await reader.readline())
    writer.close()
    if "error" in resp:
        raise RuntimeError(f"control: {resp['error']}")
    return resp


def _setting(kv: str):
    k, v = kv.split("=", 1)
    return k, {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.5, help="mean arrivals per second")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--programs", default=",".join(os.path.basename(p) for p in PROGRAMS), help="subset, by file name")
    ap.add_argument("--no-warmup", action="store_true", help="skip the sequential run that warms jac's compile cache")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="server setting: speculate=true|false gpu_slots=N host_slots=N")
    ap.add_argument("--no-reset", action="store_true", help="keep the server's engine placement and counters")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = [p for p in PROGRAMS if os.path.basename(p) in args.programs.split(",")]
    log: list = []
    if not args.no_warmup:  # concurrent cold compiles lock jac's sqlite cache
        for p in paths:
            await run_one(p, 0, [], args.verbose)

    cfg = await control({"set": dict(_setting(kv) for kv in args.set), "reset": not args.no_reset})
    print(f"[config] {cfg}", flush=True)
    started = time.perf_counter()
    tasks = []
    for _ in range(args.count):
        p = rng.choice(paths)
        tasks.append(asyncio.create_task(run_one(p, rng.randrange(PROGRAMS[p][1]), log, args.verbose)))
        await asyncio.sleep(rng.expovariate(args.rate))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - started

    print(f"\n[summary] rate={args.rate}/s count={args.count} wall={wall:.1f}s throughput={len(log) / wall:.3f}/s "
          f"failed={sum(rc != 0 for *_, rc in log)}")
    for p in paths:
        lats = sorted(l for path, _, l, rc in log if path == p and rc == 0)
        if lats:
            print(f"  {os.path.basename(p):<18} n={len(lats):<3} mean={statistics.mean(lats):6.1f}s "
                  f"p50={lats[len(lats) // 2]:6.1f}s p95={lats[min(len(lats) - 1, int(0.95 * len(lats)))]:6.1f}s")
    print(f"  server: {json.dumps((await control({'stats': True}))['stats'])}")
    lats = sorted(l for *_, l, rc in log if rc == 0)
    if lats:
        print(f"  {'all':<18} n={len(lats):<3} mean={statistics.mean(lats):6.1f}s p50={lats[len(lats) // 2]:6.1f}s "
              f"p95={lats[min(len(lats) - 1, int(0.95 * len(lats)))]:6.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
