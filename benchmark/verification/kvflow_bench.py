"""KVFlow-shaped microbench: sequential sessions of one benchmark/kvflow/*.jac program.

Protocol (fresh server per arm; programs are learned online and in memory only):

    python benchmark/datasets/gen_kvflow.py                 # once
    python -m server.server --no-spec --kv 24000 > nospec.log      # arm 1
    python -m benchmark.kvflow_bench pipeline --tag nospec --server-log nospec.log
    python -m server.server --manage-only --kv 24000 > manage.log  # arm 2
    python -m benchmark.kvflow_bench pipeline --tag manage --server-log manage.log
    ... same for fanout (add --fanout 3 for the branching variant) and fanin
    python -m benchmark.kvflow_bench pipeline,fanout,fanin --concurrency 6 --sessions 12 ...
                                                             # high-concurrency: 6 lanes, mixed programs

8 agents x ~4k fixed tokens = ~32k > a 24k device pool, so every round evicts.
Under LRU the victim is the agent that runs next (KVFlow Fig. 1); with the
planner's priority map the victim is the transient tail and the farthest use.

One `jac run` per session (pid = server session id). `--warmup` sessions per
program let the server learn each cycle and are excluded. With `--concurrency C`
the measured sessions run in C parallel lanes, cycling through the given programs
(KVFlow 4.2: independent, non-sharing workflows competing for one pool). Server-side numbers come from the log:
[serve] ttft/cache split joined by pid + call index, [call] labels, planner counters.
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from benchmark.run import CALL_RE, REPO, SERVE_RE, _lines, _pct

PROGRAMS = os.path.join(REPO, "benchmark", "kvflow")


def run_session(args, idx, phase, logdir, program):
    env = dict(os.environ, PYTHONUNBUFFERED="1", KVF_ROUNDS=str(args.rounds),
               KVF_TASK=f"{args.task} (session {phase}-{idx})")
    if args.fanout:
        env["KVF_FANOUT"] = str(args.fanout)
    env["JAC_DATA_PATH"] = os.path.join(logdir, f"state-{phase}-{idx:03d}")
    os.makedirs(env["JAC_DATA_PATH"], exist_ok=True)
    t0 = time.monotonic()
    with open(os.path.join(logdir, f"{phase}-{idx:03d}-{program}.log"), "w") as out:
        proc = subprocess.Popen([args.jac, "run", os.path.join(PROGRAMS, program + ".jac")],
                                env=env, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
        rc = proc.wait(timeout=args.timeout)
    return {"pid": str(proc.pid), "phase": phase, "idx": idx, "program": program, "rc": rc,
            "wall": round(time.monotonic() - t0, 3)}


def parse_log(path, pids):
    """[serve] rows keyed (pid, call index) with the [call] label joined in."""
    labels, serves, counters = {}, [], {}
    for ln in _lines(path):
        m = CALL_RE.match(ln)
        if m and m.group(1) in pids:
            labels[(m.group(1), int(m.group(2)) + 1)] = m.group(3).split("(", 1)[0]
            continue
        m = SERVE_RE.match(ln)
        if m and m.group(1) in pids:
            kv = dict(p.split("=", 1) for p in m.group(4).split() if "=" in p)
            serves.append({"pid": m.group(1), "call": int(m.group(2)),
                           **{k: float(v) for k, v in kv.items()}})
            continue
        for tag in ("[promote-rpc] plan-", "[promote-deferred]", "[priority]",
                    "[promote-issued]", "[promote-flush]", "[prefill] plan-", "[ledger] drift"):
            if ln.startswith(tag):
                counters[tag] = counters.get(tag, 0) + 1
    for s in serves:
        s["site"] = labels.get((s["pid"], s["call"]), "?")
    return serves, counters


def report(tag, records, serves, counters):
    measured = [r for r in records if r["phase"] == "measured"]
    print(f"\n[{tag}] ==== {len(measured)} measured sessions, failed={sum(r['rc'] != 0 for r in measured)}")
    for prog in sorted({r["program"] for r in measured}):
        walls = [r["wall"] for r in measured if r["program"] == prog and r["rc"] == 0]
        if walls:
            print(f"[{tag}] {prog:10s} n={len(walls):3d} wall p50={_pct(walls, .5):6.1f}s "
                  f"p95={_pct(walls, .95):6.1f}s mean={statistics.mean(walls):6.1f}s")
    if not serves:
        print(f"[{tag}] (no serve lines joined)")
        return
    print(f"[{tag}] {'site':12s} {'n':>4s} {'ttft p50':>9s} {'ttft p95':>9s} {'prompt':>7s} "
          f"{'device%':>8s} {'host%':>6s}")
    for site in sorted({s["site"] for s in serves}):
        ss = [s for s in serves if s["site"] == site]
        tt = [s["ttft_ms"] for s in ss]
        prompt = sum(s["prompt_tokens"] for s in ss)
        dev = sum(s["cached_device"] for s in ss)
        host = sum(s["cached_host"] for s in ss)
        print(f"[{tag}] {site:12s} {len(ss):4d} {_pct(tt, .5):8.1f}ms {_pct(tt, .95):8.1f}ms "
              f"{prompt / len(ss):7.0f} {100 * dev / prompt:7.1f}% {100 * host / prompt:5.1f}%")
    later = [s["ttft_ms"] for s in serves if s["call"] >= 2]
    prompt = sum(s["prompt_tokens"] for s in serves)
    print(f"[{tag}] all call2+: ttft p50={_pct(later, .5):.1f}ms p95={_pct(later, .95):.1f}ms "
          f"device={100 * sum(s['cached_device'] for s in serves) / prompt:.1f}% "
          f"host={100 * sum(s['cached_host'] for s in serves) / prompt:.1f}%")
    print(f"[{tag}] planner: " + " ".join(f"{k.strip('[] ').rstrip('plan-').strip()}={v}"
                                        for k, v in sorted(counters.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("programs", help="comma list of pipeline,fanout,fanin; sessions cycle through it")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1, help="parallel session lanes")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--fanout", type=int, default=0, help="fanout.jac: executors per round (0 = all)")
    ap.add_argument("--task", default="summarize the quarterly plan in one line")
    ap.add_argument("--jac", default=os.path.join(os.path.dirname(sys.executable), "jac"))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tag", default="kvflow")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    programs = [p for p in args.programs.split(",") if p]
    assert all(p in ("pipeline", "fanout", "fanin") for p in programs), programs
    args.tag = f"{args.tag}/{'+'.join(programs)}" + (f"x{args.concurrency}" if args.concurrency > 1 else "")
    logdir = args.logdir or tempfile.mkdtemp(prefix="kvflow-")
    print(f"[{args.tag}] {args.warmup} warmup/program + {args.sessions} sessions in "
          f"{args.concurrency} lane(s), rounds={args.rounds} fanout={args.fanout or 'all'}; "
          f"logs in {logdir}", flush=True)
    records, lock = [], threading.Lock()
    for i in range(args.warmup):
        for prog in programs:            # sequential: the server learns each cycle cleanly
            r = run_session(args, i, "warmup", logdir, prog)
            records.append(r)
            print(f"[{args.tag}] warmup {prog} rc={r['rc']} wall={r['wall']}s", flush=True)
    todo = [(i, programs[i % len(programs)]) for i in range(args.sessions)]

    def lane():
        while True:
            with lock:
                if not todo:
                    return
                i, prog = todo.pop(0)
            r = run_session(args, i, "measured", logdir, prog)
            with lock:
                records.append(r)
            print(f"[{args.tag}] session {i} {prog} rc={r['rc']} wall={r['wall']}s", flush=True)

    lanes = [threading.Thread(target=lane) for _ in range(args.concurrency)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    serves, counters = [], {}
    if args.server_log and os.path.exists(args.server_log):
        pids = {r["pid"] for r in records if r["phase"] == "measured"}
        time.sleep(1.0)   # let the server flush the last session's lines
        serves, counters = parse_log(args.server_log, pids)
    report(args.tag, records, serves, counters)


if __name__ == "__main__":
    main()
