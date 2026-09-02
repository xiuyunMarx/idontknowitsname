"""Open-loop benchmark over the four real jac programs.

    python -m benchmark.run --rate 0.45 --duration 240 --seed 7 \
        --tag spec --server-log /path/to/server.log

Arrival model (CacheScout-style): session starts are a Poisson process at
`--rate` sessions/s for `--duration` seconds; each start picks an app by
`--mix` weights and a payload from that app's dataset pool — all drawn from
`--seed`, so a spec run and a --no-spec run on fresh servers replay the
identical trace. Before the measured window, `--warmup` sessions per app run
in four parallel per-app lanes (sequential within a lane) so the server can
learn each program online; warmup cycles the dataset labels (kb/web/calc,
shop/library/clinic) so every routed callsite is seen at least twice (the
const/copy rules and the prefix LCP all need two observations). Warmup
sessions are excluded from the report. `--docs` caps the report_gen pool so
documents are contended AND revisited across sessions.

Each session is one `jac run` subprocess: its pid is the server session id
(InterceptorLLM sends `user=<pid>` and closes the session at exit), which is
how the report joins client-side records with the server log — [serve] lines
carry `<pid>-<call>t<turn>-<hex>` request ids. Client side measures session
wall time and exit status; TTFT, cache split, predicted hit/miss, [probe]/
[prefill]/[steer] activity all come from `--server-log`.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "benchmark", "datasets")
PROGRAMS = os.path.join(REPO, "benchmark", "programs")


def _lines(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _tsv_rows(path):
    return [tuple(ln.split("\t", 1)) for ln in _lines(path)]


# app -> (payload env var, pool of (label, payload)); the label is the routed
# target where there is one, so warmup can cover every target evenly
APPS = {
    "group_chat": ("GC_QUERY", lambda: _tsv_rows(os.path.join(DATA, "chat", "queries.tsv"))),
    "text2sql": ("T2S_QUESTION", lambda: _tsv_rows(os.path.join(DATA, "sql", "questions.tsv"))),
    "report_gen": ("RG_DOC", lambda: [("doc", p) for p in sorted(
        os.path.join(DATA, "docs", f) for f in os.listdir(os.path.join(DATA, "docs"))
        if f.endswith(".txt"))]),
    "math_pipeline": ("MP_QUESTION", lambda: [
        ("math", q) for q in _lines(os.path.join(DATA, "math", "questions.txt"))]),
}


def make_trace(args, apps, pools):
    """Deterministic (t_start, app, payload) list: Poisson arrivals, weighted app mix."""
    rng = random.Random(args.seed)
    weights = [float(w) for w in args.mix.split(",")] if args.mix else [1.0] * len(apps)
    t, trace = 0.0, []
    while True:
        t += rng.expovariate(args.rate)
        if t >= args.duration:
            break
        app = rng.choices(apps, weights=weights[:len(apps)])[0]
        trace.append((round(t, 3), app, rng.choice(pools[app])[1]))
    return trace


class Session(threading.Thread):
    """One jac subprocess, watched to completion or timeout."""

    def __init__(self, args, idx, phase, app, payload, logdir):
        super().__init__(daemon=True)
        self.rec = {"idx": idx, "phase": phase, "app": app, "pid": None,
                    "wall": None, "rc": None, "timed_out": False}
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env[APPS[app][0]] = payload
        # private anchor store: concurrent runs of one program otherwise race on
        # the shared .jac root anchor and die in WriteConflict at exit
        env["JAC_DATA_PATH"] = os.path.join(logdir, f"state-{phase}-{idx:03d}")
        os.makedirs(env["JAC_DATA_PATH"], exist_ok=True)
        self._cmd = [args.jac, "run", os.path.join(PROGRAMS, f"{app}.jac")]
        self._env, self._timeout = env, args.timeout
        self._log = os.path.join(logdir, f"{phase}-{idx:03d}-{app}.log")

    def run(self):
        t0 = time.monotonic()
        with open(self._log, "w") as out:
            proc = subprocess.Popen(self._cmd, env=self._env, cwd=REPO,
                                    stdout=out, stderr=subprocess.STDOUT)
            self.rec["pid"] = proc.pid
            try:
                self.rec["rc"] = proc.wait(timeout=self._timeout)
            except subprocess.TimeoutExpired:
                self.rec["timed_out"] = True
                proc.kill()
                proc.wait()
        self.rec["wall"] = round(time.monotonic() - t0, 3)


def run_warmup(args, apps, pools, logdir):
    """`--warmup` sessions per app: parallel across apps, sequential within one."""
    rng = random.Random(args.seed + 1)
    plans = {}
    for app in apps:
        labels = sorted({lb for lb, _ in pools[app]})
        plans[app] = [rng.choice([p for lb, p in pools[app] if lb == labels[i % len(labels)]])
                      for i in range(args.warmup)]
    records, lock = [], threading.Lock()

    def lane(app):
        for i, payload in enumerate(plans[app]):
            s = Session(args, i, "warmup", app, payload, logdir)
            s.start()
            s.join()
            with lock:
                records.append(s.rec)

    lanes = [threading.Thread(target=lane, args=(app,)) for app in apps]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    return records


def run_measured(args, trace, logdir):
    """Open loop: spawn on schedule regardless of completions, then drain."""
    sessions, t0 = [], time.monotonic()
    for i, (t_arr, app, payload) in enumerate(trace):
        delay = t0 + t_arr - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        live = sum(1 for s in sessions if s.is_alive())
        if live >= args.max_live:
            print(f"[run] {live} sessions live at t={t_arr:.0f}s — box saturated, still spawning",
                  flush=True)
        s = Session(args, i, "measured", app, payload, logdir)
        s.rec["t_arrive"] = t_arr
        s.start()
        sessions.append(s)
    for s in sessions:
        s.join(timeout=args.timeout + 30)
    return [s.rec for s in sessions]


# ---- report --------------------------------------------------------------------

SERVE_RE = re.compile(r"\[serve\] (\d+)-(\d+)t(\d+)-[0-9a-f]+ (.+)")
CALL_RE = re.compile(r"\[call\] (\d+) #(\d+) (.+?)(?: predicted=(hit|miss))?$")


def parse_server_log(path, pid2app):
    """Pull per-request serve stats (joined to apps by pid), planner activity, and
    the [timing] prediction-validity feedback (deadline error + prefill progress
    at each real arrival)."""
    serves, calls, timings = [], [], []
    counters = {"probe": 0, "probe-miss": 0, "steer": 0, "prefill-plan": 0,
                "prefill-aborted": 0, "promote": 0, "drift-lines": 0, "unplanned": 0}
    for ln in _lines(path):
        if ln.startswith("[timing] "):
            parts = ln.split()
            if parts[1] in pid2app:
                if parts[2] == "unplanned":
                    counters["unplanned"] += 1
                else:
                    kv = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
                    done, total = kv.get("done", "0/0").split("/")
                    timings.append({"app": pid2app[parts[1]], "err": float(kv["err_s"]),
                                    "done": int(done), "total": int(total),
                                    "kind": kv.get("kind", "")})
            continue
        m = SERVE_RE.match(ln)
        if m:
            pid, epoch, turn = m.group(1), int(m.group(2)), int(m.group(3))
            if pid in pid2app:
                kv = dict(p.split("=", 1) for p in m.group(4).split() if "=" in p)
                serves.append({"app": pid2app[pid], "call": epoch, "turn": turn,
                               **{k: float(v) for k, v in kv.items()}})
            continue
        m = CALL_RE.match(ln)
        if m and m.group(1) in pid2app:
            calls.append({"app": pid2app[m.group(1)], "n": int(m.group(2)),
                          "site": m.group(3), "predicted": m.group(4)})
        elif ln.startswith("[probe] "):
            counters["probe"] += 1
        elif ln.startswith("[probe-miss]"):
            counters["probe-miss"] += 1
        elif ln.startswith("[steer]"):
            counters["steer"] += 1
        elif ln.startswith("[prefill] plan-"):
            counters["prefill-plan"] += 1
            counters["prefill-aborted"] += "aborted=1" in ln
        elif ln.startswith("[promote-rpc] plan-"):
            counters["promote"] += 1
        elif ln.startswith("[ledger] drift"):
            counters["drift-lines"] += 1
    return serves, calls, counters, timings


def _pct(xs, q):
    return sorted(xs)[int(q * (len(xs) - 1))] if xs else float("nan")


def report(tag, records, serves, calls, counters, timings=()):
    measured = [r for r in records if r["phase"] == "measured"]
    print(f"\n[{tag}] ==== client side ({len(measured)} measured sessions) ====")
    for app in sorted({r["app"] for r in measured}):
        rs = [r for r in measured if r["app"] == app]
        walls = [r["wall"] for r in rs if r["wall"] is not None and not r["timed_out"]
                 and r["rc"] == 0]
        bad = sum(1 for r in rs if r["timed_out"] or r["rc"] != 0)
        line = f"[{tag}] {app:14s} n={len(rs):3d} failed={bad:2d}"
        if walls:
            line += (f" wall p50={_pct(walls, .5):6.1f}s p95={_pct(walls, .95):6.1f}s"
                     f" mean={statistics.mean(walls):6.1f}s")
        print(line)
    if not serves:
        print(f"[{tag}] (no --server-log serve lines joined; server-side table skipped)")
        return
    print(f"[{tag}] ==== server side ({len(serves)} measured serves) ====")
    for app in sorted({s["app"] for s in serves}):
        ss = [s for s in serves if s["app"] == app]
        # call 1 is reachable only through the static prefix; calls >= 2 are the
        # ones structure prediction can pre-build — report them apart
        tt1 = [s["ttft_ms"] for s in ss if "ttft_ms" in s and s["call"] <= 1]
        ttn = [s["ttft_ms"] for s in ss if "ttft_ms" in s and s["call"] >= 2]
        cached = sum(s.get("cached_tokens", 0) for s in ss)
        prompt = sum(s.get("prompt_tokens", 0) for s in ss)
        host = sum(s.get("cached_host", 0) for s in ss)
        preds = [c["predicted"] for c in calls if c["app"] == app and c["predicted"]]
        hits = sum(1 for p in preds if p == "hit")
        print(f"[{tag}] {app:14s} serves={len(ss):4d}"
              f" ttft call1 p50={_pct(tt1, .5):7.1f}ms |"
              f" call2+ p50={_pct(ttn, .5):7.1f}ms p95={_pct(ttn, .95):7.1f}ms"
              f" cache={cached / prompt * 100 if prompt else 0:5.1f}%"
              f" (host {host / prompt * 100 if prompt else 0:4.1f}%)"
              f" predicted={hits}/{len(preds)}")
    print(f"[{tag}] planner: " + " ".join(f"{k}={v}" for k, v in counters.items()))
    if timings:
        errs = sorted(t["err"] for t in timings)
        prog = [(t["done"], t["total"]) for t in timings if t["total"]]
        full = sum(1 for d, n in prog if d >= n)
        part = sum(1 for d, n in prog if 0 < d < n)
        print(f"[{tag}] prediction validity: planned arrivals={len(timings)} "
              f"unplanned={counters.get('unplanned', 0)} "
              f"deadline_err p50={_pct(errs, .5):+.2f}s p90={_pct(errs, .9):+.2f}s "
              f"(pos = call after deadline) prefilled full={full} partial={part} "
              f"none={len(prog) - full - part}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=0.3, help="sessions per second")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--warmup", type=int, default=6,
                    help="learning sessions per app, cycling the routed targets")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mix", default=None, help="app weights, e.g. 1,1,1,1")
    ap.add_argument("--apps", default=",".join(APPS))
    ap.add_argument("--jac", default=shutil.which("jac")
                    or os.path.join(os.path.dirname(sys.executable), "jac"))
    ap.add_argument("--timeout", type=float, default=300.0, help="per-session kill timeout")
    ap.add_argument("--max-live", type=int, default=48)
    ap.add_argument("--docs", type=int, default=3,
                    help="cap the report_gen doc pool (0 = all): a small pool makes doc KV "
                         "contended AND reused, the regime where eviction policy matters")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--logdir", default=None, help="per-session stdout dir")
    ap.add_argument("--out", default=None, help="write raw records + serves as JSON")
    ap.add_argument("--report-from", default=None,
                    help="re-report a finished run: its --out JSON, joined with --server-log")
    args = ap.parse_args()

    if args.report_from:
        with open(args.report_from) as f:
            prior = json.load(f)
        pid2app = {str(r["pid"]): r["app"] for r in prior["records"]
                   if r["phase"] == "measured" and r["pid"]}
        serves, calls, counters, timings = parse_server_log(args.server_log, pid2app)
        report(args.tag, prior["records"], serves, calls, counters, timings)
        return

    apps = [a for a in args.apps.split(",") if a in APPS]
    pools = {a: APPS[a][1]() for a in apps}
    if args.docs and "report_gen" in pools:
        pools["report_gen"] = pools["report_gen"][:args.docs]
    logdir = args.logdir or tempfile.mkdtemp(prefix=f"jacbench-{args.tag}-")
    os.makedirs(logdir, exist_ok=True)
    trace = make_trace(args, apps, pools)
    print(f"[{args.tag}] trace: {len(trace)} sessions over {args.duration:.0f}s "
          f"(rate={args.rate}/s) + {args.warmup}/app warmup; logs in {logdir}", flush=True)

    records = run_warmup(args, apps, pools, logdir) if args.warmup else []
    print(f"[{args.tag}] warmup done ({len(records)} sessions)", flush=True)
    records += run_measured(args, trace, logdir)

    serves, calls, counters, timings = [], [], {}, []
    if args.server_log and os.path.exists(args.server_log):
        pid2app = {str(r["pid"]): r["app"] for r in records
                   if r["phase"] == "measured" and r["pid"]}
        serves, calls, counters, timings = parse_server_log(args.server_log, pid2app)
    report(args.tag, records, serves, calls, counters, timings)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "records": records, "serves": serves,
                       "calls": calls, "counters": counters, "timings": timings}, f, indent=1)
        print(f"[{args.tag}] wrote {args.out}")


if __name__ == "__main__":
    main()
