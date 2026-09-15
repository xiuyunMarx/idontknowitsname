"""Mixed-workload driver: fact_check and coding_agent sessions interleaved on one server.

    python -m server.server Qwen/Qwen3-8B --host 8 > ours.log
    python -m benchmark.mixed_workload.run --fact-lanes 8 --code-lanes 8 \
        --fact-sessions 3 --code-sessions 7 --tag ours --server-log ours.log

Protocol (closed loop, like fact_bench / PBKV's "number of concurrent workflows"):
  * warmup: --warmup sequential sessions of fact_check, then of coding_agent, so the server
    learns both programs before anything is measured (excluded from the numbers);
  * measured: --fact-lanes lanes each running --fact-sessions fact_check sessions and
    --code-lanes lanes each running --code-sessions coding_agent sessions, all lanes started
    at once. A per-lane count of 0 means "follow": those lanes keep taking fresh inputs until
    the other program's lanes have all finished (sessions in flight complete and count), so
    both programs share the pool for the whole run without guessing a JCT ratio. Default:
    coding fixed at 5 per lane, fact_check follows.
  * registration: both programs' static analyses are registered with the server before
    the warmup (static_analysis.agent_launcher), fact_check with no_header_last unless
    --fact-header-last, so its headers stay in front while coding_agent's sites may go
    header-last. An opaque baseline answers 404; that is its design.
  * every session is its own `jac run` process; the server log is joined by pid, so each
    program's [serve] rows are attributed exactly.

Reported per program (never merged: the two JCT scales differ by >2x): JCT full-run and
in the overlap window (sessions whose whole lifetime lies inside the interval where both
programs had measured sessions running, first synchronized wave dropped), per-site /
per-invocation TTFT and device/host/miss split, accuracy (fact verdict, coding all-pass).
Plus one server-wide cache row over both programs and the [program] sanity counters
(expect exactly two "new" programs and zero "joined mid-program" sessions).
"""
import argparse
import os
import statistics
import sys
import threading
import time
from types import SimpleNamespace

from benchmark.fact_check import fact_bench as fb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FACT_PROGRAM = os.path.join(REPO, "benchmark", "fact_check", "fact_check.jac")   # the canonical programs, not copies
CODE_PROGRAM = os.path.join(REPO, "benchmark", "coding", "coding_agent.jac")
FACT_CLAIMS = os.path.join(REPO, "benchmark", "fact_check", "hover_claims_120.tsv")
CODE_TASKS = os.path.join(REPO, "benchmark", "coding", "tasks.txt")

PROGRAMS = {   # name -> (program path arg, claim env var, claims file arg, first synchronized wave key)
    "fact": ("fact_program", "FC_CLAIM", "claims"),
    "code": ("code_program", "CA_TASKS", "tasks"),
}


def run_one(pargs, name, idx, phase, logdir, label, claim, records, lock, tag):
    t0 = time.monotonic()
    r = fb.run_session(pargs, idx, phase, os.path.join(logdir, name), label, claim)
    r.update(program=name, t0=t0, t1=time.monotonic())
    with lock:
        records.append(r)
    print(f"[{tag}/{name}] {phase} {idx} rc={r['rc']} wall={r['wall']}s rounds={r['rounds']} "
          f"findings={r['findings']} verdict={r['verdict']}/{r['label']}", flush=True)
    return r


def overlap_window(records, lanes):
    """[start, end] where both programs had measured sessions running, minus the first
    synchronized wave of each program (its idx < lanes)."""
    starts, ends = [], []
    for name in PROGRAMS:
        rs = [r for r in records if r["program"] == name and r["phase"] == "measured"]
        if not rs:
            return None
        later = [r for r in rs if r["idx"] >= lanes[name]] or rs
        starts.append(min(r["t0"] for r in later))
        ends.append(max(r["t1"] for r in rs))
    return max(starts), min(ends)


def report_program(tag, name, records, serves, window):
    ok = [r for r in records if r["program"] == name and r["phase"] == "measured" and r["rc"] == 0]
    n_fail = sum(r["program"] == name and r["phase"] == "measured" and r["rc"] != 0 for r in records)
    t = f"{tag}/{name}"
    print(f"\n[{t}] ==== {len(ok)} measured sessions, failed={n_fail}")
    if ok:
        walls = [r["wall"] for r in ok]
        acc = sum(r["verdict"] == r["label"] for r in ok)
        print(f"[{t}] JCT p50={fb.pct(walls, .5):6.1f}s p95={fb.pct(walls, .95):6.1f}s mean={statistics.mean(walls):6.1f}s "
              f"rounds mean={statistics.mean(r['rounds'] for r in ok):.2f} "
              f"findings mean={statistics.mean(r['findings'] for r in ok):.1f} accuracy={acc}/{len(ok)}")
        if window:
            a, b = window
            inner = [r["wall"] for r in ok if r["t0"] >= a and r["t1"] <= b]
            if len(inner) >= 4:
                print(f"[{t}] JCT overlap ({b - a:.0f}s window) p50={fb.pct(inner, .5):6.1f}s "
                      f"p95={fb.pct(inner, .95):6.1f}s mean={statistics.mean(inner):6.1f}s n={len(inner)}")
            else:
                print(f"[{t}] JCT overlap: only {len(inner)} sessions inside the window, not reported")
    if not serves:
        print(f"[{t}] (no serve lines joined)")
        return
    print(f"[{t}] {'site / invocation':36s} {'n':>4s} {'ttft p50':>9s} {'ttft p95':>9s} {'prompt':>7s} "
          f"{'device%':>8s} {'host%':>7s} {'miss%':>7s}")
    for site in sorted({s["site"] for s in serves}):
        ss = [s for s in serves if s["site"] == site]
        fb.row(t, site, ss)
        for inv in sorted({s["inv"] for s in ss}):
            fb.row(t, f"  #{inv}", [s for s in ss if s["inv"] == inv])
    fb.row(t, "all", serves)
    fb.row(t, "all, invocation >= 2", [s for s in serves if s["inv"] >= 2] or serves)


def program_counters(path):
    """(registered program files, opaque requests): the server registers each
    program once at startup of the run and serves any request without a matching
    call site as opaque text."""
    registered, opaque = [], 0
    with open(path) as f:
        for ln in f:
            if ln.startswith("[program] ") and " sites, " in ln:
                registered.append(os.path.basename(ln[len("[program] "):].split(":", 1)[0]))
            elif ln.startswith("[call] ") and "opaque request" in ln:
                opaque += 1
    return registered, opaque


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fact-lanes", type=int, default=8, help="parallel fact_check lanes")
    ap.add_argument("--code-lanes", type=int, default=8, help="parallel coding_agent lanes")
    ap.add_argument("--fact-sessions", type=int, default=0, help="fact_check sessions per lane; 0 = follow the coding lanes")
    ap.add_argument("--code-sessions", type=int, default=5, help="coding_agent sessions per lane; 0 = follow the fact lanes")
    ap.add_argument("--warmup", type=int, default=2, help="sequential learning sessions per program, excluded")
    ap.add_argument("--fact-program", default=FACT_PROGRAM)
    ap.add_argument("--code-program", default=CODE_PROGRAM)
    ap.add_argument("--claims", default=FACT_CLAIMS, help="HoVer claims, last TAB column")
    ap.add_argument("--tasks", default=CODE_TASKS, help="HumanEval bundles, last TAB column")
    ap.add_argument("--jac", default=os.path.join(os.path.dirname(sys.executable), "jac"))
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--tag", default="mixed")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--server", default="localhost:8964", metavar="HOST:PORT", help="where to register the programs")
    ap.add_argument("--no-register", action="store_true", help="skip the registration (opaque baselines)")
    ap.add_argument("--fact-header-last", action="store_true", help="let fact_check's sites go header-last too")
    args = ap.parse_args()

    os.environ.setdefault("FC_CACHE_DIR", os.path.join(REPO, "benchmark", "fact_check", "wiki_cache"))
    if not args.no_register:
        from static_analysis.agent_launcher import analyze_program, register
        for prog, nhl in ((args.fact_program, not args.fact_header_last), (args.code_program, False)):
            payload = analyze_program(os.path.abspath(prog))
            try:
                reply = register(args.server, payload, no_header_last=nhl)
                print(f"[{args.tag}] registered {os.path.basename(prog)}: {reply} no_header_last={nhl}", flush=True)
            except RuntimeError as e:
                print(f"[{args.tag}] registration skipped for {os.path.basename(prog)}: {e}", flush=True)
    lanes = {"fact": args.fact_lanes, "code": args.code_lanes}
    per_lane = {"fact": args.fact_sessions, "code": args.code_sessions}
    follow = {name for name in PROGRAMS if per_lane[name] == 0}
    assert len(follow) < 2, "at most one program can follow the other"
    inputs, pargs, todo = {}, {}, {}
    for name, (prog_attr, env, file_attr) in PROGRAMS.items():
        inputs[name] = fb.load_claims(getattr(args, file_attr))
        pool = len(inputs[name]) - args.warmup                 # warmup takes its inputs from the back
        total = pool if name in follow else lanes[name] * per_lane[name]
        assert pool >= total and pool >= lanes[name], \
            f"{name}: {total} measured sessions need {total + args.warmup} distinct inputs, have {len(inputs[name])}"
        todo[name] = list(range(total))
        pargs[name] = SimpleNamespace(program=getattr(args, prog_attr), claim_env=env, jac=args.jac, timeout=args.timeout)
    tag = f"{args.tag}/mixf{args.fact_lanes}c{args.code_lanes}"
    logdir = args.logdir or os.path.join(REPO, "results", "mixed_tmp", tag.replace("/", "_"))
    for name in PROGRAMS:
        os.makedirs(os.path.join(logdir, name), exist_ok=True)
    def plan(name):
        return f"{lanes[name]} lanes x " + ("follow" if name in follow else str(per_lane[name]))
    print(f"[{tag}] warmup {args.warmup}x2, then fact {plan('fact')} + code {plan('code')}; logs in {logdir}", flush=True)

    records, lock = [], threading.Lock()
    for name in PROGRAMS:   # warmup from the back of each input file, sequential, one program after the other
        for i in range(args.warmup):
            run_one(pargs[name], name, i, "warmup", logdir, *inputs[name][-(i + 1)], records, lock, tag) #type: ignore

    stop = threading.Event()                                  # set when the last fixed-count lane exits
    fixed_lanes = [sum(lanes[n] for n in PROGRAMS if n not in follow)]

    def lane(name):
        while True:
            with lock:
                if not todo[name] or (name in follow and stop.is_set()):
                    if name not in follow:
                        fixed_lanes[0] -= 1
                        if fixed_lanes[0] == 0:
                            stop.set()
                    return
                i = todo[name].pop(0)
            run_one(pargs[name], name, i, "measured", logdir, *inputs[name][i], records, lock, tag) #type: ignore

    threads = [threading.Thread(target=lane, args=(name,)) for name in PROGRAMS for _ in range(lanes[name])]
    t_start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[{tag}] measured phase wall={time.monotonic() - t_start:.0f}s", flush=True)

    window = overlap_window(records, lanes)
    serves, counters = {name: [] for name in PROGRAMS}, {}
    if args.server_log and os.path.exists(args.server_log):
        time.sleep(1.0)
        for name in PROGRAMS:
            pids = {r["pid"] for r in records if r["program"] == name and r["phase"] == "measured"}
            serves[name], counters = fb.parse_log(args.server_log, pids)
        new, opaque = program_counters(args.server_log)
        print(f"[{tag}] programs registered={len(new)} ({', '.join(new)}) opaque_requests={opaque}"
              + ("" if len(new) == 2 and opaque == 0 else "   <-- UNEXPECTED: check the registration and the callsite field"), flush=True)
    for name in PROGRAMS:
        report_program(tag, name, records, serves[name], window)
    both = serves["fact"] + serves["code"]
    if both:
        print(f"\n[{tag}] server-wide cache over both programs")
        fb.row(tag, "all", both)
        fb.row(tag, "all, invocation >= 2", [s for s in both if s["inv"] >= 2] or both)
        print(f"[{tag}] server: " + " ".join(f"{k.strip('[]')}={v}" for k, v in sorted(counters.items())))


if __name__ == "__main__":
    main()
