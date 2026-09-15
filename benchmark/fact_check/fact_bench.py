"""PBKV-style benchmark driver for benchmark/reproduce/fact_check.jac.

    python -m server.server --lru         --kv 24000 --host 4 > lru.log     # arm 1: LRU
    python -m benchmark.reproduce.fact_bench --sessions 12 --concurrency 4 --tag lru --server-log lru.log
    python -m server.server --kv 24000 --host 4 > ours.log    # arm 2: ours
    python -m benchmark.reproduce.fact_bench --sessions 12 --concurrency 4 --tag ours --server-log ours.log

`--host` equal to the device pool mirrors PBKV's HICACHE_RATIO=1: once both tiers
overflow, a miss is a full re-prefill of the session's accumulated evidence.

Every session is one `jac run` of fact_check.jac on its own HoVer claim (FC_CLAIM
from --claims: measured sessions take claims from the front of the file, warmup
sessions from the back, so no two sessions share private context). `--warmup`
sequential sessions let the server learn the graph, the value-flow rules and the
binding layout; they are excluded from the numbers.

Reported, PBKV Table 1 style: workflow latency (JCT), per-agent TTFT, and the
GPU (device) / host / miss split of prompt tokens — per callsite and per
invocation index within a session (the k-th call of a site extends the (k-1)-th
call's prompt, so the k>=2 rows are where workflow-aware eviction shows).
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROGRAM = os.path.join(HERE, "fact_check.jac")
CLAIMS = os.path.join(HERE, "hover_claims.tsv")

SERVE_RE = re.compile(r"\[serve\] (\d+)-(\d+)t(\d+)-[0-9a-f]+ (.+)")
CALL_RE = re.compile(r"\[call\] (\d+) #(\d+) (.+?)(?: predicted=(hit|miss))?$")
COUNTERS = ("[priority]", "[promote-rpc]", "[promote-deferred]", "[promote-dep]", "[layout]", "[timing]")


def pct(xs, q):
    return sorted(xs)[int(q * (len(xs) - 1))] if xs else float("nan")


def load_claims(path):
    """(label, claim) per non-comment line: label<TAB>hops<TAB>claim."""
    out = []
    with open(path) as f:
        for ln in f:
            if ln.strip() and not ln.startswith("#"):
                cols = ln.rstrip("\n").split("\t")
                out.append((cols[0], cols[-1]))
    return out


def run_session(args, idx, phase, logdir, label, claim):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAC_DATA_PATH=os.path.join(logdir, f"state-{phase}-{idx:03d}"))
    env[args.claim_env] = claim
    os.makedirs(env["JAC_DATA_PATH"], exist_ok=True)
    out_path = os.path.join(logdir, f"{phase}-{idx:03d}.log")
    t0 = time.monotonic()
    with open(out_path, "w") as out:
        proc = subprocess.Popen([args.jac, "run", args.program], env=env, cwd=REPO,
                                stdout=out, stderr=subprocess.STDOUT)
        rc = proc.wait(timeout=args.timeout)
    wall = time.monotonic() - t0
    verdict, rounds, findings = "", 0, 0
    with open(out_path) as f:
        for ln in f:
            if ln.startswith("Verdict: "):
                verdict = ln.split(": ", 1)[1].strip()
            m = re.match(r"Rounds: (\d+), (?:findings|notes|attempts): (\d+)", ln)
            if m:
                rounds, findings = int(m.group(1)), int(m.group(2))
    return {"pid": str(proc.pid), "phase": phase, "idx": idx, "rc": rc, "wall": round(wall, 3),
            "label": label, "verdict": verdict, "rounds": rounds, "findings": findings}


def parse_log(path, pids):
    """[serve] rows for the measured sessions, joined with their [call] label and
    numbered by invocation of the site within the session."""
    labels, serves, counters = {}, [], defaultdict(int)
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            m = CALL_RE.match(ln)
            if m:
                if m.group(1) in pids:
                    labels[(m.group(1), int(m.group(2)) + 1)] = m.group(3).split("(", 1)[0]
                continue
            m = SERVE_RE.match(ln)
            if m:
                if m.group(1) in pids:
                    kv = dict(p.split("=", 1) for p in m.group(4).split() if "=" in p)
                    serves.append({"pid": m.group(1), "call": int(m.group(2)), "turn": int(m.group(3)),
                                   **{k: float(v) for k, v in kv.items()}})
                continue
            for tag in COUNTERS:
                if ln.startswith(tag):
                    counters[tag] += 1
    # one row per call: its first turn (a typed retry re-sends the prompt plus feedback,
    # mostly cached; it would flatter the hit rate)
    first = {}
    for s in serves:
        k = (s["pid"], s["call"])
        if k not in first or s["turn"] < first[k]["turn"]:
            first[k] = s
    serves = list(first.values())
    seen = defaultdict(int)
    for s in sorted(serves, key=lambda s: (s["pid"], s["call"])):
        s["site"] = labels.get((s["pid"], s["call"]), "?")
        seen[(s["pid"], s["site"])] += 1
        s["inv"] = seen[(s["pid"], s["site"])]
    return serves, counters


def row(tag, name, ss):
    tt = [s["ttft_ms"] for s in ss]
    prompt = sum(s["prompt_tokens"] for s in ss)
    dev = sum(s["cached_device"] for s in ss)
    host = sum(s["cached_host"] for s in ss)
    print(f"[{tag}] {name:36s} {len(ss):4d} {pct(tt, .5):8.0f}ms {pct(tt, .95):8.0f}ms "
          f"{prompt / len(ss):7.0f} {100 * dev / prompt:7.1f}% {100 * host / prompt:6.1f}% "
          f"{100 * (prompt - dev - host) / prompt:6.1f}%")


def report(tag, records, serves, counters, concurrency=1):
    ok = [r for r in records if r["phase"] == "measured" and r["rc"] == 0]
    n_fail = sum(r["phase"] == "measured" and r["rc"] != 0 for r in records)
    print(f"\n[{tag}] ==== {len(ok)} measured sessions, failed={n_fail}")
    if ok:
        walls = [r["wall"] for r in ok]
        acc = sum(r["verdict"] == r["label"] for r in ok if r["verdict"] in ("SUPPORTED", "NOT_SUPPORTED"))
        print(f"[{tag}] JCT p50={pct(walls, .5):6.1f}s p95={pct(walls, .95):6.1f}s mean={statistics.mean(walls):6.1f}s "
              f"rounds mean={statistics.mean(r['rounds'] for r in ok):.2f} "
              f"findings mean={statistics.mean(r['findings'] for r in ok):.1f} "
              f"accuracy={acc}/{len(ok)}")
        c = concurrency
        inner = [r["wall"] for r in ok if c <= r["idx"] < len(ok) - c]   # drop the synchronized first wave and the draining last wave
        if len(inner) >= 4:
            print(f"[{tag}] JCT steady (sessions {c}..{len(ok) - c - 1}) p50={pct(inner, .5):6.1f}s p95={pct(inner, .95):6.1f}s "
                  f"mean={statistics.mean(inner):6.1f}s n={len(inner)}")
    if not serves:
        print(f"[{tag}] (no serve lines joined)")
        return
    print(f"[{tag}] {'site / invocation':36s} {'n':>4s} {'ttft p50':>9s} {'ttft p95':>9s} {'prompt':>7s} "
          f"{'device%':>8s} {'host%':>7s} {'miss%':>7s}")
    for site in sorted({s["site"] for s in serves}):
        ss = [s for s in serves if s["site"] == site]
        row(tag, site, ss)
        for inv in sorted({s["inv"] for s in ss}):
            row(tag, f"  #{inv}", [s for s in ss if s["inv"] == inv])
    row(tag, "all", serves)
    row(tag, "all, invocation >= 2", [s for s in serves if s["inv"] >= 2] or serves)
    print(f"[{tag}] server: " + " ".join(f"{k.strip('[]')}={v}" for k, v in sorted(counters.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=4,
                    help="measured sessions PER LANE; total = concurrency * sessions, so every lane stays busy for the whole run")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel session lanes")
    ap.add_argument("--warmup", type=int, default=2, help="sequential learning sessions, excluded")
    ap.add_argument("--claims", default=CLAIMS, help="one claim/topic per line, last TAB column")
    ap.add_argument("--program", default=PROGRAM, help="jac program to run per session")
    ap.add_argument("--claim-env", default="FC_CLAIM", help="env var that carries the claim/topic to the program")
    ap.add_argument("--jac", default=os.path.join(os.path.dirname(sys.executable), "jac"))
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--tag", default="fact")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--server", default="localhost:8964", metavar="HOST:PORT",
                    help="where to register the program's static analysis before the sessions")
    ap.add_argument("--no-register", action="store_true",
                    help="skip the registration (opaque baselines have no /v1/programs/register)")
    args = ap.parse_args()

    claims = load_claims(args.claims)
    total = args.concurrency * args.sessions
    assert len(claims) >= total + args.warmup, \
        f"{total} measured sessions need {total + args.warmup} distinct claims, have {len(claims)} (a repeated claim would share its prefix across sessions)"
    tag = f"{args.tag}/{os.path.splitext(os.path.basename(args.program))[0]}" + (f"x{args.concurrency}" if args.concurrency > 1 else "")
    logdir = args.logdir or tempfile.mkdtemp(prefix="fact-")
    print(f"[{tag}] {args.warmup} warmup + {total} sessions ({args.sessions} per lane) in {args.concurrency} lane(s); "
          f"logs in {logdir}", flush=True)
    if not args.no_register:
        # The sessions are plain `jac run`s: register the program's static analysis
        # once, up front (the launcher does the same per run). An opaque baseline
        # answers 404 here; that is its design, not a failure of the sweep.
        from static_analysis.agent_launcher import analyze_program, register
        payload = analyze_program(os.path.abspath(args.program))
        try:
            reply = register(args.server, payload)
            print(f"[{tag}] registered {len(payload['program']['sites'])} call sites with {args.server}: {reply}", flush=True)
        except RuntimeError as e:
            print(f"[{tag}] registration skipped: {e}", flush=True)
    records, lock = [], threading.Lock()
    for i in range(args.warmup):
        r = run_session(args, i, "warmup", logdir, *claims[-(i + 1)])
        records.append(r)
        print(f"[{tag}] warmup {i} rc={r['rc']} wall={r['wall']}s rounds={r['rounds']} "
              f"verdict={r['verdict']}/{r['label']}", flush=True)
    todo = list(range(total))

    def lane():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            r = run_session(args, i, "measured", logdir, *claims[i])
            with lock:
                records.append(r)
            print(f"[{tag}] session {i} rc={r['rc']} wall={r['wall']}s rounds={r['rounds']} "
                  f"findings={r['findings']} verdict={r['verdict']}/{r['label']}", flush=True)

    lanes = [threading.Thread(target=lane) for _ in range(args.concurrency)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    serves, counters = [], {}
    if args.server_log and os.path.exists(args.server_log):
        time.sleep(1.0)   # let the server flush the last session's lines
        serves, counters = parse_log(args.server_log, {r["pid"] for r in records if r["phase"] == "measured"})
    report(tag, records, serves, counters, args.concurrency)


if __name__ == "__main__":
    main()
