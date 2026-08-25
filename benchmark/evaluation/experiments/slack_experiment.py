"""E1: how much idle time the evaluation workloads leave the engine.

Speculative prefill can only use time the engine is not spending on real
requests, so this experiment measures that time on the baseline server
(prefix caching only, no speculation) with a timestamped server log:

    single   every program runs alone, `--cases` consecutive dataset cases each
    blend    one trial (`--blend-trial`) of every blend given, same schedule the
             load experiment runs, per-instance cache isolation

For every workflow: span (first request in -> last decode out), the share of
the span the engine spends prefilling / decoding its requests, the share where
the workflow has no request pending (the client is running Python or a tool),
the window between registration and the first call, and every inter-call gap
with the next call's uncached prefill (so a gap can be judged long enough to
prefill the next call entirely).  For a blend: the share of the trace during
which the engine is empty, decoding only, or has a real prefill in flight.

Usage (from the repository root):
    python -m benchmark.evaluation.experiments.slack_experiment --cases 3 \
        --blend benchmark/evaluation/blends/poisson-0.5.json,benchmark/evaluation/blends/uniform-w3.json,benchmark/evaluation/blends/burst.json
"""

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict

from benchmark.evaluation.experiments.smoke_experiment import (
    REPO, PROGRAMS, TENANTS, CELLS, ANSI, SERVE, DECODE, stop_server,
)
from benchmark.evaluation.experiments.load_experiment import (
    REGISTERED, SERVE_ID, CLIENT_ENV, run_entry, wait_ready, mean, pct,
)

STAMP = re.compile(r"^(\d+\.\d+) (.*)$")
CALL = re.compile(r"call from (\S+)#(\d+) key=")
FINAL = re.compile(r"\[final\] (\S+?):(.+?):s(\d+)c(\d+) typed_ok=\S+ output=(.*)$")


# ------------------------------------------------------------------ running

def start_server(programs, extra, log_path, model, workers):
    """Like the load harness, but every server line is prefixed with time.time()."""
    cmd = [sys.executable, os.path.join(REPO, "start_server.py"), "--model", model, "--workers", str(workers)]
    for program in programs:
        cmd += ["--program", program]
    cmd += extra
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
    log = open(log_path, "w")

    def pump():
        for raw in proc.stdout:
            log.write(f"{time.time():.6f} {raw.decode(errors='replace')}")
            log.flush()
        log.close()

    threading.Thread(target=pump, daemon=True).start()
    return proc


def wipe_dbs():
    for db in glob.glob(os.path.join(REPO, ".jac", "data", "*.db")) + glob.glob(os.path.join(REPO, PROGRAMS, ".jac", "data", "*.db")):
        os.remove(db)


def run_single(args, out_root, results):
    out_dir = os.path.join(out_root, "single")
    os.makedirs(out_dir, exist_ok=True)
    wipe_dbs()
    programs = [f"{name}:{PROGRAMS}/{src}:{port}" for name, src, port, *_ in TENANTS]
    log_path = os.path.join(out_dir, "server.log")
    server = start_server(programs, CELLS["off"], log_path, args.model, args.workers)
    lock = threading.Lock()
    try:
        wait_ready(log_path, server, len(programs))
        t0 = time.time()
        order = 0
        for case in range(args.cases):
            for name, src, port, var, size, extra in TENANTS:
                entry = {"order": order, "tenant": name, "program_name": name, "case_index": case % size,
                         "jac_file": f"{PROGRAMS}/{src}", "env": {var: str(case % size), **extra},
                         "start_offset_s": None}
                run_entry(entry, args, out_dir, t0, results, lock, 0, "single")
                order += 1
    finally:
        stop_server(server)
    print(f"[single] {order} workflows {time.time() - t0:.0f}s", flush=True)


def run_blend(path, args, out_root, results):
    blend = json.load(open(path))
    tag = os.path.splitext(os.path.basename(path))[0]
    trial_spec = next(t for t in blend["trials"] if t["trial"] == args.blend_trial)
    entries = sorted(trial_spec["entries"], key=lambda e: e["order"])
    out_dir = os.path.join(out_root, f"blend-{tag}")
    os.makedirs(out_dir, exist_ok=True)
    wipe_dbs()
    log_path = os.path.join(out_dir, "server.log")
    extra = CELLS["off"] + (["--cache-isolation", "instance"] if args.isolation == "instance" else [])
    server = start_server(blend["server"]["programs"], extra, log_path, args.model, args.workers)
    lock = threading.Lock()
    try:
        wait_ready(log_path, server, len(blend["server"]["programs"]))
        t0 = time.time()

        def go(entry):
            time.sleep(max(0.0, t0 + entry["start_offset_s"] - time.time()))
            run_entry(entry, args, out_dir, t0, results, lock, args.blend_trial, f"blend-{tag}")
        threads = [threading.Thread(target=go, args=(e,)) for e in entries]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        stop_server(server)
    print(f"[blend-{tag}] {len(entries)} workflows {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------- aggregating

def parse_log(path):
    """-> registered {(prog, session): t}, calls {(prog, session): [call dicts in order]}"""
    registered, calls, pending = {}, defaultdict(list), {}
    for line in open(path, errors="replace"):
        m = STAMP.match(ANSI.sub("", line.rstrip("\n")))
        if not m:
            continue
        t, text = float(m.group(1)), m.group(2)
        if m2 := REGISTERED.search(text):
            registered[(m2.group(3), int(m2.group(4)))] = t
        elif m2 := CALL.search(text):
            pending[(m2.group(1), int(m2.group(2)))] = t
        elif m2 := SERVE.search(text):
            sid = SERVE_ID.match(m2.group(1))
            if sid is None:
                continue
            key = (sid.group(1), int(sid.group(3)))
            dur = float(m2.group(2)) / 1000
            calls[key].append({"site": sid.group(2).split("@")[0], "id": m2.group(1),
                               "arrival": pending.pop(key, t - dur), "start": t - dur, "first_token": t,
                               "ttft": dur, "cached": int(m2.group(3)), "prompt": int(m2.group(4)),
                               "end": None, "decode": 0.0, "tool": False})
        elif m2 := DECODE.search(text):
            rid = text.split("[decode] ", 1)[1].split()[0]
            sid = SERVE_ID.match(rid)
            if sid is None:
                continue
            for c in reversed(calls[(sid.group(1), int(sid.group(3)))]):
                if c["id"] == rid:
                    c["end"], c["decode"] = t, float(m2.group(1)) / 1000
                    break
        elif m2 := FINAL.search(text):
            key = (m2.group(1), int(m2.group(3)))
            for c in reversed(calls[key]):
                if c["end"] is not None:
                    c["tool"] = m2.group(5).startswith("'<tool_call>")
                    break
    return registered, calls


def workflow_stats(registered, key, seq):
    seq = [c for c in seq if c["end"] is not None]
    if not seq:
        return None
    span = seq[-1]["end"] - seq[0]["arrival"]
    pending = sum(c["end"] - c["arrival"] for c in seq)
    gaps = []
    for a, b in zip(seq, seq[1:]):
        gaps.append({"gap": b["arrival"] - a["end"], "after_tool": a["tool"],
                     "next_uncached": b["prompt"] - b["cached"], "next_site": b["site"]})
    return {"program": key[0], "session": key[1], "calls": len(seq), "span": span,
            "prefill": sum(c["ttft"] for c in seq), "decode": sum(c["decode"] for c in seq),
            "pending": pending, "idle": span - pending,
            "startup": seq[0]["arrival"] - registered[key] if key in registered else None,
            "gaps": gaps}


def engine_stats(calls):
    """Share of the trace (first arrival -> last end) with 0 requests, decode only, prefill in flight."""
    events = []
    for seq in calls.values():
        for c in seq:
            if c["end"] is None:
                continue
            events.append((c["arrival"], "p", 1)); events.append((c["first_token"], "p", -1))
            events.append((c["first_token"], "d", 1)); events.append((c["end"], "d", -1))
    if not events:
        return None
    events.sort()
    t0, t1 = events[0][0], max(e[0] for e in events)
    p = d = 0
    prev = t0
    empty = decode_only = prefill = 0.0
    conc = 0.0
    for t, kind, delta in events:
        dt = t - prev
        if p > 0:
            prefill += dt
        elif d > 0:
            decode_only += dt
        else:
            empty += dt
        conc += (p + d) * dt
        prev = t
        if kind == "p":
            p += delta
        else:
            d += delta
    span = t1 - t0
    return {"span": span, "empty": empty / span, "decode_only": decode_only / span,
            "prefill": prefill / span, "mean_inflight": conc / span}


def fit_slope(calls):
    """ms per uncached prompt token of the baseline prefill (OLS with intercept)."""
    xs, ys = [], []
    for seq in calls.values():
        for c in seq:
            if c["end"] is not None:
                xs.append(c["prompt"] - c["cached"]); ys.append(c["ttft"] * 1000)
    if len(xs) < 3:
        return float("nan"), float("nan")
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    return slope, my - slope * mx


def aggregate(root):
    out = ["# E1: idle time in the evaluation workloads", ""]
    report = {}
    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for run in runs:
        log = os.path.join(root, run, "server.log")
        if not os.path.exists(log):
            continue
        registered, calls = parse_log(log)
        flows = [w for w in (workflow_stats(registered, k, s) for k, s in calls.items()) if w]
        slope, icpt = fit_slope(calls)
        report[run] = {"workflows": flows, "engine": engine_stats(calls), "slope_ms_per_token": slope, "intercept_ms": icpt}
        out += [f"## {run}", "",
                f"baseline prefill cost: {slope:.3f} ms per uncached token + {icpt:.1f} ms ({sum(len(s) for s in calls.values())} calls)", ""]
        out += ["| program | workflows | calls | span s | prefill % | decode % | idle % | startup ms | gaps | gap p10 / p50 / p90 ms | after tool | gap ≥ next prefill | next uncached p50 |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        by_prog = defaultdict(list)
        for w in flows:
            by_prog[w["program"]].append(w)
        for prog in sorted(by_prog):
            ws = by_prog[prog]
            gaps = [g for w in ws for g in w["gaps"]]
            gl = [g["gap"] * 1000 for g in gaps]
            covered = [g["gap"] * 1000 >= icpt + slope * g["next_uncached"] for g in gaps]
            starts = [w["startup"] * 1000 for w in ws if w["startup"] is not None]
            out.append(f"| {prog} | {len(ws)} | {mean([w['calls'] for w in ws]):.1f} | {mean([w['span'] for w in ws]):.1f} "
                       f"| {100 * sum(w['prefill'] for w in ws) / sum(w['span'] for w in ws):.1f} "
                       f"| {100 * sum(w['decode'] for w in ws) / sum(w['span'] for w in ws):.1f} "
                       f"| {100 * sum(w['idle'] for w in ws) / sum(w['span'] for w in ws):.1f} "
                       f"| {mean(starts):.0f} | {len(gaps) / len(ws):.1f} "
                       f"| {pct(gl, 0.10):.0f} / {pct(gl, 0.50):.0f} / {pct(gl, 0.90):.0f} "
                       f"| {100 * mean([g['after_tool'] for g in gaps]) if gaps else float('nan'):.0f}% "
                       f"| {100 * mean(covered) if covered else float('nan'):.0f}% "
                       f"| {pct([g['next_uncached'] for g in gaps], 0.5):.0f} |")
        gaps = [g for w in flows for g in w["gaps"]]
        gl = [g["gap"] * 1000 for g in gaps]
        covered = [g["gap"] * 1000 >= icpt + slope * g["next_uncached"] for g in gaps]
        out.append(f"| all | {len(flows)} | {mean([w['calls'] for w in flows]):.1f} | {mean([w['span'] for w in flows]):.1f} "
                   f"| {100 * sum(w['prefill'] for w in flows) / sum(w['span'] for w in flows):.1f} "
                   f"| {100 * sum(w['decode'] for w in flows) / sum(w['span'] for w in flows):.1f} "
                   f"| {100 * sum(w['idle'] for w in flows) / sum(w['span'] for w in flows):.1f} "
                   f"| {mean([w['startup'] * 1000 for w in flows if w['startup'] is not None]):.0f} | {len(gaps) / len(flows):.1f} "
                   f"| {pct(gl, 0.10):.0f} / {pct(gl, 0.50):.0f} / {pct(gl, 0.90):.0f} "
                   f"| {100 * mean([g['after_tool'] for g in gaps]):.0f}% | {100 * mean(covered):.0f}% "
                   f"| {pct([g['next_uncached'] for g in gaps], 0.5):.0f} |")
        e = report[run]["engine"]
        out += ["", f"engine over the trace ({e['span']:.0f} s, mean requests in flight {e['mean_inflight']:.2f}): "
                    f"empty {100 * e['empty']:.1f}%, decode only {100 * e['decode_only']:.1f}%, real prefill in flight {100 * e['prefill']:.1f}%", ""]
        # gap by site: where the gaps are
        by_site = defaultdict(list)
        for g in gaps:
            by_site[g["next_site"]].append(g)
        out += ["| gap before | n | gap p50 ms | after tool | next uncached p50 | gap ≥ next prefill |", "|---|---|---|---|---|---|"]
        for site in sorted(by_site, key=lambda s: -len(by_site[s])):
            gs = by_site[site]
            out.append(f"| {site} | {len(gs)} | {pct([g['gap'] * 1000 for g in gs], 0.5):.0f} | {100 * mean([g['after_tool'] for g in gs]):.0f}% "
                       f"| {pct([g['next_uncached'] for g in gs], 0.5):.0f} "
                       f"| {100 * mean([g['gap'] * 1000 >= icpt + slope * g['next_uncached'] for g in gs]):.0f}% |")
        out.append("")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(out))
    with open(os.path.join(root, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cases", type=int, default=3, help="consecutive dataset cases per program in single mode")
    ap.add_argument("--blend", default="", help="comma-separated blend json paths (one trial each)")
    ap.add_argument("--blend-trial", type=int, default=0)
    ap.add_argument("--isolation", choices=["none", "instance"], default="instance")
    ap.add_argument("--tool-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "slack-7B"))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.aggregate_only:
        results = []
        if args.cases > 0:
            run_single(args, args.out, results)
        for path in [p for p in args.blend.split(",") if p]:
            run_blend(path, args, args.out, results)
        with open(os.path.join(args.out, "clients.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    aggregate(args.out)


if __name__ == "__main__":
    main()
