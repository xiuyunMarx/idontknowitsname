"""Steady-window table of a mix run, from the per-arm raw records (sessions.jsonl + requests.jsonl).

    python -m benchmark.mix_steady_csv --dir benchmark/mixed_results/mix4x4_h10                  # progress window
    python -m benchmark.mix_steady_csv --dir benchmark/mixed_results/mix4x4_h10 --lo 2100 --hi 3060

The default window is per arm and defined by progress, not by the clock: it opens when the fixed
program (the one with a session count, coding_agent in the 4x4 mix) starts its input --start-idx and
closes at the arm's own full-load end (the first fixed lane finishing). Inputs are taken in order and
prompts lengthen with the input index, so a fixed clock window covers different inputs in a fast and a
slow arm; the progress window covers about the same inputs in every arm. --lo/--hi give one fixed
clock window for all arms instead.

Throughput in the window is prompt ktokens/min: the full prompt length (cached + recomputed) of every
request whose prefill started inside (lo, hi] seconds after the arm's measurement start, per minute.
recompute ktokens/min is the uncached part of it, the prefill work the GPU actually did. calls/min and
sessions/min are reported but not the headline; session counts are quantized too coarsely there
(fact/coding sessions run 6-15 min). Cache columns are token-weighted over the same requests.
Writes <dir>/<name>_steady.csv (one row per arm) and <name>_steady_programs.csv.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

ARM_ORDER = ["vanilla", "continuum", "cachescout", "kvflow", "kvonly", "relayout", "continuum_relayout",
             "cachescout_relayout", "kvflow_relayout", "ours_noprefetch", "ours"]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def load(arm_dir):
    sess = [json.loads(l) for l in open(os.path.join(arm_dir, "sessions.jsonl")) if l.strip()]
    sess = [r for r in sess if r["phase"] == "measured"]
    prog = {str(r["pid"]): r["program"] for r in sess}
    ev = defaultdict(dict)
    for l in open(os.path.join(arm_dir, "requests.jsonl")):
        e = json.loads(l)
        ev[e["request_id"]][e["event"]] = e
    calls = []
    for e in ev.values():
        if "first_token" not in e:
            continue
        sid = e["request_admitted"]["session_id"]
        if sid in prog:
            calls.append((e["request_start"]["time_ns"] / 1e9, prog[sid], e["first_token"]))
    return sess, calls


def progress_window(sess, start_idx, t0):
    """(start of the fixed program's input start_idx, first fixed lane's last finish), in s after t0."""
    fixed = [r for r in sess if r["fixed"]]
    first = [r["t0"] for r in fixed if r["idx"] == start_idx]
    if not first:
        raise SystemExit(f"no fixed-program session with idx {start_idx}")
    full_load = min(max(r["t1"] for r in fixed if r["lane"] == lane) for lane in {r["lane"] for r in fixed})
    return first[0] - t0, full_load - t0


def row(arm, program, sess, calls, lo, hi, t0):
    mins = (hi - lo) / 60
    done = [r for r in sess if lo < r["t1"] - t0 <= hi and r["rc"] == 0]
    c = [ft for t, _, ft in calls if lo < t - t0 <= hi]
    P = sum(f["prompt_tokens"] for f in c)
    M = sum(f["prompt_tokens"] - f["cached_tokens"] for f in c)
    out = {"arm": arm, "program": program, "window_lo_s": round(lo, 1), "window_hi_s": round(hi, 1),
           "window_min": round(mins, 1), "calls": len(c), "calls_per_min": round(len(c) / mins, 2),
           "sessions_done": len(done), "sessions_per_min": round(len(done) / mins, 2),
           "jct_done_mean_s": round(statistics.mean(r["wall"] for r in done), 1) if done else "",
           "jct_done_p50_s": round(q([r["wall"] for r in done], 0.5), 1) if done else ""}
    if c:
        out.update({"prompt_ktokens_per_min": round(P / 1000 / mins, 1),
                    "recompute_ktokens_per_min": round(M / 1000 / mins, 1),
                    "prompt_tokens_per_call": round(P / len(c)),
                    "recompute_tokens_per_call": round(M / len(c)),
                    "miss_pct": round(100 * M / P, 1),
                    "host_pct": round(100 * sum(f["cached_host"] for f in c) / P, 1),
                    "device_pct": round(100 * sum(f["cached_device"] for f in c) / P, 1),
                    "ttft_mean_s": round(statistics.mean(f["ttft_ms"] for f in c) / 1000, 1),
                    "ttft_p50_s": round(q([f["ttft_ms"] for f in c], 0.5) / 1000, 1)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="mix run dir holding <arm>/sessions.jsonl and requests.jsonl")
    ap.add_argument("--start-idx", type=int, default=12,
                    help="progress window: open when the fixed program starts this input index (default 12)")
    ap.add_argument("--lo", type=float, help="fixed clock window start (s after measurement start); needs --hi")
    ap.add_argument("--hi", type=float, help="fixed clock window end (s)")
    a = ap.parse_args()
    if (a.lo is None) != (a.hi is None):
        ap.error("--lo and --hi go together")
    arms = [d for d in os.listdir(a.dir) if os.path.exists(os.path.join(a.dir, d, "requests.jsonl"))]
    arms.sort(key=lambda x: ARM_ORDER.index(x) if x in ARM_ORDER else 99)
    rows, prows = [], []
    for arm in arms:
        sess, calls = load(os.path.join(a.dir, arm))
        if not sess:
            continue
        t0 = min(r["t0"] for r in sess)
        if a.lo is not None:
            lo, hi = a.lo, a.hi
        else:
            lo, hi = progress_window(sess, a.start_idx, t0)
        rows.append(row(arm, "all", sess, calls, lo, hi, t0))
        for p in sorted({r["program"] for r in sess}):
            prows.append(row(arm, p, [r for r in sess if r["program"] == p],
                             [x for x in calls if x[1] == p], lo, hi, t0))
    van = next((r for r in rows if r["arm"] == "vanilla"), None)
    for r in rows:
        r["prompt_norm_vanilla"] = round(r["prompt_ktokens_per_min"] / van["prompt_ktokens_per_min"], 3) if van else ""
        r["calls_norm_vanilla"] = round(r["calls_per_min"] / van["calls_per_min"], 3) if van else ""
    name = os.path.basename(os.path.normpath(a.dir))
    for suffix, data in (("_steady.csv", rows), ("_steady_programs.csv", prows)):
        path = os.path.join(a.dir, name + suffix)
        fields = list(dict.fromkeys(k for r in data for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print("wrote", path)
    for r in rows:
        print(f"{r['arm']:20s} ({r['window_lo_s']:.0f},{r['window_hi_s']:.0f}] "
              f"prompt ktok/min={r['prompt_ktokens_per_min']:6.1f} ({r['prompt_norm_vanilla']}) "
              f"recompute ktok/min={r['recompute_ktokens_per_min']:5.1f} "
              f"calls/min={r['calls_per_min']:5.1f} ({r['calls_norm_vanilla']}) recompute/call={r.get('recompute_tokens_per_call', '')} miss={r.get('miss_pct', '')} "
              f"host={r.get('host_pct', '')} ttft={r.get('ttft_mean_s', '')} sess/min={r['sessions_per_min']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
