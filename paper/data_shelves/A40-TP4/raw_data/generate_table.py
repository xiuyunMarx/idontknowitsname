"""Metric tables of the benchmark results.

  <arm>/per-agent/<agent>/<agent>.csv       the arm's sweep of that agent, one row per concurrency c
  mixed_workload_metrics/mix_L<n>.csv       every arm at lane count n, one row per arm
  mixed_workload_metrics/mix_L<n>_programs.csv   the same split by agent

    python3 benchmark/results/generate_table.py
"""
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "mixed_workload_metrics")
ARM_ORDER = ["vanilla", "adapter", "continuum", "cachescout", "kvflow", "kvonly", "relayout", "continuum_relayout",
             "cachescout_relayout", "kvflow_relayout", "ours_noprefetch", "ours"]


def median(xs):
    return statistics.median(xs) if xs else 0.0


def load(level_dir):
    """Measured sessions, and (start time, program, first_token event) of every call that got a first token."""
    with open(os.path.join(level_dir, "sessions.jsonl")) as f:
        sessions = [json.loads(l) for l in f if l.strip()]
    sessions = [s for s in sessions if s["phase"] == "measured"]
    program_of = {str(s["pid"]): s["program"] for s in sessions}
    events = defaultdict(dict)
    with open(os.path.join(level_dir, "requests.jsonl")) as f:
        for l in f:
            e = json.loads(l)
            events[e["request_id"]][e["event"]] = e
    calls = []
    for e in events.values():
        if "first_token" not in e:
            continue
        sid = e["request_admitted"]["session_id"]
        if sid in program_of:
            calls.append((e["request_start"]["time_ns"] / 1e9, program_of[sid], e["first_token"]))
    return sessions, calls


def full_load_window(sessions):
    """(t0, hi): measurement start and seconds until the first fixed lane finishes."""
    t0 = min(s["t0"] for s in sessions)
    fixed = [s for s in sessions if s["fixed"]]
    lanes = {s["lane"] for s in fixed}
    hi = min(max(s["t1"] for s in fixed if s["lane"] == lane) for lane in lanes)
    return t0, hi - t0


def row(arm, program, sessions, calls, t0, hi):
    """Metrics inside the full-load window: sessions finished and calls started within hi seconds of t0.
    Throughput is prompt ktokens/min (all prompt tokens) and recompute ktokens/min (the uncached part)."""
    minutes = hi / 60
    done = [s for s in sessions if s["t1"] - t0 <= hi and s["rc"] == 0]
    ft = [f for t, _, f in calls if t - t0 <= hi]
    prompt = sum(f["prompt_tokens"] for f in ft)
    recompute = sum(f["prompt_tokens"] - f["cached_tokens"] for f in ft)
    r = {"arm": arm, "program": program, "window_lo_s": 0, "window_hi_s": round(hi, 1), "window_min": round(minutes, 1),
         "calls": len(ft), "calls_per_min": round(len(ft) / minutes, 2),
         "sessions_done": len(done), "sessions_per_min": round(len(done) / minutes, 2),
         "jct_done_mean_s": round(statistics.mean(s["wall"] for s in done), 1) if done else "",
         "jct_done_p50_s": round(median([s["wall"] for s in done]), 1) if done else ""}
    if ft:
        r.update({"prompt_ktokens_per_min": round(prompt / 1000 / minutes, 1),
                  "recompute_ktokens_per_min": round(recompute / 1000 / minutes, 1),
                  "prompt_tokens_per_call": round(prompt / len(ft)),
                  "recompute_tokens_per_call": round(recompute / len(ft)),
                  "miss_pct": round(100 * recompute / prompt, 1),
                  "host_pct": round(100 * sum(f["cached_host"] for f in ft) / prompt, 1),
                  "device_pct": round(100 * sum(f["cached_device"] for f in ft) / prompt, 1),
                  "ttft_mean_s": round(statistics.mean(f["ttft_ms"] for f in ft) / 1000, 1),
                  "ttft_p50_s": round(median([f["ttft_ms"] for f in ft]) / 1000, 1)})
    return r


SWEEP_COLUMNS = ["c", "sessions", "failed", "accuracy", "sessions_per_min", "jct_mean_s", "jct_p50_s", "jct_p95_s",
                 "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "device_pct", "host_pct", "miss_pct",
                 "calls_per_min", "prompt_ktokens_per_min", "recompute_ktokens_per_min"]


def per_agent_table(agent_dir):
    rows = []
    for level_dir in glob.glob(os.path.join(agent_dir, "sweep_*_c*")):
        if not os.path.exists(os.path.join(level_dir, "summary.json")):
            continue
        with open(os.path.join(level_dir, "summary.json")) as f:
            s = json.load(f)
        (program, r), = s["programs"].items()
        rows.append(dict(r, c=s["lanes"][program], sessions_per_min=s["all"]["sessions_per_min_full_load"]))
    if rows:
        rows.sort(key=lambda r: r["c"])
        write(os.path.join(agent_dir, os.path.basename(agent_dir) + ".csv"), rows, SWEEP_COLUMNS)


def write(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def level_table(level):
    rows, program_rows = [], []
    for level_dir in glob.glob(os.path.join(ROOT, "*", "mixed", level)):
        if not os.path.exists(os.path.join(level_dir, "summary.json")):
            continue
        arm = level_dir.split(os.sep)[-3]
        sessions, calls = load(level_dir)
        t0, hi = full_load_window(sessions)
        rows.append(row(arm, "all", sessions, calls, t0, hi))
        for p in sorted({s["program"] for s in sessions}):
            program_rows.append(row(arm, p, [s for s in sessions if s["program"] == p],
                                    [c for c in calls if c[1] == p], t0, hi))
    if not rows:
        return
    order = lambda r: (ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99, r["program"])
    rows.sort(key=order)
    program_rows.sort(key=order)
    vanilla = next((r for r in rows if r["arm"] == "vanilla"), None)
    for r in rows:
        r["prompt_norm_vanilla"] = round(r["prompt_ktokens_per_min"] / vanilla["prompt_ktokens_per_min"], 3) if vanilla else ""
        r["calls_norm_vanilla"] = round(r["calls_per_min"] / vanilla["calls_per_min"], 3) if vanilla else ""
    write(os.path.join(OUT, level + ".csv"), rows)
    write(os.path.join(OUT, level + "_programs.csv"), program_rows)


if __name__ == "__main__":
    for agent_dir in sorted(glob.glob(os.path.join(ROOT, "*", "per-agent", "*"))):
        per_agent_table(agent_dir)
    os.makedirs(OUT, exist_ok=True)
    levels = {os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "*", "mixed", "mix_L*"))}
    for level in sorted(levels, key=lambda l: int(l[5:])):
        level_table(level)
