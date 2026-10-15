"""Tables from the per-level summaries and raw records under benchmark/results/<arm>/.

    python -m benchmark.summarize [--results benchmark/results] [--start-frac 0.5]

Writes into <results>/:
  sweep_<program>.csv       one row per (arm, c): JCT, TTFT, cache split, throughput per level
  mixed.csv                 one row per (arm, lanes): whole-run totals from summary.json
  mixed_programs.csv        the same per program
  mixed_steady.csv          one row per (arm, lanes): throughput inside the progress window
  mixed_steady_programs.csv the same per program

Throughput columns: prompt_ktokens_per_min is the full prompt length (cached + recomputed) of every
request per minute; recompute_ktokens_per_min is the uncached part, the prefill work the GPU did;
output_ktokens_per_min the generated tokens. The steady window is defined by progress, not the
clock: it opens when the fixed program (coding_agent) starts its input --start-frac of the way
through its total sessions and closes at the first fixed lane's end, so it covers about the same
inputs in a fast and a slow arm. The window's requests come from the level's requests.jsonl slice
(request_start inside the window, first_token for the cache split), attributed to programs by
session pid from sessions.jsonl.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ORDER = ["vanilla", "adapter", "continuum", "cachescout", "kvflow", "kvonly", "relayout", "continuum_relayout",
             "cachescout_relayout", "kvflow_relayout", "ours_noprefetch", "ours"]
SWEEP_COLUMNS = ["arm", "c", "k", "sessions", "failed", "timed_out", "jct_p50_s", "jct_p95_s", "jct_mean_s",
                 "ttft_p50_ms", "ttft_p95_ms", "ttft_mean_ms", "device_pct", "host_pct", "miss_pct", "predicted_hit",
                 "accuracy", "calls", "prompt_tokens_per_call", "elapsed_s", "sessions_per_min", "calls_per_min",
                 "prompt_ktokens_per_min", "recompute_ktokens_per_min", "output_ktokens_per_min", "decode_ms_mean"]


def arm_key(arm: str) -> int:
    return ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)


def write_csv(path: str, rows: List[dict], columns: List[str] = None) -> None:
    if not rows:
        return
    columns = columns or ["arm"] + sorted({k for r in rows for k in r} - {"arm"})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def sweeps(results: str) -> None:
    per_program: Dict[str, List[dict]] = defaultdict(list)
    for path in glob.glob(os.path.join(results, "*", "sweep_*_c*", "summary.json")):
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            s = json.load(f)
        (program, row), = s["programs"].items()
        per_program[program].append(dict(row, arm=arm, c=s["lanes"][program], k=s["sessions"][program]))
    for program, rows in per_program.items():
        rows.sort(key=lambda r: (arm_key(r["arm"]), r["c"]))
        write_csv(os.path.join(results, f"sweep_{program}.csv"), rows, SWEEP_COLUMNS)


def mixed(results: str, start_frac: float) -> None:
    whole, per_prog, steady, steady_prog = [], [], [], []
    for path in sorted(glob.glob(os.path.join(results, "*", "mix_L*", "summary.json"))):
        level_dir = os.path.dirname(path)
        arm = os.path.basename(os.path.dirname(level_dir))
        with open(path) as f:
            s = json.load(f)
        lanes = s["lanes"]["coding_agent"]
        whole.append(dict(s["all"], arm=arm, lanes=lanes, elapsed_s=s["elapsed_s"], full_load_s=s["full_load_s"]))
        for p, row in s["programs"].items():
            per_prog.append(dict(row, arm=arm, lanes=lanes, program=p))
        try:
            w, rows = steady_window(level_dir, start_frac)
        except (FileNotFoundError, SystemExit) as e:
            print(f"{arm}/mix_L{lanes}: no steady window ({e})")
            continue
        steady.append(dict(w, arm=arm, lanes=lanes))
        for r in rows:
            steady_prog.append(dict(r, arm=arm, lanes=lanes))
    for rows in (whole, per_prog, steady, steady_prog):
        rows.sort(key=lambda r: (arm_key(r["arm"]), r["lanes"], r.get("program", "")))
    write_csv(os.path.join(results, "mixed.csv"), whole)
    write_csv(os.path.join(results, "mixed_programs.csv"), per_prog)
    write_csv(os.path.join(results, "mixed_steady.csv"), steady)
    write_csv(os.path.join(results, "mixed_steady_programs.csv"), steady_prog)


def steady_window(level_dir: str, start_frac: float):
    """Throughput and cache split of the requests whose prefill started inside the progress window."""
    sess = [json.loads(l) for l in open(os.path.join(level_dir, "sessions.jsonl")) if l.strip()]
    sess = [r for r in sess if r["phase"] == "measured"]
    prog = {str(r["pid"]): r["program"] for r in sess}
    t0 = min(r["t0"] for r in sess)
    fixed = [r for r in sess if r["fixed"]]
    if not fixed:
        raise SystemExit("no fixed lanes")
    total = len(fixed)
    start_idx = int(total * start_frac)
    first = [r["t0"] for r in fixed if r["idx"] == start_idx]
    if not first:
        raise SystemExit(f"no fixed session with idx {start_idx}")
    lo = first[0] - t0
    hi = min(max(r["t1"] for r in fixed if r["lane"] == lane) for lane in {r["lane"] for r in fixed}) - t0
    if hi <= lo:
        raise SystemExit(f"window empty: lo={lo:.0f} hi={hi:.0f}")

    ev: Dict[str, dict] = defaultdict(dict)
    for l in open(os.path.join(level_dir, "requests.jsonl")):
        e = json.loads(l)
        ev[e["request_id"]][e["event"]] = e
    calls = []
    for e in ev.values():
        if "first_token" not in e or "request_start" not in e or "request_admitted" not in e:
            continue
        sid = e["request_admitted"]["session_id"]
        t = e["request_start"]["time_ns"] / 1e9 - t0
        if sid in prog and lo < t <= hi:
            calls.append((prog[sid], e["first_token"], e.get("request_end", {})))
    minutes = (hi - lo) / 60

    def agg(cs, **extra):
        prompt = sum(c[1]["prompt_tokens"] for c in cs)
        cached = sum(c[1]["cached_tokens"] for c in cs)
        out = sum(c[2].get("output_tokens", 0) for c in cs)
        ttft = [c[1]["ttft_ms"] for c in cs]
        return dict(extra, window_lo_s=round(lo), window_hi_s=round(hi), window_s=round(hi - lo), calls=len(cs),
                    calls_per_min=round(len(cs) / minutes, 2),
                    prompt_ktokens_per_min=round(prompt / 1000 / minutes, 1),
                    recompute_ktokens_per_min=round((prompt - cached) / 1000 / minutes, 1),
                    output_ktokens_per_min=round(out / 1000 / minutes, 1),
                    device_pct=round(100 * sum(c[1]["cached_device"] for c in cs) / prompt, 1) if prompt else 0.0,
                    host_pct=round(100 * sum(c[1]["cached_host"] for c in cs) / prompt, 1) if prompt else 0.0,
                    miss_pct=round(100 * (1 - cached / prompt), 1) if prompt else 0.0,
                    ttft_p50_ms=round(q(ttft, 0.5)), ttft_p95_ms=round(q(ttft, 0.95)),
                    ttft_mean_ms=round(statistics.mean(ttft)) if ttft else 0,
                    sessions_done=sum(lo < r["t1"] - t0 <= hi for r in sess))
    rows = [agg([c for c in calls if c[0] == p], program=p) for p in sorted({c[0] for c in calls})]
    return agg(calls, start_idx=start_idx), rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--start-frac", type=float, default=0.5,
                    help="steady window opens at the fixed program's session idx = total * frac")
    a = ap.parse_args()
    sweeps(a.results)
    mixed(a.results, a.start_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
