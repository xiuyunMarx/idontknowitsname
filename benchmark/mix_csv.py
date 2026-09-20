"""Per-arm summary CSV of one mixed-workload configuration.

    python -m benchmark.mix_csv [--prefix mix20h10] [--out benchmark/mixed_results/mix/mix_h10.csv]

Reads <prefix>_<arm>/{summary.json,sessions.jsonl} (driver dirs, looked up under mixed_results/mix/ and
mixed_results/cache/) for every arm found
and writes one row per arm:

  throughput      sessions/min over the SAME fixed window for all arms (the shortest full-load window
                  across arms, `window_s`), plus each arm's own full-load window and done counts.
  jct_w_mean_s    lane-weighted JCT: sum_p (lanes_p / lanes) * mean JCT_p. Programs differ in JCT
                  (coding ~10x BFCL), so a plain session mean would be dominated by whichever program
                  completed more sessions in follow mode; lane weights are fixed by the configuration.
  jct_w_p50_s     the same weighting applied to the per-program p50.
  jct_norm        lane-weighted mean of JCT_p / JCT_p(reference arm); reference = vanilla if present
                  else relayout (column `jct_ref`). 1.0 = the reference, 0.8 = 20% faster.
  ttft_w_mean_ms  lane-weighted mean TTFT; ttft_w_p50_ms likewise; ttft_mean_ms = call-weighted.
  device_pct, host_pct, miss_pct   cache statistics pooled over all calls (call-weighted).
  per-program columns <metric>_<prog>: jct_mean_s, jct_p50_s, ttft_mean_ms, miss_pct, host_pct,
                  device_pct, sessions, failed, accuracy.
A second file <out stem>_programs.csv keeps the full per-program rows (tag, program, every summary field).
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(HERE, "mixed_results", "mix"), os.path.join(HERE, "mixed_results", "cache")]
SHORT = {"fact_check": "fact", "BFCL_agent": "bfcl", "coding_agent": "coding"}
ARM_ORDER = ["vanilla", "kvonly", "relayout", "ours", "continuum", "cachescout"]


def load(prefix: str) -> dict:
    arms = {}
    for d in sorted(d for base in DIRS for d in glob.glob(os.path.join(base, f"{prefix}_*")) if os.path.isdir(d)):
        arm = os.path.basename(d)[len(prefix) + 1:]
        if arm in arms:
            continue
        try:
            summ = json.load(open(os.path.join(d, "summary.json")))
            recs = [json.loads(l) for l in open(os.path.join(d, "sessions.jsonl"))]
        except FileNotFoundError:
            continue
        arms[arm] = (summ, [r for r in recs if r["phase"] == "measured"])
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mix20h10")
    ap.add_argument("--out", default=os.path.join(HERE, "mixed_results", "mix", "mix_h10.csv"))
    a = ap.parse_args()
    arms = load(a.prefix)
    if not arms:
        print(f"no {a.prefix}_* dirs under {DIRS}")
        return 1
    window = min(s["full_load_s"] for s, _ in arms.values())
    ref = "vanilla" if "vanilla" in arms else ("relayout" if "relayout" in arms else None)
    order = sorted(arms, key=lambda x: ARM_ORDER.index(x) if x in ARM_ORDER else 99)
    rows, prog_rows = [], []
    for arm in order:
        s, recs = arms[arm]
        lanes = s["lanes"]
        tot = sum(lanes.values())
        progs = s["programs"]
        w = {p: lanes[p] / tot for p in progs}
        calls = {p: progs[p]["calls"] for p in progs}
        ncalls = sum(calls.values())
        done_w = sum(r["t1"] - s["start"] <= window for r in recs)
        row = {
            "arm": arm, "tag": s["tag"], "window_s": window,
            "throughput": round(done_w / (window / 60), 2), "done_window": done_w,
            "own_window_s": s["full_load_s"], "own_throughput": s["all"]["sessions_per_min_full_load"],
            "done_600s": s["all"]["done_600s"], "done_1200s": s["all"]["done_1200s"],
            "sessions": sum(v["sessions"] for v in progs.values()), "failed": sum(v["failed"] for v in progs.values()),
            "jct_w_mean_s": round(sum(w[p] * progs[p]["jct_mean_s"] for p in progs), 1),
            "jct_w_p50_s": round(sum(w[p] * progs[p]["jct_p50_s"] for p in progs), 1),
            "jct_ref": ref or "",
            "jct_norm": round(sum(w[p] * progs[p]["jct_mean_s"] / arms[ref][0]["programs"][p]["jct_mean_s"] for p in progs), 3) if ref else "",
            "ttft_w_mean_ms": round(sum(w[p] * progs[p]["ttft_mean_ms"] for p in progs)),
            "ttft_w_p50_ms": round(sum(w[p] * progs[p]["ttft_p50_ms"] for p in progs)),
            "ttft_mean_ms": round(sum(calls[p] * progs[p]["ttft_mean_ms"] for p in progs) / ncalls),
            "calls": ncalls,
        }
        for k in ("device_pct", "host_pct", "miss_pct"):
            row[k] = round(sum(calls[p] * progs[p][k] for p in progs) / ncalls, 1)
        for p in progs:
            sp = SHORT.get(p, p)
            for k in ("jct_mean_s", "jct_p50_s", "ttft_mean_ms", "miss_pct", "host_pct", "device_pct", "sessions", "failed", "accuracy"):
                row[f"{k}_{sp}"] = progs[p][k]
            prog_rows.append(dict(arm=arm, tag=s["tag"], program=p, lanes=lanes[p], **progs[p]))
        rows.append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    pout = os.path.splitext(a.out)[0] + "_programs.csv"
    with open(pout, "w", newline="") as f:
        fields = ["arm", "tag", "program", "lanes"] + sorted({k for r in prog_rows for k in r} - {"arm", "tag", "program", "lanes"})
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(prog_rows)
    for r in rows:
        print(f"[{r['arm']}] window {window} s: {r['throughput']} sess/min ({r['done_window']} done); "
              f"JCT lane-weighted mean {r['jct_w_mean_s']} s (norm {r['jct_norm']} vs {ref}); "
              f"TTFT lane-weighted mean {r['ttft_w_mean_ms']} ms; device/host/miss {r['device_pct']}/{r['host_pct']}/{r['miss_pct']}%")
    print(f"wrote {a.out} and {pout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
