"""Per-call cache status for every finished run: one CSV row per HTTP request.

    python -m benchmark.export_calls [--tags fact_ours_c12,...] [--out benchmark/mixed_results/calls]

For each driver tag under mixed_results/cache/<tag>/ (sessions.jsonl gives the sessions'
pids), the server log that served it is found among mixed_results/*/logs/*.log and
mixed_results/mix/*.log, and its [call] / [serve] / [decode] rows are joined by pid:

    tag, program, arm, c, idx, pid, call, turn, site, prompt_tokens, cached_device,
    cached_host, recomputed, recomputed_frac, ttft_ms, output_tokens, decode_ms

`recomputed` = prompt_tokens - cached_device - cached_host. turn > 0 is a typed-output
retry of the same call. Also prints, per tag, recomputed/all over every call.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "mixed_results")

SERVE = re.compile(r"^\[serve\] (\d+)-(\d+)t(\d+)-\S+ prompt_tokens=(\d+) cached_tokens=(\d+) "
                   r"cached_device=(\d+) cached_host=(\d+) ttft_ms=([\d.]+)")
DECODE = re.compile(r"^\[decode\] (\d+)-(\d+)t(\d+)-\S+ decode_ms=([\d.]+) output_tokens=(\d+)")
CALL = re.compile(r"^\[call\] (\d+) #(\d+) (\S+)")
TAG = re.compile(r"^(?:evict_|reserve_)?([a-z]+)_([a-z_]+?)(?:_gate|_r6400)?_c(\d+)$")


def server_logs():
    return sorted(set(glob.glob(os.path.join(RESULTS, "*", "logs", "*server*.log"))
                      + glob.glob(os.path.join(RESULTS, "mix", "*server*.log"))))


def scan(log):
    """pid -> {(call, turn): serve fields}, pid -> {(call, turn): decode fields}, pid -> {call: site}"""
    serve, decode, site = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    with open(log, errors="replace") as f:
        for line in f:
            m = SERVE.match(line)
            if m:
                pid, call, turn = int(m.group(1)), int(m.group(2)), int(m.group(3))
                serve[pid][(call, turn)] = (int(m.group(4)), int(m.group(6)), int(m.group(7)), float(m.group(8)))
                continue
            m = DECODE.match(line)
            if m:
                decode[int(m.group(1))][(int(m.group(2)), int(m.group(3)))] = (float(m.group(4)), int(m.group(5)))
                continue
            m = CALL.match(line)
            if m:
                site[int(m.group(1))][int(m.group(2)) + 1] = m.group(3)
    return serve, decode, site


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="", help="comma-separated; default every tag under mixed_results/cache")
    ap.add_argument("--out", default=os.path.join(RESULTS, "calls"))
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(RESULTS, "cache", "*")))
    os.makedirs(a.out, exist_ok=True)
    logs = {log: None for log in server_logs()}          # scanned lazily
    summary = []
    for tag in tags:
        sj = os.path.join(RESULTS, "cache", tag, "sessions.jsonl")
        if not os.path.exists(sj):
            continue
        sess = {s["pid"]: s for s in map(json.loads, open(sj)) if s["phase"] == "measured"}
        if not sess:
            continue
        # the server log holding the most of this tag's pids
        best, best_n = None, 0
        for log in logs:
            if logs[log] is None:
                logs[log] = scan(log)
            n = sum(1 for pid in sess if pid in logs[log][0])
            if n > best_n:
                best, best_n = log, n
        if best is None or best_n < len(sess) // 2:
            print(f"[skip] {tag}: no server log covers its sessions", flush=True)
            continue
        serve, decode, site = logs[best]
        m = TAG.match(tag)
        prefix, arm, c = (m.group(1), m.group(2), int(m.group(3))) if m else (tag, "", 0)
        rows = []
        for pid, s in sess.items():
            for (call, turn), (prompt, dev, host, ttft) in sorted(serve.get(pid, {}).items()):
                dms, out = decode.get(pid, {}).get((call, turn), (0.0, 0))
                rec = prompt - dev - host
                rows.append(dict(tag=tag, program=s["program"], arm=arm, c=c, idx=s["idx"], pid=pid, call=call, turn=turn,
                                 site=site.get(pid, {}).get(call, ""), prompt_tokens=prompt, cached_device=dev,
                                 cached_host=host, recomputed=rec, recomputed_frac=round(rec / prompt, 4) if prompt else 0,
                                 ttft_ms=ttft, output_tokens=out, decode_ms=dms))
        path = os.path.join(a.out, f"{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        tot = sum(r["prompt_tokens"] for r in rows)
        rec = sum(r["recomputed"] for r in rows)
        summary.append((tag, len(rows), tot, rec, os.path.basename(best)))
        print(f"{tag:28} calls={len(rows):5} prompt={tot:9} recomputed={rec:9} recomputed/all={rec/tot:.3f}  <- {os.path.basename(best)}", flush=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "calls", "prompt_tokens", "recomputed", "recomputed_frac", "server_log"])
        for tag, n, tot, rec, log in summary:
            w.writerow([tag, n, tot, rec, round(rec / tot, 4), log])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
