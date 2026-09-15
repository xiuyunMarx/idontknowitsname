"""Consolidate mixed_workload runs into one CSV, one row per (run, mix, arm, program).

usage: python -m benchmark.mixed_workload.sweep_csv OUT.csv LABEL:MODEL:DIR [LABEL:MODEL:DIR ...]
Reads DIR/{arm}_f{F}c{C}.out (run.py output) and the matching .log ([ledger] capacity).
Arms are recorded under the paper spelling: lruraw -> sglang, lru -> reorder-only."""
import csv, glob, os, re, sys

ARMS = {"lruraw": "sglang", "lru": "reorder-only", "kvonly": "kvonly", "ours": "ours",
        "cachescout": "cachescout", "continuum": "continuum"}
FIELDS = ["run", "model", "fact_lanes", "code_lanes", "concurrency", "arm", "program", "device_tokens", "host_tokens",
          "sessions", "failed", "jct_p50_s", "jct_p95_s", "jct_mean_s",
          "jct_overlap_p50_s", "jct_overlap_p95_s", "jct_overlap_mean_s", "overlap_n",
          "ttft_p50_ms", "ttft_p95_ms", "ttft_p50_inv2_ms", "calls", "prompt_tokens_per_call",
          "device_pct", "host_pct", "miss_pct", "accuracy", "programs_registered", "opaque_requests"]
ROW_RE = r"\s+(\d+)\s+(\d+)ms\s+(\d+)ms\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"


def cache_cols(txt, prefix):
    out = {}
    m = re.search(re.escape(prefix) + r" all " + ROW_RE, txt)
    if m:
        out.update(calls=m.group(1), ttft_p50_ms=m.group(2), ttft_p95_ms=m.group(3), prompt_tokens_per_call=m.group(4),
                   device_pct=m.group(5), host_pct=m.group(6), miss_pct=m.group(7))
    m = re.search(re.escape(prefix) + r" all, invocation >= 2" + ROW_RE, txt)
    if m:
        out["ttft_p50_inv2_ms"] = m.group(2)
    return out


def rows_for(label, model, path):
    name = os.path.basename(path)[:-4]
    m = re.match(r"([a-z_]+)_f(\d+)c(\d+)$", name)
    if not m or m.group(1) not in ARMS:
        return []
    arm, f, c = ARMS[m.group(1)], int(m.group(2)), int(m.group(3))
    txt = open(path).read()
    base = {"run": label, "model": model, "fact_lanes": f, "code_lanes": c, "concurrency": f + c, "arm": arm}
    mm = re.search(r"programs registered=(\d+) .* opaque_requests=(\d+)", txt)
    if mm:
        base["programs_registered"], base["opaque_requests"] = mm.group(1), mm.group(2)
    log = path[:-4] + ".log"
    if os.path.exists(log):
        for l in open(log):
            if l.startswith("[ledger] device="):
                a = re.search(r"device=(\d+) tokens host=(\d+) tokens", l)
                base["device_tokens"], base["host_tokens"] = a.group(1), a.group(2)
                break
    rows = []
    for prog in ("fact", "code"):
        prefix = re.search(r"^(\[[^\]]*/" + prog + r"\])", txt, re.M)
        if not prefix:
            continue
        p = prefix.group(1)
        r = dict(base, program=prog)
        mm = re.search(re.escape(p) + r" ==== (\d+) measured sessions, failed=(\d+)", txt)
        if not mm:
            continue
        r["sessions"], r["failed"] = mm.group(1), mm.group(2)
        mm = re.search(re.escape(p) + r" JCT p50=\s*([\d.]+)s p95=\s*([\d.]+)s mean=\s*([\d.]+)s.*accuracy=(\d+/\d+)", txt)
        if mm:
            r.update(jct_p50_s=mm.group(1), jct_p95_s=mm.group(2), jct_mean_s=mm.group(3), accuracy=mm.group(4))
        mm = re.search(re.escape(p) + r" JCT overlap \(\d+s window\) p50=\s*([\d.]+)s p95=\s*([\d.]+)s mean=\s*([\d.]+)s n=(\d+)", txt)
        if mm:
            r.update(jct_overlap_p50_s=mm.group(1), jct_overlap_p95_s=mm.group(2), jct_overlap_mean_s=mm.group(3), overlap_n=mm.group(4))
        r.update(cache_cols(txt, p))
        rows.append(r)
    both = re.search(r"^(\[[^\]]*\]) server-wide cache", txt, re.M)
    if both:
        r = dict(base, program="both")
        r.update(cache_cols(txt, both.group(1)))
        rows.append(r)
    return rows


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    rows = []
    for spec in specs:
        label, model, d = spec.split(":", 2)
        for p in sorted(glob.glob(os.path.join(d, "*_f*c*.out"))):
            rows.extend(rows_for(label, model, p))
    rows.sort(key=lambda r: (r["run"], r["concurrency"], r["fact_lanes"], r["arm"], r["program"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"{out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
