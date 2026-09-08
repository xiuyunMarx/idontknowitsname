"""Consolidate fact_bench sweep arms into one CSV.

usage: python -m benchmark.fact_check.sweep_csv OUT.csv LABEL:MODEL:DIR [LABEL:MODEL:DIR ...]
One row per (label, concurrency, arm) from DIR/{lruraw,lru,ours}_c{N}.out and the matching .log
(TTFT mean and decode figures come from the server log's [serve]/[decode] lines)."""
import csv, glob, os, re, sys

FIELDS = ["run", "model", "concurrency", "arm", "device_tokens", "host_tokens", "sessions", "failed",
          "jct_p50_s", "jct_p95_s", "jct_mean_s", "ttft_p50_ms", "ttft_mean_ms", "ttft_p95_ms", "ttft_p50_inv2_ms",
          "calls", "prompt_tokens_per_call", "device_pct", "host_pct", "miss_pct",
          "output_tokens", "decode_ms_per_token", "demand_loads", "load_ms_p50", "promotions_started", "accuracy"]


def row_for(label, model, path):
    tag = os.path.basename(path)[:-4]
    arm, _, c = tag.rpartition("_c")
    if not c.isdigit() or arm not in ("lruraw", "lru", "ours", "cachescout"):
        return None
    txt = open(path).read()
    m = re.search(r"JCT p50=\s*([\d.]+)s p95=\s*([\d.]+)s mean=\s*([\d.]+)s", txt)
    if not m:
        return None
    r = {"run": label, "model": model, "concurrency": int(c), "arm": arm,
         "jct_p50_s": m.group(1), "jct_p95_s": m.group(2), "jct_mean_s": m.group(3)}
    m = re.search(r"==== (\d+) measured sessions, failed=(\d+)", txt)
    r["sessions"], r["failed"] = m.group(1), m.group(2)
    m = re.search(r"accuracy=(\d+/\d+)", txt); r["accuracy"] = m.group(1) if m else ""
    for name, suffix in (("all ", ""), ("all, invocation >= 2", "_inv2")):
        mm = re.search(r"\] " + re.escape(name) + r"\s+(\d+)\s+(\d+)ms\s+(\d+)ms\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", txt)
        if mm and not suffix:
            r.update(calls=mm.group(1), ttft_p50_ms=mm.group(2), ttft_p95_ms=mm.group(3),
                     prompt_tokens_per_call=mm.group(4), device_pct=mm.group(5), host_pct=mm.group(6), miss_pct=mm.group(7))
        elif mm:
            r["ttft_p50_inv2_ms"] = mm.group(2)
    log = path[:-4] + ".log"
    if os.path.exists(log):
        dec_ms = out_tok = 0; loads = []; started = 0; serves = []
        for l in open(log):
            if l.startswith("[serve]"):
                a = re.search(r"^\[serve\] (\d+)-\S+ .*ttft_ms=([\d.]+)", l)
                if a: serves.append((int(a.group(1)), float(a.group(2))))
            elif l.startswith("[decode]"):
                a = re.search(r"decode_ms=([\d.]+) output_tokens=(\d+)", l)
                if a: dec_ms += float(a.group(1)); out_tok += int(a.group(2))
            elif l.startswith("[ledger] device="):
                a = re.search(r"device=(\d+) tokens host=(\d+) tokens", l)
                r["device_tokens"], r["host_tokens"] = a.group(1), a.group(2)
            elif l.startswith("[load] "):
                a = re.search(r"ms=([\d.]+)", l); loads.append(float(a.group(1)))
            elif l.startswith("[promote] ") and "started=0" not in l:
                started += 1
        warm = set(sorted({p for p, _ in serves})[:2])   # first two pids are the warmup sessions
        measured = [t for p, t in serves if p not in warm]
        r["ttft_mean_ms"] = f"{sum(measured) / len(measured):.0f}" if measured else ""
        r["output_tokens"] = out_tok
        r["decode_ms_per_token"] = f"{dec_ms / out_tok:.2f}" if out_tok else ""
        r["demand_loads"] = len(loads)
        r["load_ms_p50"] = f"{sorted(loads)[len(loads) // 2]:.0f}" if loads else ""
        r["promotions_started"] = started
    return r


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    rows = []
    for spec in specs:
        label, model, d = spec.split(":", 2)
        for p in sorted(glob.glob(os.path.join(d, "*_c*.out"))):
            r = row_for(label, model, p)
            if r: rows.append(r)
    rows.sort(key=lambda r: (r["run"], r["concurrency"], r["arm"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"{out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
