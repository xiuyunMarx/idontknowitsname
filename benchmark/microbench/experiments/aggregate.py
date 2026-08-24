"""Aggregate one run_experiment.py results directory into markdown tables.

Parses each condition's server.log:
    [serve]   <program>:<site>:<id>-<hex> duration_ms=T cached_tokens=C prompt_tokens=P
    [decode]  <same id> decode_ms=D output_tokens=N
    [prefill] prefill-... cost_tokens=K duration_ms=...
plus clients.jsonl for end-to-end wall times.

Usage: python experiments/aggregate.py <results_dir>
"""

import json
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SERVE = re.compile(r"\[serve\] (\S+) duration_ms=([\d.]+) cached_tokens=(\d+) prompt_tokens=(\d+)")
DECODE = re.compile(r"\[decode\] (\S+) decode_ms=([\d.]+) output_tokens=(\d+)")
PREFILL = re.compile(r"\[prefill\] (\S+) cost_tokens=(\d+) duration_ms=([\d.]+)")


def parse_condition(cond_dir):
    serves, tbts, spec_tokens, spec_count = [], [], 0, 0
    with open(os.path.join(cond_dir, "server.log")) as f:
        for line in f:
            line = ANSI.sub("", line)
            if m := SERVE.search(line):
                rid, ttft, cached, prompt = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
                tenant, site = (rid.split(":", 2) + ["?", "?"])[:2]
                serves.append({"tenant": tenant, "site": site, "ttft": ttft,
                               "cached": cached, "prompt": prompt})
            elif m := DECODE.search(line):
                if (n := int(m.group(3))) > 1:
                    tbts.append(float(m.group(2)) / (n - 1))
            elif m := PREFILL.search(line):
                spec_count += 1
                spec_tokens += int(m.group(2))
    walls = {}
    clients = os.path.join(cond_dir, "clients.jsonl")
    if os.path.exists(clients):
        with open(clients) as f:
            for line in f:
                row = json.loads(line)
                walls.setdefault(row["tenant"], []).append(row["wall"])
    return {"serves": serves, "tbts": tbts, "spec_tokens": spec_tokens,
            "spec_count": spec_count, "walls": walls}


def pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    root = sys.argv[1]
    conds = [d for d in ("off", "newest-idle", "global-idle", "global-budget")
             if os.path.exists(os.path.join(root, d, "server.log"))]
    data = {c: parse_condition(os.path.join(root, c)) for c in conds}

    print("## Overall\n")
    print("| condition | serves | TTFT p50 (ms) | TTFT p95 (ms) | cache hit | mean TBT (ms) | spec prefills | spec tokens | E2E p50 (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        d = data[c]
        ttfts = [s["ttft"] for s in d["serves"]]
        cached = sum(s["cached"] for s in d["serves"])
        prompt = sum(s["prompt"] for s in d["serves"])
        walls = [w for ws in d["walls"].values() for w in ws]
        hit = f"{cached / prompt:.1%}" if prompt else "-"
        tbt = f"{statistics.mean(d['tbts']):.2f}" if d["tbts"] else "-"
        print(f"| {c} | {len(ttfts)} | {pct(ttfts, 0.5):.1f} | {pct(ttfts, 0.95):.1f} | {hit} | {tbt} | {d['spec_count']} | {d['spec_tokens']} | {pct(walls, 0.5):.1f} |")

    print("\n## TTFT p50 (ms) per tenant\n")
    tenants = sorted({s["tenant"] for d in data.values() for s in d["serves"]})
    print("| tenant | " + " | ".join(conds) + " |")
    print("|---|" + "---|" * len(conds))
    for t in tenants:
        row = [f"{pct([s['ttft'] for s in data[c]['serves'] if s['tenant'] == t], 0.5):.1f}" for c in conds]
        print(f"| {t} | " + " | ".join(row) + " |")

    print("\n## Cache-hit ratio per tenant\n")
    print("| tenant | " + " | ".join(conds) + " |")
    print("|---|" + "---|" * len(conds))
    for t in tenants:
        row = []
        for c in conds:
            ss = [s for s in data[c]["serves"] if s["tenant"] == t]
            p = sum(s["prompt"] for s in ss)
            row.append(f"{sum(s['cached'] for s in ss) / p:.1%}" if p else "-")
        print(f"| {t} | " + " | ".join(row) + " |")

    print("\n## TTFT p50 (ms) per callsite (top by volume)\n")
    sites = sorted({(s["tenant"], s["site"]) for d in data.values() for s in d["serves"]})
    print("| tenant:site | " + " | ".join(conds) + " |")
    print("|---|" + "---|" * len(conds))
    for t, site in sites:
        row = [f"{pct([s['ttft'] for s in data[c]['serves'] if s['tenant'] == t and s['site'] == site], 0.5):.1f}" for c in conds]
        print(f"| {t}:{site} | " + " | ".join(row) + " |")

    if "off" in data and data["off"]["tbts"]:
        base = statistics.mean(data["off"]["tbts"])
        print("\n## TBT no-harm check (vs off, +10% tolerated)\n")
        for c in conds:
            if data[c]["tbts"]:
                m = statistics.mean(data[c]["tbts"])
                flag = "OK" if m <= base * 1.10 else "VIOLATION"
                print(f"- {c}: {m:.2f} ms ({m / base - 1:+.1%}) {flag}")


if __name__ == "__main__":
    main()
