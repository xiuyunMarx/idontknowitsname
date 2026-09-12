"""Per-invocation recompute / cached token tables for both sweeps, all call sites, all arms.

recompute_by_site_invocation.csv : one row per (workload, concurrency, site, invocation) and per site total;
                                   per arm: n, prompt, recompute, device, host  (mean tokens per call)
recompute_by_call_index.csv      : one row per (workload, concurrency, call index within the session), all sites
                                   pooled, same per-arm columns (CacheScout included; it has no site labels)

usage: python benchmark/recompute_csv.py [--out results/figs_recompute]
"""
import argparse, csv, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_figs import WORKLOADS, ARMS, load  # noqa: E402

ARM_ORDER = [a for a, _, _, _ in ARMS]
COLS = ["n", "prompt", "recompute", "device", "host"]


def stats(g):
    n = len(g)
    return dict(n=n, prompt=round(sum(s["prompt_tokens"] for s in g) / n),
                recompute=round(sum(s["recompute"] for s in g) / n),
                device=round(sum(s["cached_device"] for s in g) / n),
                host=round(sum(s["cached_host"] for s in g) / n))


def wide(key_names, groups):
    """groups: {(key..., arm): [serves]} -> rows keyed by key with per-arm columns."""
    keys = sorted({k[:-1] for k in groups}, key=lambda k: tuple((0, x) if isinstance(x, int) else (1, str(x)) for x in k))
    for k in keys:
        row = dict(zip(key_names, k))
        for arm in ARM_ORDER:
            g = groups.get(k + (arm,))
            for c in COLS:
                row[f"{arm}_{c}"] = stats(g)[c] if g else ""
        yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/figs_recompute")
    args = ap.parse_args()
    by_site, by_call = defaultdict(list), defaultdict(list)
    for wl in WORKLOADS:
        for (c, arm), ss in load(wl).items():
            for s in ss:
                if s["site"] not in ("?", ""):
                    by_site[(wl, c, s["site"], s["inv"], arm)].append(s)
                    by_site[(wl, c, s["site"], "all", arm)].append(s)
                by_call[(wl, c, s["call"], arm)].append(s)
    fields = lambda keys: keys + [f"{a}_{c}" for a in ARM_ORDER for c in COLS]
    for name, keys, groups in (("recompute_by_site_invocation.csv", ["workload", "concurrency", "site", "invocation"], by_site),
                               ("recompute_by_call_index.csv", ["workload", "concurrency", "call_index"], by_call)):
        path = os.path.join(args.out, name)
        rows = list(wide(keys, groups))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields(keys)); w.writeheader(); w.writerows(rows)
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
