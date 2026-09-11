"""Recompute (prefill-from-scratch) figures for the concurrency sweeps.

Reads the per-call [serve] lines of each DIR/{arm}_c{N}.log (joined with [call] labels
by fact_bench.parse_log), computes recompute tokens = prompt - cached_device - cached_host
per call, and writes:

  recompute_vs_c_<wl>.{pdf,png}            recompute fraction vs concurrency, all calls / invocation >= 2
  recompute_by_invocation_<wl>.{pdf,png}   recompute tokens per call vs invocation index of the main site,
                                           one panel per turning-point concurrency, prompt length in grey
  recompute_timeline_<wl>_c<N>.{pdf,png}   rolling recompute fraction over the run (miss waves)
  recompute_cdf_<wl>.{pdf,png}             CDF of recompute tokens per call (invocation >= 2)
  recompute_table.csv                      per (workload, c, arm, site, invocation) stats behind the figures

usage: python benchmark/recompute_figs.py [--out results/figs_recompute]
"""
import argparse, csv, glob, os, re, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fact_check"))
from fact_bench import parse_log, CALL_RE  # noqa: E402

WORKLOADS = {
    "coding": dict(dir="results/coding_sweep", title="Coding agent (host 54k)",
                   main_site="Coder.implement", turning=[12, 28, 40], timeline_c=28),
    "fact": dict(dir="results/fact_sweep_h16_inputfirst", title="Fact check, input-first (host 108k)",
                 main_site="EvidenceAnalyzer.verify_claim", turning=[8, 12, 16], timeline_c=12),
}
# fixed order and fixed hue per arm (never cycled)
ARMS = [("lruraw", "Vanilla SGLang", "#6E6E6E", "-"),
        ("cachescout", "CacheScout", "#B5589A", "-"),
        ("kvonly", "Planner only", "#009E73", "-"),
        ("lru", "Relayout only", "#0072B2", "-"),
        ("ours", "Ours", "#D55E00", "-")]
ARM_LABEL = {a: l for a, l, _, _ in ARMS}
ARM_COLOR = {a: c for a, _, c, _ in ARMS}
MARKERS = {"lruraw": "o", "cachescout": "v", "kvonly": "^", "lru": "s", "ours": "D"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
                     "pdf.fonttype": 42})


def measured_pids(log):
    """pids in order of first [call]; the first two are the sequential warmup sessions."""
    seen = []
    with open(log) as f:
        for ln in f:
            m = CALL_RE.match(ln)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
    return set(seen[2:])


def load(wl):
    """{(c, arm): [serve dict, ...]} with recompute tokens added, in log order."""
    runs = {}
    for log in sorted(glob.glob(os.path.join(WORKLOADS[wl]["dir"], "*_c*.log"))): #type: ignore
        tag = os.path.basename(log)[:-4]
        arm, _, c = tag.rpartition("_c")
        if arm not in ARM_LABEL or not c.isdigit():
            continue
        serves, _ = parse_log(log, measured_pids(log))
        # parse_log sorts by (pid, call); restore log order for the timeline
        order = {}
        with open(log) as f:
            for i, ln in enumerate(f):
                if ln.startswith("[serve] "):
                    key = ln.split()[1].rsplit("-", 1)[0]
                    order.setdefault(key, i)
        for s in serves:
            s["recompute"] = s["prompt_tokens"] - s["cached_device"] - s["cached_host"]
            s["order"] = order.get(f"{s['pid']}-{s['call']}t{s['turn']}", 0)
        runs[(int(c), arm)] = sorted(serves, key=lambda s: s["order"])
    return runs


def ratio(ss):
    p = sum(s["prompt_tokens"] for s in ss)
    return sum(s["recompute"] for s in ss) / p if p else float("nan")


def fig_legend(fig, ax, ncol=5, y=-0.02):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=7.5, handlelength=2.2)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_vs_c(wl, runs, out):
    cs = sorted({c for c, _ in runs})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, (title, pred) in zip(axes, [("all calls", lambda s: True),
                                        ("invocation ≥ 2 (re-visits only)", lambda s: s["inv"] >= 2)]):
        for arm, label, color, ls in ARMS:
            ys = [100 * ratio([s for s in runs[(c, arm)] if pred(s)]) if (c, arm) in runs else np.nan for c in cs]
            ax.plot(cs, ys, ls, color=color, marker=MARKERS[arm], label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("concurrency")
        ax.set_xticks(cs)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("recomputed prompt tokens (%)")
    fig_legend(fig, axes[1])
    fig.suptitle(WORKLOADS[wl]["title"], fontsize=9, y=1.02)
    save(fig, out, f"recompute_vs_c_{wl}")


def fig_by_invocation(wl, runs, out, min_n=10):
    site = WORKLOADS[wl]["main_site"]
    cs = WORKLOADS[wl]["turning"]
    fig, axes = plt.subplots(1, len(cs), figsize=(2.3 * len(cs), 2.5), sharey=True)
    for ax, c in zip(axes, cs):
        for arm, label, color, ls in ARMS:
            ss = [s for s in runs.get((c, arm), []) if s["site"] == site]
            invs = sorted({s["inv"] for s in ss})
            xs, ys, ps = [], [], []
            for inv in invs:
                g = [s for s in ss if s["inv"] == inv]
                if len(g) < min_n:
                    break
                xs.append(inv); ys.append(np.mean([s["recompute"] for s in g])); ps.append(np.mean([s["prompt_tokens"] for s in g]))
            if xs:
                ax.plot(xs, ys, ls, color=color, marker=MARKERS[arm], label=label)
            if arm == "ours":
                ax.plot(xs, ps, "--", color="#9A9A9A", lw=1.1, label="prompt length")
        ax.set_title(f"c = {c}", fontsize=9)
        ax.set_xlabel("invocation # within session")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    top = max(l.get_ydata().max() for ax in axes for l in ax.get_lines() if len(l.get_ydata()))
    axes[0].set_ylim(0, 1.05 * top)
    axes[0].set_ylabel("tokens per call")
    fig_legend(fig, axes[-1], ncol=6)
    fig.suptitle(f"{WORKLOADS[wl]['title']}: recomputed tokens vs prompt growth, site {site}", fontsize=9, y=1.03)
    save(fig, out, f"recompute_by_invocation_{wl}")


def fig_timeline(wl, runs, out):
    c = WORKLOADS[wl]["timeline_c"]
    w = 3 * c
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for arm, label, color, ls in ARMS:
        ss = runs.get((c, arm))
        if not ss:
            continue
        rec = np.array([s["recompute"] for s in ss], float)
        pr = np.array([s["prompt_tokens"] for s in ss], float)
        k = np.ones(w)
        y = 100 * np.convolve(rec, k, "valid") / np.convolve(pr, k, "valid")
        x = np.arange(len(y)) + w
        ax.plot(x, y, ls, color=color, label=label)
    ax.set_xlabel(f"LLM call index over the run (rolling window of {w} calls)")
    ax.set_ylabel("recomputed prompt tokens (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{WORKLOADS[wl]['title']}, c = {c}", fontsize=9)
    fig_legend(fig, ax, y=-0.08)
    save(fig, out, f"recompute_timeline_{wl}_c{c}")


def fig_cdf(wl, runs, out):
    cs = WORKLOADS[wl]["turning"]
    fig, axes = plt.subplots(1, len(cs), figsize=(2.3 * len(cs), 2.5), sharey=True)
    for ax, c in zip(axes, cs):
        for arm, label, color, ls in ARMS:
            v = np.sort([s["recompute"] for s in runs.get((c, arm), []) if s["inv"] >= 2])
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=color, label=label)
        ax.set_title(f"c = {c}", fontsize=9)
        ax.set_xlabel("recomputed tokens per call")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("CDF (invocation ≥ 2)")
    fig_legend(fig, axes[-1])
    fig.suptitle(WORKLOADS[wl]["title"], fontsize=9, y=1.02)
    save(fig, out, f"recompute_cdf_{wl}")


def table_rows(wl, runs):
    for (c, arm), ss in sorted(runs.items()):
        groups = defaultdict(list)
        for s in ss:
            groups[(s["site"], s["inv"])].append(s)
            groups[(s["site"], "all")].append(s)
            groups[("all", "all")].append(s)
            if s["inv"] >= 2:
                groups[("all", ">=2")].append(s)
        for (site, inv), g in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            rec = [s["recompute"] for s in g]
            yield dict(workload=wl, concurrency=c, arm=arm, site=site, invocation=inv, n=len(g),
                       prompt_mean=round(np.mean([s["prompt_tokens"] for s in g])),
                       recompute_mean=round(np.mean(rec)), recompute_p50=round(np.median(rec)),
                       recompute_pct=round(100 * ratio(g), 1),
                       device_pct=round(100 * sum(s["cached_device"] for s in g) / sum(s["prompt_tokens"] for s in g), 1),
                       host_pct=round(100 * sum(s["cached_host"] for s in g) / sum(s["prompt_tokens"] for s in g), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/figs_recompute")
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for wl in args.workloads:
        runs = load(wl)
        print(f"{wl}: {len(runs)} runs, c={sorted({c for c, _ in runs})}")
        fig_vs_c(wl, runs, args.out)
        fig_by_invocation(wl, runs, args.out)
        fig_timeline(wl, runs, args.out)
        fig_cdf(wl, runs, args.out)
        rows.extend(table_rows(wl, runs))
    with open(os.path.join(args.out, "recompute_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} table rows and figures to {args.out}/")


if __name__ == "__main__":
    main()
