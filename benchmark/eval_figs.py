"""
Evaluation figures for the paper, from the two sweep record CSVs.

  benchmark/records/FactCheck/fact_sweep_inputfirst.csv   (fact_check, input-first, host 16 GB)
  benchmark/records/codeAgent/coding_sweep.csv            (coding_agent, host 8 GB)

Four figures per workload, written to benchmark/records/figs/<name>_<wl>.{pdf,png}:

  speedup_<wl>   grouped bars, speed-up over SGLang of mean JCT (left) and mean TTFT (right)
  jct_<wl>       mean JCT vs concurrency, one line per arm
  ttft_<wl>      mean TTFT vs concurrency, one line per arm
  cache_<wl>     stacked bars per (concurrency, arm): device hits / host hits / misses, in prompt tokens

Arms in fixed order and fixed hue: SGLang, CacheScout, Continuum, KV-only, Reorder-only, Ours.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


WORKLOADS = OrderedDict([
    ("fact",   dict(csv="benchmark/records/FactCheck/fact_sweep_inputfirst.csv", title="Fact Check")),
    ("coding", dict(csv="benchmark/records/codeAgent/coding_sweep.csv",          title="Coding Agent")),
])

# CSV arm names -> figure label
ARM_ALIAS = {
    "sglang": "SGLang",
    "lruraw": "SGLang",
    "cachescout": "CacheScout",
    "continuum": "Continuum",
    "kvonly": "KV-only",
    "reorder-only": "Reorder-only",
    "lru": "Reorder-only",
    "ours": "Ours",
}

ARMS = ["SGLang", "CacheScout", "Continuum", "KV-only", "Reorder-only", "Ours"]
BASELINE = "SGLang"

# arm palette of the reference figure script (Tol-like), Continuum added in purple
COLOR = {
    "SGLang": "#777777",
    "CacheScout": "#0072B2",
    "Continuum": "#AA4499",
    "KV-only": "#E69F00",
    "Reorder-only": "#009E73",
    "Ours": "#CC3311",
}

MARKER = {
    "SGLang": "s",
    "CacheScout": "^",
    "Continuum": "P",
    "KV-only": "D",
    "Reorder-only": "v",
    "Ours": "o",
}

LINESTYLE = {
    "SGLang": "--",
    "CacheScout": "--",
    "Continuum": ":",
    "KV-only": "-.",
    "Reorder-only": "-.",
    "Ours": "-",
}

CACHE = [
    ("device_pct", "Device", "#4C78A8", ""),
    ("host_pct",   "Host",   "#72B7B2", "////"),
    ("miss_pct",   "Miss",   "#D9D9D9", ""),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "legend.frameon": False, "hatch.linewidth": 0.5,
})


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)


def load(path):
    """Return {(concurrency, arm_label): row} for q8b run only."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["run"] != "q8b":
                continue
            arm_raw = r["arm"]
            if arm_raw not in ARM_ALIAS:
                continue
            out[(int(r["concurrency"]), ARM_ALIAS[arm_raw])] = r
    return out


def series(rows, arm, field, cs):
    vals = []
    for c in cs:
        if (c, arm) in rows:
            vals.append(float(rows[(c, arm)][field]))
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out_dir, f"{name}.{ext}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


def fig_lines(wl, rows, cs, field, ylabel, out_dir, name, scale=1.0):
    fig, ax = plt.subplots(figsize=(5.7, 3.8))
    for a in ARMS:
        ys = series(rows, a, field, cs) * scale
        if np.all(np.isnan(ys)):
            continue
        ours = a == "Ours"
        ax.plot(
            cs, ys, label=a, color=COLOR[a], marker=MARKER[a], linestyle=LINESTYLE[a],
            linewidth=2 if ours else 1.5, markersize=5.5,
            markerfacecolor=COLOR[a] if ours else "white", markeredgewidth=1.2,
            zorder=5 if ours else 3,
        )
    clean(ax)
    span = cs[-1] - cs[0]
    ax.set(xlabel="Concurrency", ylabel=ylabel, xticks=cs, xlim=(cs[0] - 0.04 * span, cs[-1] + 0.04 * span))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=2, columnspacing=1, handlelength=2.1)
    fig.tight_layout()
    save(fig, out_dir, name)


def fig_speedup(wl, rows, cs, out_dir, name):
    """
    Two panels sharing one y-axis: speed-up of mean JCT and mean TTFT over SGLang.
    speed-up = SGLang / arm, so > 1 is better; the dashed line is SGLang itself.
    """
    arms = [a for a in ARMS if a != BASELINE]
    n_c = len(cs)
    w = 0.8 / len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), sharey=True)
    top = 1.0
    for ax, (field, title) in zip(axes, (("jct_mean_s", "Mean JCT"), ("ttft_mean_ms", "Mean TTFT"))):
        base = series(rows, BASELINE, field, cs)
        for j, a in enumerate(arms):
            sp = base / series(rows, a, field, cs)
            top = max(top, np.nanmax(sp))
            ax.bar(np.arange(n_c) + (j - (len(arms) - 1) / 2) * w, sp, w * 0.91,
                   color=COLOR[a], edgecolor="#444444", linewidth=0.4, label=a)
        ax.axhline(1.0, color="#444444", linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)
        clean(ax)
        ax.set(xticks=np.arange(n_c), xticklabels=cs, xlim=(-0.51, n_c - 0.49), xlabel="Concurrency")
        ax.set_title(title, pad=6)
        for i in range(n_c - 1):
            ax.axvline(i + 0.5, color="#bbbbbb", lw=0.5, ymax=0.99)
    axes[0].set(ylabel="Speed-up over SGLang", ylim=(0, np.ceil(top * 1.08 * 4) / 4))
    axes[1].tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(facecolor=COLOR[a], edgecolor="#444444", linewidth=0.4, label=a) for a in arms],
               loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=len(arms), fontsize=10)
    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.06, right=0.99, wspace=0.06)
    save(fig, out_dir, name)


CACHE_STYLE = {   # the cache-rate figure follows the reference script's look, not the global rcParams
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "legend.fontsize": 10, "hatch.linewidth": 0.5,
}
CACHE_REF = [("device_pct", "Device", "#4477AA", ""), ("host_pct", "Host", "#66CCAA", "///"),
             ("miss_pct", "Miss", "#E4E4E4", "")]


def fig_cache(wl, rows, cs, out_dir, name):
    """
    One axis, one group per concurrency, one stacked bar per arm inside the group
    (stack bottom to top: device hits, host hits, misses). Arm names are the tick
    labels under every bar, the concurrency number sits under each group, thin
    separators between groups, Device/Host/Miss legend above the plot.
    """
    n_arms = len(ARMS)
    n_c = len(cs)
    w = 0.8 / n_arms
    with plt.rc_context(CACHE_STYLE):
        fig, ax = plt.subplots(figsize=(11.8 * n_c / 6, 4.2))
        positions, names = [], []
        for j, a in enumerate(ARMS):
            x = np.arange(n_c) + (j - (n_arms - 1) / 2) * w
            bottom = np.zeros(n_c)
            for field, label, color, hatch in CACHE_REF:
                v = np.array([float(rows[(c, a)][field]) if (c, a) in rows else np.nan for c in cs])
                ax.bar(x, v, w * 0.91, bottom=bottom, color=color, edgecolor="#444444", linewidth=0.4,
                       hatch=hatch, label=label if j == 0 else None)
                bottom += np.nan_to_num(v)
            positions.extend(x)
            names.extend([a] * n_c)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.55)
        ax.set(ylim=(0, 101), yticks=[0, 20, 40, 60, 80, 100], ylabel="Cache rate (%)", xlim=(-0.51, n_c - 0.49))
        ax.set_xticks(positions, names, rotation=60, ha="right", fontsize=8)
        ax.tick_params(axis="x", length=0, pad=3)
        for i, c in enumerate(cs):
            ax.text(i, -0.36, str(c), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
            if i < n_c - 1:
                ax.axvline(i + 0.5, color="#bbbbbb", lw=0.5, ymax=0.99)
        ax.set_xlabel("Concurrency")
        ax.xaxis.set_label_coords(0.5, -0.50)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
        fig.subplots_adjust(bottom=0.37, top=0.9, left=0.065 * 6 / n_c, right=0.99)
        save(fig, out_dir, name)


def print_speedups(wl, rows, cs):
    """Speed-up = SGLang / arm on mean JCT and mean TTFT, per concurrency and summarised
    across concurrencies (geometric mean, min, max). Also Ours over Reorder-only, i.e.
    what the planner adds on top of the re-layout."""
    def line(label, sp):
        sp = sp[np.isfinite(sp)]
        gm = float(np.exp(np.mean(np.log(sp))))
        cells = " ".join(f"{v:5.2f}" for v in sp)
        print(f"  {label:<26} {cells} | geomean {gm:.2f}  min {sp.min():.2f}  max {sp.max():.2f}")
    print(f"speed-up on {wl} (c = {' '.join(f'{c:>5}' for c in cs)})")
    for field, metric in (("jct_mean_s", "mean JCT"), ("ttft_mean_ms", "mean TTFT")):
        base = series(rows, BASELINE, field, cs)
        for a in ARMS:
            if a == BASELINE:
                continue
            line(f"{metric}: {a} vs SGLang", base / series(rows, a, field, cs))
        line(f"{metric}: Ours vs Reorder-only", series(rows, "Reorder-only", field, cs) / series(rows, "Ours", field, cs))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="benchmark/records/figs")
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for wl in args.workloads:
        rows = load(WORKLOADS[wl]["csv"])
        cs = sorted({c for c, _ in rows})

        fig_speedup(wl, rows, cs, args.out, f"speedup_{wl}")
        fig_lines(wl, rows, cs, "jct_mean_s", "Mean JCT (s)", args.out, f"jct_{wl}")
        fig_lines(wl, rows, cs, "ttft_mean_ms", "Mean TTFT (s)", args.out, f"ttft_{wl}", scale=1e-3)
        fig_cache(wl, rows, cs, args.out, f"cache_{wl}")

        arms_present = [a for a in ARMS if any((c, a) in rows for c in cs)]
        print(f"{wl}: c={cs}, arms={arms_present} -> {args.out}")
        print_speedups(wl, rows, cs)


if __name__ == "__main__":
    main()