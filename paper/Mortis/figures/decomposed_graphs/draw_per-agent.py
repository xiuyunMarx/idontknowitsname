r"""Single-column relayout+baselines figure (fig:lane_sweep_baseline_relayout), drawn as three PDFs stacked in LaTeX:
legend_shared.pdf, speedup.pdf (throughput and JCT speedup over Mortis-Relayout-only per agent) and mixed.pdf
(throughput, JCT and hit rate on the mixed workload). Baselines are served with Mortis's re-laid prompts."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

import draw_mixed

SHELF = Path("/home/xiaoyu/idontknowitsname/paper/data_shelves/A40/sweep_graph")
OUT = Path("/home/xiaoyu/idontknowitsname/paper/Mortis_Lingjia/figures/decomposed_graphs/per-agent/relayout+baselines")
AGENTS = ["fact_check", "coding", "BFCL", "finance"]
TITLES = {"fact_check": "Fact-check", "coding": "Coding", "BFCL": "BFCL", "finance": "Finance"}
# The unit sits in the row label so tick labels stay narrow.
ROWS = [("tp_speedup", r"Throughput ($\times$)"), ("jct_speedup", r"JCT ($\times$)")]
# The mixed panels are wide enough to carry the unit on each speedup tick label.
MIXED_PANELS = [("tp_speedup", "Throughput"), ("jct_speedup", "JCT"), ("hit_rate", r"Hit rate (\%)")]
BASE_ARM = "relayout"
ARMS = ["continuum_relayout", "cachescout_relayout", "kvflow_relayout", "ours"]
# No-Prefetch was only run on the mixed workload.
MIXED_ARMS = ["continuum_relayout", "cachescout_relayout", "kvflow_relayout", "ours_noprefetch", "ours"]

DISPLAY_NAMES = {
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "continuum_relayout": "Continuum+Relayout",
    "cachescout_relayout": "CacheScout+Relayout",
    "kvflow_relayout": "KVFlow+Relayout",
    "ours_noprefetch": r"\textsc{Mortis}-No-Prefetch",
    "ours": r"\textsc{Mortis}",
}
# A *_relayout arm shares its raw arm's color and marker, so a system reads the same as in the raw-baselines figure.
COLORS = {"continuum": "#E69F00", "cachescout": "#CC79A7", "kvflow": "#56B4E9", "ours": "#D55E00"}
MARKERS = {"continuum": "D", "cachescout": "v", "kvflow": "P", "ours": "*"}
BASELINE_STYLE: dict = dict(color="#7F7F7F", linewidth=0.8, linestyle=(0, (4, 2)))


def line_style(arm: str) -> dict:
    base = arm.removesuffix("_relayout").removesuffix("_noprefetch")
    style = dict(
        color=COLORS[base],
        marker=MARKERS[base],
        markersize=3.2 if arm == "ours" else 2.2,
        markeredgewidth=0.5,
        linewidth=1.0 if arm == "ours" else 0.6,
        linestyle="-",
    )
    if arm == "ours_noprefetch":
        # A variant of Mortis: same color and marker, dashed and hollow.
        style.update(markersize=2.6, markerfacecolor="white", linewidth=0.7, linestyle=(0, (3, 1.5)))
    return style


def read_agent(agent: str) -> dict:
    """{arm: {c: {tp_speedup, jct_speedup}}} relative to BASE_ARM."""
    pattern = re.compile(r"[a-z]+_sweep_([a-z_]+)\.csv")
    raw = {}
    for path in (SHELF / agent).iterdir():
        if match := pattern.fullmatch(path.name):
            with open(path, newline="", encoding="utf-8") as f:
                raw[match.group(1)] = {
                    int(row["c"]): (float(row["jct_mean_s"]), float(row["prompt_ktokens_per_min"]))
                    for row in csv.DictReader(f)
                }
    base = raw[BASE_ARM]
    return {
        arm: {c: {"jct_speedup": base[c][0] / jct, "tp_speedup": tp / base[c][1]} for c, (jct, tp) in data.items()}
        for arm, data in raw.items()
    }


def style_axis(ax) -> None:
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))


def plot_per_agent(path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(3.33, 1.42), sharey="col", sharex="col")
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.16, top=0.916, wspace=0.2, hspace=0.06)
    for col, agent in enumerate(AGENTS):
        data = read_agent(agent)
        levels = sorted(data[BASE_ARM])
        for row, (metric, ylabel) in enumerate(ROWS):
            ax = axes[row, col]
            ax.axhline(1.0, zorder=1, **BASELINE_STYLE)
            for arm in ARMS:
                points = sorted(data[arm].items())
                ax.plot(range(len(points)), [m[metric] for _, m in points],
                        zorder=4 if arm == "ours" else 2, **line_style(arm))
            # Levels evenly spaced so every concurrency label is readable at this width.
            ax.set_xticks(range(len(levels)), [str(c) for c in levels], fontsize=5.5)
            # Each agent's two panels share a range fitted to its data; steps keep one-decimal labels exact.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
            style_axis(ax)
            ax.margins(x=0.06, y=0.08)
            if row == 0:
                ax.set_title(TITLES[agent])
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=1)
    fig.supxlabel("Concurrency", y=0.0, fontsize=7)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


def plot_mixed(path: Path) -> None:
    records = draw_mixed.load(BASE_ARM)
    levels = sorted(records[BASE_ARM])
    height = 0.8
    fig, axes = plt.subplots(1, 3, figsize=(3.33, height))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.23, top=1 - 0.12 / height, wspace=0.32)
    for ax, (metric, title) in zip(axes, MIXED_PANELS):
        if metric == "hit_rate":
            # Hit rate is absolute, so the baseline is a real curve in the same gray dashed style.
            ax.plot(levels, [records[BASE_ARM][l][metric] for l in levels], zorder=1, **BASELINE_STYLE)
        else:
            ax.axhline(1.0, zorder=1, **BASELINE_STYLE)
        for arm in MIXED_ARMS:
            points = sorted(records[arm].items())
            ax.plot([l for l, _ in points], [m[metric] for _, m in points],
                    zorder=4 if arm == "ours" else 2, **line_style(arm))
        ax.set_xticks(levels, [str(l) for l in levels], fontsize=5.5)
        if metric == "hit_rate":
            ax.set_ylim(0, 62)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
            ax.margins(x=0.06)
        else:
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"{v:.1f}$\times$"))
            ax.margins(x=0.06, y=0.08)
        style_axis(ax)
        ax.set_title(title)
    fig.supxlabel("Lanes per agent", y=0.0, fontsize=7)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


def plot_legend(path: Path) -> None:
    # Six entries in a 3x2 grid read left to right; matplotlib fills columns first, so reorder.
    handles = [Line2D([], [], label=DISPLAY_NAMES[BASE_ARM] + r" (=1$\times$)", **BASELINE_STYLE)]
    handles += [Line2D([], [], label=DISPLAY_NAMES[a], **line_style(a)) for a in MIXED_ARMS]
    handles = [handles[r * 3 + c] for c in range(3) for r in range(2)]
    fig = plt.figure(figsize=(3.33, 0.19))
    fig.legend(handles=handles, ncol=3, loc="center", frameon=False, fontsize=6.3, columnspacing=0.5,
               labelspacing=0.12, handlelength=1.2, handletextpad=0.25, borderpad=0, borderaxespad=0)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.titlepad": 2,
        "axes.linewidth": 0.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plot_legend(OUT / "legend_shared.pdf")
    plot_per_agent(OUT / "speedup.pdf")
    plot_mixed(OUT / "mixed.pdf")


if __name__ == "__main__":
    main()
