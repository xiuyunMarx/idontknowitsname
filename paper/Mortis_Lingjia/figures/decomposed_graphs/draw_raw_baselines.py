r"""Per-agent raw_baselines figure (fig:e2e_speedup): throughput (top) and JCT (bottom) speedup over SGLang
for the four agents under varying concurrency, legend on top, printed at \textwidth."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

SHELF = Path("/home/xiaoyu/idontknowitsname/paper/data_shelves/A40/sweep_graph")
OUT = Path("/home/xiaoyu/idontknowitsname/paper/Mortis_Lingjia/figures/decomposed_graphs/per-agent/raw_baselines")
AGENTS = ["fact_check", "coding", "BFCL", "finance"]
TITLES = {"fact_check": "Fact-check", "coding": "Coding", "BFCL": "BFCL", "finance": "Finance"}
ROWS = [("tp_speedup", "Throughput speedup"), ("jct_speedup", "JCT speedup")]
BASE_ARM = "vanilla"
ARMS = ["adapter", "continuum", "cachescout", "kvflow", "kvonly", "relayout", "ours"]

DISPLAY_NAMES = {
    "vanilla": "SGLang",
    "adapter": "Static-Adapter",
    "kvonly": r"\textsc{Mortis}-No-Relayout",
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "continuum": "Continuum",
    "cachescout": "CacheScout",
    "kvflow": "KVFlow",
    "ours": r"\textsc{Mortis}",
}
COLORS = {
    "adapter": "#A0522D",
    "relayout": "#009E73",
    "kvonly": "#0072B2",
    "continuum": "#E69F00",
    "cachescout": "#CC79A7",
    "kvflow": "#56B4E9",
    "ours": "#D55E00",
}
MARKERS = {"adapter": "h", "relayout": "^", "kvonly": "s", "continuum": "D", "cachescout": "v", "kvflow": "P", "ours": "*"}
BASELINE_STYLE: dict = dict(color="#7F7F7F", linewidth=0.8, linestyle=(0, (4, 2)))
COMBINED_RC = {
    # Printed at 1:1 (\textwidth = 6.75in), so these are the on-page sizes.
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.titlepad": 2.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "legend.fontsize": 7.5,
}


def line_style(arm: str) -> dict:
    is_ours = arm == "ours"
    return dict(
        color=COLORS[arm],
        marker=MARKERS[arm],
        markersize=4.6 if is_ours else 3.0,
        markeredgewidth=0.5,
        linewidth=1.3 if is_ours else 0.8,
        linestyle="-",
    )


def speedup_axis(ax) -> None:
    # At least four ticks, so flat-but-tall panels (BFCL throughput) are still readable.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
    ax.set_axisbelow(True)


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


def main() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "axes.linewidth": 0.5,
        "axes.labelpad": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        **COMBINED_RC,
    })
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 2.15), sharex="col")
    fig.subplots_adjust(left=0.062, right=0.997, top=0.885, bottom=0.11, wspace=0.28, hspace=0.07)
    for col, agent in enumerate(AGENTS):
        data = read_agent(agent)
        for row, (metric, ylabel) in enumerate(ROWS):
            ax = axes[row, col]
            ax.axhline(1.0, zorder=1, **BASELINE_STYLE)
            for arm in ARMS:
                points = sorted(data[arm].items())
                ax.plot([c for c, _ in points], [m[metric] for _, m in points],
                        zorder=4 if arm == "ours" else 2, **line_style(arm))
            ax.set_xticks(sorted(data[BASE_ARM]))
            speedup_axis(ax)
            if row == 0:
                ax.set_title(TITLES[agent])
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.supxlabel("Concurrency", y=0.0, fontsize=8)
    # One row at the tick-label size; the spacing is tightened so it fits \textwidth.
    handles = [Line2D([], [], label=DISPLAY_NAMES[BASE_ARM] + r" (=1$\times$)", **BASELINE_STYLE)]
    handles += [Line2D([], [], label=DISPLAY_NAMES[a], **line_style(a)) for a in ARMS]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="upper center",
               bbox_to_anchor=(0.53, 1.0), fontsize=7, columnspacing=0.6, handlelength=1.3,
               handletextpad=0.3, borderaxespad=0.0)
    path = OUT / "speedup.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
