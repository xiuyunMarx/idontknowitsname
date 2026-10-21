import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from pathlib import Path
import csv
import re

# One combined figure for the paper: throughput (top) and JCT (bottom) speedup over
# vanilla, one column per workload, legend on top. Styles match */draw.py.
WORKLOADS = [
    ("fact_check", "Fact check"),
    ("coding", "Coding"),
    ("BFCL", "BFCL"),
    ("finance", "Finance"),
]
ROWS = [
    ("tp_speedup", "Throughput speedup"),
    ("jct_speedup", "JCT speedup"),
]

DISPLAY_NAMES = {
    "vanilla": "Vanilla SGLang",
    "adapter": "Static-Adapter",
    "kvonly": r"\textsc{Mortis}-No-Relayout",
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "continuum": "Continuum",
    "cachescout": "CacheScout",
    "continuum_relayout": "Continuum+Relayout",
    "cachescout_relayout": "CacheScout+Relayout",
    "kvflow": "KVFlow",
    "kvflow_relayout": "KVFlow+Relayout",
    "ours": r"\textsc{Mortis}",
}
COLORS = {
    "vanilla": "#4D4D4D",
    "adapter": "#A0522D",
    "kvonly": "#0072B2",
    "relayout": "#009E73",
    "continuum": "#E69F00",
    "cachescout": "#CC79A7",
    "continuum_relayout": "#E69F00",
    "cachescout_relayout": "#CC79A7",
    "kvflow": "#56B4E9",
    "kvflow_relayout": "#56B4E9",
    "ours": "#D55E00",
}
MARKERS = {
    "vanilla": "o",
    "adapter": "h",
    "kvonly": "s",
    "relayout": "^",
    "continuum": "D",
    "cachescout": "v",
    "continuum_relayout": "D",
    "cachescout_relayout": "v",
    "kvflow": "P",
    "kvflow_relayout": "P",
    "ours": "*",
}
# Legend fills column by column in two rows; see draw_legend.py. None leaves a blank slot.
LEGEND_ORDER = [
    "vanilla", "adapter",
    "relayout", "kvonly",
    "continuum", "continuum_relayout",
    "cachescout", "cachescout_relayout",
    "kvflow", "kvflow_relayout",
    "ours", None,
]
# Draw order: Mortis last so it sits on top.
DRAW_ORDER = [
    "vanilla", "adapter", "kvonly", "relayout", "continuum", "continuum_relayout",
    "cachescout", "cachescout_relayout", "kvflow", "kvflow_relayout", "ours",
]


def read_workload(directory: Path) -> dict:
    """Return {arm: {c: {tp_speedup, jct_speedup}}} for one workload directory."""
    pattern = re.compile(r"[a-z]+_sweep_([a-z_]+)\.csv")
    raw = {}
    for path in directory.iterdir():
        if match := pattern.fullmatch(path.name):
            with open(path, newline="", encoding="utf-8") as f:
                raw[match.group(1)] = {
                    int(row["c"]): (float(row["jct_mean_s"]), float(row["prompt_ktokens_per_min"]))
                    for row in csv.DictReader(f)
                }
    base = raw["vanilla"]
    return {
        arm: {
            c: {"jct_speedup": base[c][0] / jct, "tp_speedup": tp / base[c][1]}
            for c, (jct, tp) in data.items()
        }
        for arm, data in raw.items()
    }


def line_style(arm: str) -> dict:
    is_ours = arm == "ours"
    is_variant = arm.endswith("_relayout")
    return dict(
        color=COLORS[arm],
        marker=MARKERS[arm],
        markersize=4.2 if is_ours else 2.6,
        markeredgewidth=0.5,
        markerfacecolor="white" if is_variant else COLORS[arm],
        linewidth=1.1 if is_ours else 0.7,
        linestyle=(0, (3, 1.5)) if is_variant or arm == "vanilla" else "-",
    )


def draw(output_file: str) -> Path:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        # Printed at 1:1 (\textwidth = 6.75in), so these are the on-page sizes.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1.5,
        "axes.titlepad": 2.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    root = Path(__file__).resolve().parent
    records = {key: read_workload(root / key) for key, _ in WORKLOADS}

    fig, axes = plt.subplots(2, 4, figsize=(6.75, 2.45), sharex="col")
    fig.subplots_adjust(left=0.062, right=0.997, top=0.815, bottom=0.1, wspace=0.28, hspace=0.1)

    for col, (key, title) in enumerate(WORKLOADS):
        data = records[key]
        concurrencies = sorted({c for arm in data.values() for c in arm})
        for row, (metric, ylabel) in enumerate(ROWS):
            ax = axes[row, col]
            ax.axhline(1.0, color="#9A9A9A", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
            for arm in [a for a in DRAW_ORDER if a in data]:
                points = sorted(data[arm].items())
                ax.plot(
                    [c for c, _ in points],
                    [m[metric] for _, m in points],
                    zorder=4 if arm == "ours" else (3 if arm.endswith("_relayout") else 2),
                    **line_style(arm),
                )
            ax.set_xticks(concurrencies)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylabel)

    fig.supxlabel("Concurrency", y=0.0, fontsize=8)

    handles = []
    for arm in LEGEND_ORDER:
        if arm is None:
            handles.append(plt.Line2D([], [], linestyle="none", marker="none", label=" "))
        else:
            handles.append(plt.Line2D([], [], label=DISPLAY_NAMES[arm], **line_style(arm)))
    fig.legend(
        handles=handles,
        ncol=len(handles) // 2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    output_path = root / output_file
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved {draw('speedup.pdf')}")
