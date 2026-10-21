import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, ScalarFormatter
from pathlib import Path
from typing import Dict
import csv

FILE = "lane_sweep.csv"
# Drawn bottom to top in this order; every arm is measured at every lane level.
ARMS = [
    "vanilla", "adapter", "kvonly", "relayout", "continuum", "continuum_relayout",
    "cachescout", "cachescout_relayout", "kvflow", "kvflow_relayout", "ours_noprefetch", "ours",
]


def read_csv(file_path) -> dict:
    metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            arm = row["arm"]
            if arm not in ARMS:
                continue
            lanes = int(row["lanes"])
            metrics.setdefault(arm, {})[lanes] = {
                "throughput": float(row["prompt_ktokens_per_min"]),
                "hit_rate": float(row["device_pct"]) + float(row["host_pct"]),
            }
    return metrics


class TimesFormatter(ScalarFormatter):
    """Speedup tick labels: the default number followed by a times sign."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos) + r"$\times$"


def draw_line_chart(records: dict):
    """Draw throughput speedup and hit rate curves, in identical styles, as vector PDFs."""
    if not records:
        raise ValueError("records is empty")

    # Styles must match the ones in ../sweep_graph/*/draw.py.
    display_names = {
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
        "ours_noprefetch": r"\textsc{Mortis}-No-Prefetch",
        "ours": r"\textsc{Mortis}",
    }
    # A variant (*_relayout, ours_noprefetch) reuses its base arm's color and
    # marker; it is drawn dashed with a hollow marker.
    colors = {
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
        "ours_noprefetch": "#D55E00",
        "ours": "#D55E00",
    }
    markers = {
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
        "ours_noprefetch": "*",
        "ours": "*",
    }
    variants = {arm for arm in ARMS if arm.endswith("_relayout") or arm == "ours_noprefetch"}
    arm_order = [arm for arm in ARMS if arm in records]
    all_levels = sorted({level for data in records.values() for level in data})

    # pgfplots-like styling: Times to match the paper body, boxed axes, inward
    # ticks on all four sides, thin strokes. Fonts are embedded in the PDFs.
    plt.rcParams.update({
        # Text is typeset by LaTeX with the paper's Times font so that labels
        # (including \textsc{Mortis}) match the body exactly.
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.5,
        "axes.labelpad": 2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    def plot_metric(metric: str, ylabel: str, output_file: str) -> Path:
        # Sized for half a column so fonts print at their true size.
        fig, ax = plt.subplots(figsize=(1.6, 1.2), constrained_layout=True)

        is_speedup = metric.endswith("_speedup")
        if is_speedup:
            # A subtle reference line makes values above/below the baseline obvious.
            ax.axhline(1.0, color="#9A9A9A", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)

        for arm in arm_order:
            points = sorted(records[arm].items())
            x = [level for level, _ in points]
            y = [values[metric] for _, values in points]

            is_ours = arm == "ours"
            color = colors.get(arm)
            if is_speedup:
                style = dict(
                    markersize=4.0 if is_ours else 2.6,
                    markeredgewidth=0.5,
                    markerfacecolor="white" if arm in variants else color,
                    linewidth=1.0 if is_ours else 0.7,
                    linestyle=(0, (3, 1.5)) if arm in variants or arm == "vanilla" else "-",
                )
            else:
                # Same color, marker and dash per arm, so the shared legend applies.
                hollow = arm in variants
                style = dict(
                    markersize=4.2 if is_ours else 3.0,
                    markeredgewidth=0.5,
                    markerfacecolor="white" if hollow else color,
                    linewidth=1.0 if is_ours else 0.7,
                    linestyle=(0, (3, 1.5)) if hollow or arm == "vanilla" else "-",
                )
            ax.plot(x, y, label=display_names.get(arm, arm), color=color,
                    marker=markers.get(arm, "o"), zorder=4 if is_ours else 2, **style)

        ax.set_xlabel("Lanes per agent")
        ax.set_ylabel(ylabel)
        # Every integer gets a tick, so an unmeasured level (7) shows as a tick without markers.
        ax.set_xticks(range(all_levels[0], all_levels[-1] + 1))
        if is_speedup:
            # Flat panels get too few automatic y ticks; fix the count.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
            ax.yaxis.set_major_formatter(TimesFormatter())
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"{v:.0f}\%"))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
        ax.set_axisbelow(True)

        output_path = Path(output_file)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return output_path

    tp_path = plot_metric("tp_speedup", "Throughput speedup", "throughput_speedup.pdf")
    # Total (device + host) hit rate.
    hit_path = plot_metric("hit_rate", "Cache hit rate", "hit_rate.pdf")

    # Shared legend strip placed above both panels in LaTeX.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=colors[a], marker=markers[a],
               markersize=4.0 if a == "ours" else 2.6, markeredgewidth=0.5,
               markerfacecolor="white" if a in variants else colors[a],
               linewidth=1.0 if a == "ours" else 0.7,
               linestyle=(0, (3, 1.5)) if a in variants or a == "vanilla" else "-",
               label=display_names[a])
        for a in arm_order
    ]
    fig = plt.figure(figsize=(3.4, 0.5))
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False,
               handlelength=1.7, columnspacing=1.0, handletextpad=0.4,
               borderpad=0, borderaxespad=0)
    legend_path = Path("legend.pdf")
    fig.savefig(legend_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return tp_path, hit_path


if __name__ == "__main__":
    records = read_csv(FILE)

    # calculate speedup over vanilla at the same lane level
    baseline = "vanilla"
    for arm, data in records.items():
        for level, metrics in data.items():
            baseline_metrics = records[baseline][level]
            metrics["tp_speedup"] = metrics["throughput"] / baseline_metrics["throughput"]

    tp_pdf, hit_pdf = draw_line_chart(records)
    print(f"Saved {tp_pdf} and {hit_pdf}")
