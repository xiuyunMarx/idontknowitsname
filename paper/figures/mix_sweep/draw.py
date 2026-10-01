import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from pathlib import Path
from typing import Dict
import csv

FILE = "lane_sweep.csv"
# Arms measured at every lane level; the rest exist only at L=4 (mix4x4_h10).
ARMS = ["vanilla", "relayout", "ours"]


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
                "mean_TTFT": float(row["ttft_mean_s"]),
            }
    return metrics


class TimesFormatter(ScalarFormatter):
    """Speedup tick labels: the default number followed by a times sign."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos) + r"$\times$"


def draw_line_chart(records: dict):
    """Draw throughput and TTFT speedup curves and save them as vector PDFs."""
    if not records:
        raise ValueError("records is empty")

    # Styles must match the ones in ../sweep_graph/*/draw.py.
    display_names = {
        "vanilla": "Vanilla SGLang",
        "relayout": "ANON-Relayout-only",
        "ours": "ANON",
    }
    colors = {
        "vanilla": "#4D4D4D",
        "relayout": "#009E73",
        "ours": "#D55E00",
    }
    markers = {
        "vanilla": "o",
        "relayout": "^",
        "ours": "*",
    }
    arm_order = [arm for arm in ARMS if arm in records]
    all_levels = sorted({level for data in records.values() for level in data})

    # pgfplots-like styling: Times to match the paper body, boxed axes, inward
    # ticks on all four sides, thin strokes. Fonts are embedded in the PDFs.
    plt.rcParams.update({
        "font.family": "serif",
        # STIX is a TrueType Times clone bundled with matplotlib (Type 42 safe).
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
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

    def plot_metric(metric: str, output_file: str, legend: bool = False) -> Path:
        # Sized for half a column so fonts print at their true size.
        fig, ax = plt.subplots(figsize=(1.6, 1.2), constrained_layout=True)

        # A subtle reference line makes values above/below the baseline obvious.
        ax.axhline(1.0, color="#9A9A9A", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)

        for arm in arm_order:
            points = sorted(records[arm].items())
            x = [level for level, _ in points]
            y = [values[metric] for _, values in points]

            is_ours = arm == "ours"
            color = colors.get(arm)
            ax.plot(
                x,
                y,
                label=display_names.get(arm, arm),
                color=color,
                marker=markers.get(arm, "o"),
                markersize=4.0 if is_ours else 2.6,
                markeredgewidth=0.5,
                markerfacecolor=color,
                linewidth=1.0 if is_ours else 0.7,
                linestyle=(0, (3, 1.5)) if arm == "vanilla" else "-",
                zorder=4 if is_ours else 2,
            )

        ax.set_xlabel("Lanes per agent")
        ax.set_xticks(all_levels)
        # Flat panels get too few automatic y ticks; fix the count.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
        ax.yaxis.set_major_formatter(TimesFormatter())
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
        ax.set_axisbelow(True)

        output_path = Path(output_file)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return output_path

    tp_path = plot_metric("tp_speedup", "throughput_speedup.pdf")
    ttft_path = plot_metric("ttft_speedup", "ttft_speedup.pdf")

    # Shared legend strip placed above the two panels in LaTeX.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=colors[a], marker=markers[a],
               markersize=4.0 if a == "ours" else 2.6, markeredgewidth=0.5,
               markerfacecolor=colors[a],
               linewidth=1.0 if a == "ours" else 0.7,
               linestyle=(0, (3, 1.5)) if a == "vanilla" else "-",
               label=display_names[a])
        for a in arm_order
    ]
    fig = plt.figure(figsize=(3.2, 0.16))
    fig.legend(handles=handles, loc="center", ncol=3, frameon=False,
               handlelength=1.7, columnspacing=1.0, handletextpad=0.4,
               borderpad=0, borderaxespad=0)
    legend_path = Path("legend.pdf")
    fig.savefig(legend_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return tp_path, ttft_path


if __name__ == "__main__":
    records = read_csv(FILE)

    # calculate speedup over vanilla at the same lane level
    baseline = "vanilla"
    for arm, data in records.items():
        for level, metrics in data.items():
            baseline_metrics = records[baseline][level]
            metrics["tp_speedup"] = metrics["throughput"] / baseline_metrics["throughput"]
            metrics["ttft_speedup"] = baseline_metrics["mean_TTFT"] / metrics["mean_TTFT"]

    tp_pdf, ttft_pdf = draw_line_chart(records)
    print(f"Saved {tp_pdf} and {ttft_pdf}")
