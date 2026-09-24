import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Dict
import re
import csv
from collections.abc import Iterator


def read_csv(file_path) -> dict:
    metrics: Dict[int, Dict[str, float]] = {}
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            c = int(row['c'])
            metrics[c] = {
                "mean_jct": float(row['jct_mean_s']),
                "mean_TTFT": float(row["ttft_mean_ms"])
            }
    return metrics
            

def iterate_all(root: str) -> Iterator[tuple[str, dict]]:
    pattern = re.compile(r"[a-z]+_sweep_([a-z_]+)\.csv")

    for path in Path(root).iterdir():
        if path.is_file() and (match := pattern.fullmatch(path.name)):
            arm = match.group(1)
            yield arm, read_csv(path)

def draw_line_chart(records: dict):
    """Draw JCT and TTFT speedup curves and save them as vector PDFs."""
    if not records:
        raise ValueError("records is empty")

    # Fixed styles keep the two figures visually consistent.
    display_names = {
        "vanilla": "Vanilla",
        "kvonly": "KVOnly",
        "relayout": "Relayout",
        "continuum": "Continuum",
        "cachescout": "CacheScout",
        "continuum_relayout": "Continuum+Relayout",
        "cachescout_relayout": "CacheScout+Relayout",
        "kvflow": "KVFlow",
        "kvflow_relayout": "KVFlow+Relayout",
        "ours": "Ours",
    }
    # A *_relayout arm reuses its base arm's color and marker; it is drawn
    # dashed with a hollow marker.
    colors = {
        "vanilla": "#4D4D4D",
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
    markers = {
        "vanilla": "o",
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
    preferred_order = [
        "vanilla", "kvonly", "relayout", "continuum", "continuum_relayout",
        "cachescout", "cachescout_relayout", "kvflow", "kvflow_relayout", "ours",
    ]
    arm_order = [arm for arm in preferred_order if arm in records]
    arm_order.extend(sorted(set(records) - set(arm_order)))
    all_concurrencies = sorted(
        {concurrency for data in records.values() for concurrency in data}
    )

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

    def plot_metric(metric: str, output_file: str) -> Path:
        fig, ax = plt.subplots(figsize=(2.25, 1.4), constrained_layout=True)

        # A subtle reference line makes values above/below the baseline obvious.
        ax.axhline(1.0, color="#9A9A9A", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)

        for arm in arm_order:
            points = sorted(records[arm].items())
            x = [concurrency for concurrency, _ in points]
            try:
                y = [values[metric] for _, values in points]
            except KeyError as exc:
                raise KeyError(f"Missing {metric!r} for arm {arm!r}") from exc

            is_ours = arm == "ours"
            is_variant = arm.endswith("_relayout")
            color = colors.get(arm)
            ax.plot(
                x,
                y,
                label=display_names.get(arm, arm.replace("_", " ").title()),
                color=color,
                marker=markers.get(arm, "o"),
                markersize=4.0 if is_ours else 2.6,
                markeredgewidth=0.5,
                markerfacecolor="white" if is_variant else color,
                linewidth=1.0 if is_ours else 0.7,
                linestyle=(0, (3, 1.5)) if is_variant or arm == "vanilla" else "-",
                zorder=4 if is_ours else (3 if is_variant else 2),
            )

        ax.set_xlabel("Concurrency")
        ax.set_xticks(all_concurrencies)
        # Flat panels get too few automatic y ticks; fix the count.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
        ax.set_axisbelow(True)
        # The legend is shared across all panels; see ../draw_legend.py.

        output_path = Path(output_file)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return output_path

    jct_path = plot_metric("jct_speedup", "jct_speedup.pdf")
    ttft_path = plot_metric("ttft_speedup", "ttft_speedup.pdf")
    return jct_path, ttft_path


if __name__ == "__main__":
    full_records: dict = {}
    for arm, data in iterate_all("."):
        full_records[arm] = data
        
    # calculate speedup
    baseline = "vanilla"
    for arm, data in full_records.items():
        for c, metrics in data.items():

            baseline_metrics = full_records[baseline][c]
            metrics["jct_speedup"] = baseline_metrics["mean_jct"] / metrics["mean_jct"]
            metrics["ttft_speedup"] = baseline_metrics['mean_TTFT']/metrics["mean_TTFT"]
    jct_pdf, ttft_pdf = draw_line_chart(full_records)
    print(f"Saved {jct_pdf} and {ttft_pdf}")
