import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    pattern = re.compile(r"[a-z]+_sweep_([a-z]+)\.csv")

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
        "ours": "Ours",
    }
    colors = {
        "vanilla": "#4D4D4D",
        "kvonly": "#0072B2",
        "relayout": "#009E73",
        "continuum": "#E69F00",
        "cachescout": "#CC79A7",
        "ours": "#D55E00",
    }
    markers = {
        "vanilla": "o",
        "kvonly": "s",
        "relayout": "^",
        "continuum": "D",
        "cachescout": "v",
        "ours": "*",
    }
    preferred_order = [
        "vanilla", "kvonly", "relayout", "continuum", "cachescout", "ours"
    ]
    arm_order = [arm for arm in preferred_order if arm in records]
    arm_order.extend(sorted(set(records) - set(arm_order)))
    all_concurrencies = sorted(
        {concurrency for data in records.values() for concurrency in data}
    )

    # Compact single-column figure styling; fonts are embedded in the PDFs.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    def plot_metric(metric: str, output_file: str) -> Path:
        fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)

        # A subtle reference line makes values above/below the baseline obvious.
        ax.axhline(1.0, color="#B0B0B0", linewidth=0.8, linestyle="--", zorder=0)

        for arm in arm_order:
            points = sorted(records[arm].items())
            x = [concurrency for concurrency, _ in points]
            try:
                y = [values[metric] for _, values in points]
            except KeyError as exc:
                raise KeyError(f"Missing {metric!r} for arm {arm!r}") from exc

            is_ours = arm == "ours"
            ax.plot(
                x,
                y,
                label=display_names.get(arm, arm.replace("_", " ").title()),
                color=colors.get(arm),
                marker=markers.get(arm, "o"),
                markersize=6.0 if is_ours else 4.2,
                markeredgewidth=0.7,
                linewidth=2.0 if is_ours else 1.35,
                linestyle="--" if arm == "vanilla" else "-",
                zorder=3 if is_ours else 2,
            )

        ax.set_xlabel("Concurrency")
        ax.set_xticks(all_concurrencies)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.7)
        ax.legend(
            ncol=2,
            frameon=False,
            loc="best",
            columnspacing=0.9,
            handlelength=2.0,
        )

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
        for c,metrics in data.items():
            baseline_metrics = full_records[baseline][c]
            metrics["jct_speedup"] = baseline_metrics["mean_jct"] / metrics["mean_jct"]
            metrics["ttft_speedup"] = baseline_metrics['mean_TTFT']/metrics["mean_TTFT"]

    jct_pdf, ttft_pdf = draw_line_chart(full_records)
    print(f"Saved {jct_pdf} and {ttft_pdf}")
