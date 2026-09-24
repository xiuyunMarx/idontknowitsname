import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict
import csv

HERE = Path(__file__).resolve().parent

# (workload directory, CSV prefix, panel title, concurrency levels left out of the paper figures)
WORKLOADS = [
    ("fact_check", "fact", "Fact check", set()),
    ("coding", "coding", "Coding", set()),
    ("BFCL", "bfcl", "BFCL", {32}),
]
# Rows top to bottom: external baselines, each followed by its re-laid variant,
# then the ablated variants, then the full system.
ARMS = [
    "vanilla",
    "continuum", "continuum_relayout",
    "cachescout", "cachescout_relayout",
    "kvflow", "kvflow_relayout",
    "kvonly", "relayout", "ours",
]
# Same names as the legend of the speedup figure (draw_legend.py).
DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "continuum": "Continuum",
    "cachescout": "CacheScout",
    "continuum_relayout": "Continuum+Relayout",
    "cachescout_relayout": "CacheScout+Relayout",
    "kvflow": "KVFlow",
    "kvflow_relayout": "KVFlow+Relayout",
    "kvonly": "KVOnly",
    "relayout": "Relayout",
    "ours": "Ours",
}
GROUP_GAP = 0.4  # blank columns between two workloads


def read_hit_rate(file_path: Path, skip: set) -> Dict[int, float]:
    """Concurrency -> cache hit rate (%) of prompt tokens, device and host together."""
    hit_rate: Dict[int, float] = {}
    if not file_path.exists():  # arm not run yet
        return hit_rate
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            c = int(row["c"])
            if c not in skip:
                hit_rate[c] = 100.0 - float(row["miss_pct"])
    return hit_rate


def draw_hit_rate(output_file: Path) -> Path:
    """One heatmap for all workloads: rows are systems, columns are concurrency levels."""
    records = {
        title: {arm: read_hit_rate(HERE / d / f"{prefix}_sweep_{arm}.csv", skip) for arm in ARMS}
        for d, prefix, title, skip in WORKLOADS
    }

    # Column positions, with a gap between workloads.
    columns = []  # (x, workload title, concurrency)
    x = 0.0
    for title, data in records.items():
        for c in sorted(data["ours"]):
            columns.append((x, title, c))
            x += 1
        x += GROUP_GAP

    vmax = max(v for data in records.values() for arm in ARMS for v in data[arm].values())
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.YlGnBu

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    for x, title, c in columns:
        for row, arm in enumerate(ARMS):
            value = records[title][arm].get(c)
            if value is None:  # not measured yet
                ax.add_patch(Rectangle((x - 0.5, row - 0.5), 1, 1, color="#F2F2F2", linewidth=0))
                ax.text(x, row, "\u2013", ha="center", va="center", fontsize=6, color="#9A9A9A")
                continue
            ax.add_patch(Rectangle((x - 0.5, row - 0.5), 1, 1, color=cmap(norm(value)), linewidth=0))
            ax.text(x, row, f"{value:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if norm(value) > 0.55 else "black")

    ax.set_xlim(columns[0][0] - 0.5, columns[-1][0] + 0.5)
    ax.set_ylim(len(ARMS) - 0.5, -0.5)
    ax.set_yticks(range(len(ARMS)), [DISPLAY_NAMES[arm] for arm in ARMS])
    ax.set_xticks([x for x, _, _ in columns], [str(c) for _, _, c in columns])
    for title in records:
        xs = [x for x, t, _ in columns if t == title]
        ax.text((xs[0] + xs[-1]) / 2, -0.75, title, ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Concurrency")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_file


if __name__ == "__main__":
    print(f"Saved {draw_hit_rate(HERE / 'hit_rate.pdf')}")
