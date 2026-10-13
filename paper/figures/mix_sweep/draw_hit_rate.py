import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Dict
import csv

HERE = Path(__file__).resolve().parent
FILE = HERE / "lane_sweep.csv"

# Same arms, order and colors as the speedup panels of this sweep (draw.py).
ARMS = ["vanilla", "adapter", "relayout", "ours"]
COLORS = {
    "vanilla": "#4D4D4D",
    "adapter": "#A0522D",
    "relayout": "#009E73",
    "ours": "#D55E00",
}
DISPLAY_NAMES = {
    "vanilla": "Vanilla SGLang",
    "adapter": "Static-Adapter",
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "ours": r"\textsc{Mortis}",
}
HOST_TINT = 0.55  # host segment: the arm color mixed with this much white
BAR_WIDTH = 0.2


def tint(color: str, amount: float):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def read_breakdown(file_path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Arm -> lanes per agent -> share (%) of prompt tokens hit in device and in host memory."""
    breakdown: Dict[str, Dict[int, Dict[str, float]]] = {}
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            if row["arm"] not in ARMS:
                continue
            device, host = float(row["device_pct"]), float(row["host_pct"])
            breakdown.setdefault(row["arm"], {})[int(row["lanes"])] = {"device": device, "host": host}
    return breakdown


def draw_hit_rate(output_file: Path) -> Path:
    """Grouped stacked bars: one group per lane count, one bar per arm (device, then host).

    The miss is the rest of 100 % and is not drawn, so the axis can stop just
    above the highest hit rate instead of spending most of its height on misses.
    """
    records = read_breakdown(FILE)
    levels = sorted(records["ours"])

    # Same styling as the speedup panels (draw.py).
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.5,
        "axes.labelpad": 2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "ytick.direction": "in",
        "ytick.right": True,
        "ytick.major.size": 2.5,
        "ytick.major.width": 0.5,
        "ytick.major.pad": 2.5,
        "xtick.major.size": 0,
        "xtick.major.pad": 2.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.3, 1.3), constrained_layout=True)
    for i, arm in enumerate(ARMS):
        offset = (i - (len(ARMS) - 1) / 2) * BAR_WIDTH
        xs = [level + offset for level in levels]
        device = [records[arm][level]["device"] for level in levels]
        host = [records[arm][level]["host"] for level in levels]
        style = dict(width=BAR_WIDTH, edgecolor="white", linewidth=0.3)
        ax.bar(xs, device, color=COLORS[arm], **style)
        ax.bar(xs, host, bottom=device, color=tint(COLORS[arm], HOST_TINT), **style)

    ax.set_xlim(levels[0] - 0.5, levels[-1] + 0.5)
    ax.set_xticks(levels)
    ax.set_xlabel("Lanes per agent")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylabel(r"Hit rate (\%)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
    ax.set_axisbelow(True)

    # Segment legend in the empty upper right, drawn in a neutral gray. It sits
    # inside the axes, so it can be an extra artist that layout ignores.
    gray = "#808080"
    segment_legend = ax.legend(
        handles=[
            Patch(facecolor=gray, label="Device hit (bottom)"),
            Patch(facecolor=tint(gray, HOST_TINT), label="Host hit (top)"),
        ],
        # White background so the grid lines do not run through the labels.
        loc="upper right", ncol=1, frameon=True, facecolor="white", edgecolor="none", framealpha=1,
        handlelength=1.0, handleheight=0.8, handletextpad=0.35, labelspacing=0.3,
        borderpad=0.2, borderaxespad=0.3)
    ax.add_artist(segment_legend)

    # Arm legend above the axes: one color per bar position. It must be the
    # axes' own legend so that constrained layout makes room for it.
    ax.legend(
        handles=[Patch(facecolor=COLORS[arm], label=DISPLAY_NAMES[arm]) for arm in ARMS],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(ARMS), frameon=False,
        handlelength=1.0, handleheight=0.8, columnspacing=0.9, handletextpad=0.35,
        borderpad=0, borderaxespad=0.3)

    fig.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_file


if __name__ == "__main__":
    print(f"Saved {draw_hit_rate(HERE / 'hit_rate.pdf')}")
