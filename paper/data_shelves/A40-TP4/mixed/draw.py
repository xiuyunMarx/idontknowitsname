"""TP=4 lane sweep, same three panels as ../../A40/mix_sweep: throughput speedup and
geometric-mean JCT speedup over Vanilla SGLang, and prompt-token cache hit rate
(device + host, stacked).

    python draw.py     -> throughput_speedup.pdf jct_speedup.pdf hit_rate.pdf legend.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator, ScalarFormatter
from pathlib import Path
import csv
import math

FILE = "lane_sweep.csv"
# Per-agent rows of one lane level; JCT must be normalized per agent (see read_jct).
PROGRAMS_FILE = "mix_L{lanes}_programs.csv"
# Drawn bottom to top in this order.
ARMS = ["vanilla", "kvonly", "relayout", "ours_noprefetch", "ours"]

# Styles match ../../A40/mix_sweep/draw.py so the two figures read alike:
# a variant (ours_noprefetch) reuses its base arm's color and marker, dashed and hollow.
display_names = {
    "vanilla": "SGLang",
    "kvonly": r"\textsc{Mortis}-No-Relayout",
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "ours_noprefetch": r"\textsc{Mortis}-No-Prefetch",
    "ours": r"\textsc{Mortis}",
}
colors = {"vanilla": "#4D4D4D", "kvonly": "#0072B2", "relayout": "#009E73", "ours_noprefetch": "#D55E00", "ours": "#D55E00"}
markers = {"vanilla": "o", "kvonly": "s", "relayout": "^", "ours_noprefetch": "*", "ours": "*"}
variants = {"ours_noprefetch"}
HOST_TINT = 0.55  # host-hit segment: the arm color mixed with this much white


def tint(color: str, amount: float):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def read_csv(path) -> dict:
    """{arm: {lanes: {throughput, device_hit, host_hit, hit_rate}}} for the arms drawn."""
    metrics = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["arm"] not in ARMS or row["program"] != "all":
                continue
            device, host = float(row["device_pct"]), float(row["host_pct"])
            metrics.setdefault(row["arm"], {})[int(row["lanes"])] = {
                "throughput": float(row["prompt_ktokens_per_min"]),
                "device_hit": device, "host_hit": host, "hit_rate": device + host,
            }
    return metrics


def read_jct(metrics: dict) -> None:
    """Attach each arm's mean JCT per agent, {program: seconds}, at every lane level."""
    levels = sorted({level for data in metrics.values() for level in data})
    for lanes in levels:
        with open(PROGRAMS_FILE.format(lanes=lanes), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["arm"] in metrics and row["program"] != "all":
                    metrics[row["arm"]][lanes].setdefault("jct", {})[row["program"]] = float(row["jct_done_mean_s"])


def jct_speedup(jct: dict, baseline_jct: dict) -> float:
    """Geometric mean over agents of the baseline-to-arm mean JCT ratio, so every agent
    weighs the same despite several-fold differences in JCT."""
    ratios = [baseline_jct[program] / seconds for program, seconds in jct.items()]
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


class TimesFormatter(ScalarFormatter):
    """Speedup tick labels: the default number followed by a times sign."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos) + r"$\times$"


def line_style(arm: str) -> dict:
    is_ours = arm == "ours"
    return dict(
        color=colors[arm], marker=markers[arm],
        markersize=4.0 if is_ours else 2.6, markeredgewidth=0.5,
        markerfacecolor="white" if arm in variants else colors[arm],
        linewidth=1.0 if is_ours else 0.7,
        linestyle=(0, (3, 1.5)) if arm in variants or arm == "vanilla" else "-",
    )


def draw(records: dict):
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
    # The two speedup panels share one row of the 3.25in column; the hit-rate bars take a full row.
    column_w, panel_h = 3.25, 0.98
    margin_l, margin_r, margin_b, margin_t = 0.29, 0.03, 0.27, 0.06
    levels = sorted({lv for data in records.values() for lv in data})

    def new_axes(panel_w, h=panel_h, left=margin_l):
        fig = plt.figure(figsize=(panel_w, h))
        ax = fig.add_axes((left / panel_w, margin_b / h,
                           1 - (left + margin_r) / panel_w, 1 - (margin_b + margin_t) / h))
        ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
        ax.set_axisbelow(True)
        ax.set_xlabel("Lanes per agent")
        return fig, ax

    def save(fig, output_file):
        fig.savefig(output_file, format="pdf")
        plt.close(fig)
        return Path(output_file)

    def plot_speedup(metric, title, output_file):
        # The y label names the metric in place of a subcaption, on two lines so the panel stays short.
        fig, ax = new_axes(column_w / 2, h=0.90, left=0.50)
        ax.set_ylabel(title.replace(" ", "\n"), linespacing=1.0)
        ax.axhline(1.0, color="#9A9A9A", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
        for arm in ARMS:
            points = sorted(records[arm].items())
            ax.plot([lv for lv, _ in points], [v[metric] for _, v in points],
                    zorder=4 if arm == "ours" else 2, **line_style(arm))
        ax.set_xticks(levels)
        ax.set_xlim(levels[0] - 0.4, levels[-1] + 0.4)
        top = max(v[metric] for data in records.values() for v in data.values())
        ax.set_ylim(0.9, math.ceil(top * 10) / 10 + 0.05)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(TimesFormatter())
        return save(fig, output_file)

    def plot_hit_rate(output_file):
        """One group per lane level, one bar per arm stacking device hits under host hits.
        Axes, colors and % ticks follow the line panels; the variant is hollow and hatched,
        mirroring its hollow marker, and host hits are a light tint of the arm color."""
        fig, ax = new_axes(column_w)
        width = 0.8 / len(ARMS)
        for i, arm in enumerate(ARMS):
            x = [k + (i - (len(ARMS) - 1) / 2) * width for k in range(len(levels))]
            device = [records[arm][lv]["device_hit"] for lv in levels]
            host = [records[arm][lv]["host_hit"] for lv in levels]
            hollow = arm in variants
            style = dict(width=width, edgecolor=colors[arm], linewidth=0.5,
                         hatch="////" if hollow else None, zorder=2)
            ax.bar(x, device, color="white" if hollow else colors[arm], **style)
            ax.bar(x, host, bottom=device, color=tint(colors[arm], HOST_TINT), **style)
        ax.set_xticks(range(len(levels)), [str(lv) for lv in levels])
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_ylim(0, 62)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"{v:.0f}\%"))

        # Segment key in the empty upper right, in Vanilla's neutral gray.
        gray = colors["vanilla"]
        ax.legend(
            handles=[Patch(facecolor=gray, edgecolor=gray, linewidth=0.5, label="Device hit"),
                     Patch(facecolor=tint(gray, HOST_TINT), edgecolor=gray, linewidth=0.5, label="Host hit")],
            loc="upper right", ncol=2, frameon=False, fontsize=6.5,
            handlelength=1.0, handleheight=0.8, handletextpad=0.35, columnspacing=0.9,
            borderpad=0.2, borderaxespad=0.3)
        return save(fig, output_file)

    plt.rcParams["hatch.linewidth"] = 0.4
    paths = [
        plot_speedup("tp_speedup", "Throughput speedup", "throughput_speedup.pdf"),
        plot_speedup("jct_speedup", "JCT speedup", "jct_speedup.pdf"),
        plot_hit_rate("hit_rate.pdf"),
    ]

    # Shared legend strip placed above the panels in LaTeX; five names do not fit one 3.25in row, so two.
    handles = [Line2D([], [], label=display_names[a], **line_style(a)) for a in ARMS]
    fig = plt.figure(figsize=(3.25, 0.22))
    fig.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=6.5,
               handlelength=1.2, columnspacing=0.5, handletextpad=0.25, labelspacing=0.25,
               borderpad=0, borderaxespad=0)
    fig.savefig("legend.pdf", format="pdf")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    records = read_csv(FILE)
    read_jct(records)
    baseline = "vanilla"
    for arm, data in records.items():
        for level, m in data.items():
            b = records[baseline][level]
            m["tp_speedup"] = m["throughput"] / b["throughput"]
            m["jct_speedup"] = jct_speedup(m["jct"], b["jct"])
            print(f"L{level} {arm:16s} tp {m['tp_speedup']:.2f}x  jct {m['jct_speedup']:.2f}x  hit {m['hit_rate']:.1f}%")
    print("Saved", *draw(records), "and legend.pdf")
