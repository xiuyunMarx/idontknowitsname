"""Sensitivity study: mixed-workload lane sweep on two extra testbeds, A40 TP=4 and
RTX 3090 (Qwen3-14B-AWQ). One row of four panels in LaTeX: subfigure (a) tp4.pdf holds the
throughput and JCT speedup over Vanilla SGLang on A40 TP=4, subfigure (b) 3090.pdf the same on
the 3090. The two panels of a testbed share the y axis; metric names are panel titles.

    python draw.py  -> legend.pdf tp4.pdf 3090.pdf

Styles follow ../TP=4 (data_shelves/A40-TP4/mixed/draw.py) so the figure reads like Fig. 7.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from pathlib import Path
import csv
import math

HERE = Path(__file__).resolve().parent
SHELVES = HERE.parents[2] / "data_shelves"

# testbed -> (lane_sweep.csv, per-agent JCT file pattern, inset label)
TESTBEDS = {
    "tp4": (SHELVES / "A40-TP4/mixed/lane_sweep.csv",
            str(SHELVES / "A40-TP4/mixed/mix_L{lanes}_programs.csv"), r"A40 TP=4"),
    "3090": (SHELVES / "3090/mix_sweep/lane_sweep.csv",
             str(SHELVES / "3090/mix_sweep/mix4x{lanes}_h10_steady_programs.csv"), r"RTX 3090"),
}
# No-Prefetch is not in the 3090 sweep, so both testbeds draw the same four arms.
ARMS = ["vanilla", "kvonly", "relayout", "ours"]
display_names = {
    "vanilla": "SGLang",
    "kvonly": r"\textsc{Mortis}-No-Relayout",
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "ours": r"\textsc{Mortis}",
}
colors = {"vanilla": "#4D4D4D", "kvonly": "#0072B2", "relayout": "#009E73", "ours": "#D55E00"}
markers = {"vanilla": "o", "kvonly": "s", "relayout": "^", "ours": "*"}


def read_testbed(sweep_file: Path, programs_file: str) -> dict:
    """{arm: {lanes: {tp_speedup, jct_speedup}}} over Vanilla at the same lane count."""
    tp = {}
    with open(sweep_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["arm"] not in ARMS or row.get("program", "all") != "all":
                continue
            tp.setdefault(row["arm"], {})[int(row["lanes"])] = float(row["prompt_ktokens_per_min"])
    jct = {}
    for lanes in sorted(tp["vanilla"]):
        with open(programs_file.format(lanes=lanes), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["arm"] in ARMS and row["program"] != "all":
                    jct.setdefault(row["arm"], {}).setdefault(lanes, {})[row["program"]] = float(row["jct_done_mean_s"])
    records = {}
    for arm in ARMS:
        for lanes in tp[arm]:
            ratios = [jct["vanilla"][lanes][p] / s for p, s in jct[arm][lanes].items()]
            records.setdefault(arm, {})[lanes] = {
                "tp_speedup": tp[arm][lanes] / tp["vanilla"][lanes],
                "jct_speedup": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
            }
    return records


class TimesFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos) + r"$\times$"


def line_style(arm: str) -> dict:
    is_ours = arm == "ours"
    return dict(
        color=colors[arm], marker=markers[arm],
        markersize=4.0 if is_ours else 2.6, markeredgewidth=0.5, markerfacecolor=colors[arm],
        linewidth=1.0 if is_ours else 0.7,
        linestyle=(0, (3, 1.5)) if arm == "vanilla" else "-",
    )


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.linewidth": 0.5, "axes.labelpad": 2,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    data = {name: read_testbed(sweep, programs) for name, (sweep, programs, _) in TESTBEDS.items()}
    for name, records in data.items():
        for arm in ARMS:
            for lanes, m in sorted(records[arm].items()):
                print(f"{name:5s} L{lanes} {arm:9s} tp {m['tp_speedup']:.2f}x  jct {m['jct_speedup']:.2f}x")

    # One y range for all four panels so the two testbeds compare at a glance.
    top = max(m[k] for r in data.values() for d in r.values() for m in d.values()
              for k in ("tp_speedup", "jct_speedup"))
    ylim = (0.9, math.ceil(top * 10) / 10 + 0.05)

    # Each testbed is one PDF of two panels, half a 3.25in column wide, sharing the y axis so
    # the tick labels appear once. Metric names sit on top as titles, not as y labels.
    pair_w, h = 3.25 / 2, 0.75
    margin_l, margin_r, margin_b, margin_t, gap = 0.27, 0.02, 0.12, 0.12, 0.05

    def plot_pair(name: str, output: str):
        records = data[name]
        levels = sorted(records["vanilla"])
        fig = plt.figure(figsize=(pair_w, h))
        panel_w = (pair_w - margin_l - margin_r - gap) / 2
        axes = []
        for i, (metric, title) in enumerate((("tp_speedup", r"Throughput ($\times$)"),
                                             ("jct_speedup", r"JCT ($\times$)"))):
            ax = fig.add_axes(((margin_l + i * (panel_w + gap)) / pair_w, margin_b / h,
                               panel_w / pair_w, 1 - (margin_b + margin_t) / h))
            ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
            ax.set_axisbelow(True)
            ax.axhline(1.0, color="#9A9A9A", linewidth=0.5, linestyle=(0, (4, 2)), zorder=1)
            for arm in ARMS:
                points = sorted(records[arm].items())
                ax.plot([lv for lv, _ in points], [v[metric] for _, v in points],
                        zorder=4 if arm == "ours" else 2, **line_style(arm))
            ax.set_title(title, fontsize=7.5, pad=2.5)
            ax.set_xticks(levels)
            ax.set_xlim(levels[0] - 0.4, levels[-1] + 0.4)
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(TimesFormatter())
            if i:
                ax.tick_params(axis="y", labelleft=False)
            axes.append(ax)
        fig.savefig(HERE / output, format="pdf")
        plt.close(fig)
        return output

    outputs = [plot_pair("tp4", "tp4.pdf"), plot_pair("3090", "3090.pdf")]
    handles = [Line2D([], [], label=display_names[a], **line_style(a)) for a in ARMS]
    fig = plt.figure(figsize=(3.25, 0.16))
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=6.5,
               handlelength=1.2, columnspacing=0.6, handletextpad=0.25,
               borderpad=0, borderaxespad=0)
    fig.savefig(HERE / "legend.pdf", format="pdf")
    plt.close(fig)
    print("Saved", *outputs, "legend.pdf")

    # Summary numbers for the text.
    for name, records in data.items():
        for metric in ("tp_speedup", "jct_speedup"):
            vals = [m[metric] for _, m in sorted(records["ours"].items())]
            gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
            print(f"{name:5s} Mortis {metric}: geomean {gm:.2f}x  max {max(vals):.2f}x")


if __name__ == "__main__":
    main()
