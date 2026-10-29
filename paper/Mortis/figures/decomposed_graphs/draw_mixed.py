"""Mixed-workload lane sweep split into two batches, each with throughput, JCT and hit-rate panels plus a legend:
raw_baselines over Vanilla SGLang, relayout+baselines over Mortis-Relayout-only.
Data loading and panel geometry come from mix_sweep/draw.py."""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

SHELF = Path("/home/xiaoyu/idontknowitsname/paper/data_shelves/A40/mix_sweep")
OUT = Path("/home/xiaoyu/idontknowitsname/paper/Mortis_Lingjia/figures/decomposed_graphs/mixed")
sys.path.insert(0, str(SHELF))
import draw as mix  # noqa: E402

DISPLAY_NAMES = {
    "vanilla": "SGLang",
    "adapter": "Static-Adapter",
    "kvonly": r"\textsc{Mortis}-No-Relayout",
    "relayout": r"\textsc{Mortis}-Relayout-only",
    "continuum": "Continuum",
    "cachescout": "CacheScout",
    "kvflow": "KVFlow",
    "continuum_relayout": "Continuum+Relayout",
    "cachescout_relayout": "CacheScout+Relayout",
    "kvflow_relayout": "KVFlow+Relayout",
    "ours_noprefetch": r"\textsc{Mortis}-No-Prefetch",
    "ours": r"\textsc{Mortis}",
}
# Same colors and markers as the per-agent batches; a *_relayout arm shares its raw arm's style.
COLORS = {
    "adapter": "#A0522D",
    "kvonly": "#0072B2",
    "relayout": "#009E73",
    "continuum": "#E69F00",
    "cachescout": "#CC79A7",
    "kvflow": "#56B4E9",
    "ours_noprefetch": "#D55E00",
    "ours": "#D55E00",
}
MARKERS = {"adapter": "h", "relayout": "^", "kvonly": "s", "continuum": "D", "cachescout": "v", "kvflow": "P",
           "ours_noprefetch": "*", "ours": "*"}

# (directory, baseline arm drawn as the gray dashed line, plotted arms in draw/legend order)
BATCHES = [
    ("raw_baselines", "vanilla", ["adapter", "continuum", "cachescout", "kvflow", "kvonly", "relayout", "ours_noprefetch", "ours"]),
    ("relayout+baselines", "relayout",
     ["continuum_relayout", "cachescout_relayout", "kvflow_relayout", "ours_noprefetch", "ours"]),
]
BASELINE_STYLE = dict(color="#7F7F7F", linewidth=0.8, linestyle=(0, (4, 2)))


def line_style(arm: str) -> dict:
    base = arm.removesuffix("_relayout")
    is_ours = arm == "ours"
    # No-Prefetch is a variant of Mortis: same color and marker, dashed and hollow.
    is_variant = arm == "ours_noprefetch"
    return dict(
        color=COLORS[base],
        marker=MARKERS[base],
        markersize=4.0 if is_ours else 2.6,
        markeredgewidth=0.5,
        markerfacecolor="white" if is_variant else COLORS[base],
        linewidth=1.0 if is_ours else 0.7,
        linestyle=(0, (3, 1.5)) if is_variant else "-",
    )


def load(base_arm: str) -> dict:
    cwd = os.getcwd()
    os.chdir(SHELF)
    try:
        records = mix.read_csv(mix.FILE)
        mix.read_jct(records)
    finally:
        os.chdir(cwd)
    for data in records.values():
        for level, m in data.items():
            base = records[base_arm][level]
            m["tp_speedup"] = m["throughput"] / base["throughput"]
            m["jct_speedup"] = mix.jct_speedup(m["jct"], base["jct"])
    return records


def plot_panel(records: dict, base_arm: str, arms: list, metric: str, speedup_ylim: tuple, path: Path) -> None:
    panel_w, panel_h = 3.25 / 3, 0.86
    margin_l, margin_r, margin_b, margin_t = 0.25, 0.03, 0.25, 0.04
    fig = plt.figure(figsize=(panel_w, panel_h))
    ax = fig.add_axes([margin_l / panel_w, margin_b / panel_h,
                       1 - (margin_l + margin_r) / panel_w, 1 - (margin_b + margin_t) / panel_h])
    levels = sorted(records[base_arm])
    is_speedup = metric.endswith("_speedup")
    if is_speedup:
        ax.axhline(1.0, zorder=1, **BASELINE_STYLE)
    else:
        # Hit rate is absolute, so the baseline is a real curve in the same gray dashed style.
        ax.plot(levels, [records[base_arm][l][metric] for l in levels], zorder=1, **BASELINE_STYLE)
    for arm in arms:
        points = sorted(records[arm].items())
        ax.plot([l for l, _ in points], [m[metric] for _, m in points],
                zorder=4 if arm == "ours" else 2, **line_style(arm))

    ax.set_xlabel("Lanes per agent")
    ax.set_xticks(range(levels[0], levels[-1] + 1, 2))
    ax.set_xticks(range(levels[0], levels[-1] + 1), minor=True)
    ax.set_xlim(levels[0] - 0.4, levels[-1] + 0.4)
    if is_speedup:
        # Throughput and JCT panels of one batch share one scale.
        ax.set_ylim(*speedup_ylim)
        step = 0.2 if speedup_ylim[1] - speedup_ylim[0] > 0.6 else 0.1
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
    else:
        ax.set_ylim(0, 62)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"{v:.0f}\%"))
    ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.4, linestyle=(0, (1, 1.5)))
    ax.set_axisbelow(True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def plot_legend(base_arm: str, arms: list, path: Path) -> None:
    # At most four entries per row; rows read left to right, but matplotlib fills columns first, so reorder.
    handles = [Line2D([], [], label=DISPLAY_NAMES[base_arm] + r" (=1$\times$)", **BASELINE_STYLE)]
    handles += [Line2D([], [], label=DISPLAY_NAMES[a], **line_style(a)) for a in arms]
    nrows = -(-len(handles) // 4)
    ncol = -(-len(handles) // nrows)
    handles += [Line2D([], [], linestyle="none", marker="none", label=" ")] * (nrows * ncol - len(handles))
    handles = [handles[r * ncol + c] for c in range(ncol) for r in range(nrows)]
    fig = plt.figure(figsize=(3.25, 0.088 * nrows))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False, fontsize=6.2,
               handlelength=1.3, columnspacing=0.6, handletextpad=0.35, labelspacing=0.12,
               borderpad=0, borderaxespad=0)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> None:
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
    for batch, base_arm, arms in BATCHES:
        records = load(base_arm)
        values = [m[k] for a in arms for m in records[a].values() for k in ("tp_speedup", "jct_speedup")]
        print(batch, f"speedup range {min(values):.3f}-{max(values):.3f}")
        lo = min(0.9, round(min(values) - 0.05, 1))
        hi = round(max(values) + 0.05, 2)
        out_dir = OUT / batch
        out_dir.mkdir(parents=True, exist_ok=True)
        for metric, name in [("tp_speedup", "throughput_speedup.pdf"), ("jct_speedup", "jct_speedup.pdf"),
                             ("hit_rate", "hit_rate.pdf")]:
            plot_panel(records, base_arm, arms, metric, (lo, hi), out_dir / name)
            print(f"Saved {out_dir / name}")
        plot_legend(base_arm, arms, out_dir / "legend.pdf")
        print(f"Saved {out_dir / 'legend.pdf'}")


if __name__ == "__main__":
    main()
