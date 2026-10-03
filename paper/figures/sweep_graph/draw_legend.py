import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def draw_legend(output_file: str) -> Path:
    """Draw the legend shared by all sweep panels as a standalone vector PDF."""
    # Styles must match the ones in */draw.py.
    display_names = {
        "vanilla": "Vanilla SGLang",
        "adapter": "Static-Adapter",
        "kvonly": r"\textsc{Mortis}-KV-only",
        "relayout": r"\textsc{Mortis}-Relayout-only",
        "continuum": "Continuum",
        "cachescout": "CacheScout",
        "continuum_relayout": "Continuum+Relayout",
        "cachescout_relayout": "CacheScout+Relayout",
        "kvflow": "KVFlow",
        "kvflow_relayout": "KVFlow+Relayout",
        "ours": r"\textsc{Mortis}",
    }
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
        "ours": "*",
    }
    # Two rows, filled column by column: the two layouts without analysis, the two
    # ablated variants, then each external system on the original layout (top) with
    # the same system on the re-laid layout (bottom), then the full system; None
    # leaves the slot blank.
    preferred_order = [
        "vanilla", "adapter",
        "relayout", "kvonly",
        "continuum", "continuum_relayout",
        "cachescout", "cachescout_relayout",
        "kvflow", "kvflow_relayout",
        "ours", None,
    ]

    plt.rcParams.update({
        # Text is typeset by LaTeX with the paper's Times font so that labels
        # (including \textsc{Mortis}) match the body exactly.
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "font.family": "serif",
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    handles = []
    for arm in preferred_order:
        if arm is None:
            handles.append(Line2D([], [], linestyle="none", marker="none", label=" "))
            continue
        is_ours = arm == "ours"
        is_variant = arm.endswith("_relayout")
        handles.append(Line2D(
            [],
            [],
            label=display_names[arm],
            color=colors[arm],
            marker=markers[arm],
            markersize=4.0 if is_ours else 2.6,
            markeredgewidth=0.5,
            markerfacecolor="white" if is_variant else colors[arm],
            linewidth=1.0 if is_ours else 0.7,
            linestyle=(0, (3, 1.5)) if is_variant or arm == "vanilla" else "-",
        ))

    fig = plt.figure(figsize=(7.0, 0.4))
    fig.legend(
        handles=handles,
        ncol=len(handles) // 2,
        frameon=False,
        loc="center",
        columnspacing=1.2,
        handlelength=1.8,
        labelspacing=0.35,
    )

    output_path = Path(output_file)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved {draw_legend('legend.pdf')}")
