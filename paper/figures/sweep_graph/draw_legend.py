import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def draw_legend(output_file: str) -> Path:
    """Draw the legend shared by all sweep panels as a standalone vector PDF."""
    # Styles must match the ones in */draw.py.
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
    # Two rows, filled column by column: each column pairs a system on the
    # original layout (top) with the same system on the re-laid layout (bottom).
    preferred_order = [
        "vanilla", "relayout",
        "continuum", "continuum_relayout",
        "cachescout", "cachescout_relayout",
        "kvflow", "kvflow_relayout",
        "kvonly", "ours",
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    handles = []
    for arm in preferred_order:
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
        columnspacing=1.6,
        handlelength=2.4,
        labelspacing=0.35,
    )

    output_path = Path(output_file)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved {draw_legend('legend.pdf')}")
