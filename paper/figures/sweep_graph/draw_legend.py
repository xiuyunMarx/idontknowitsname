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

    plt.rcParams.update({
        "font.family": "sans-serif",
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    handles = []
    for arm in preferred_order:
        is_ours = arm == "ours"
        handles.append(Line2D(
            [],
            [],
            label=display_names[arm],
            color=colors[arm],
            marker=markers[arm],
            markersize=4.5 if is_ours else 3.0,
            markeredgewidth=0.7,
            linewidth=1.5 if is_ours else 1.0,
            linestyle="--" if arm == "vanilla" else "-",
        ))

    fig = plt.figure(figsize=(7.0, 0.3))
    fig.legend(
        handles=handles,
        ncol=len(handles),
        frameon=False,
        loc="center",
        columnspacing=1.6,
        handlelength=2.4,
    )

    output_path = Path(output_file)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved {draw_legend('legend.pdf')}")
