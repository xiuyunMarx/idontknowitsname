"""Figures for REPORT.md from data/parsed/workflows.csv and summary.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, PARSED, log  # noqa: E402

LABELS = {"n8n_api": "n8n official\ntemplate library", "n8n_hf": "n8n HF dumps\n(cross-check)", "dify": "Dify DSL\ncollections",
          "flowise": "Flowise\nmarketplace", "langflow": "Langflow\nstarters"}
ORDER = ["n8n_api", "dify", "flowise", "n8n_hf"]


def fig1(df, summary):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    cols = ["#d9d9d9", "#6baed6", "#3182bd", "#08519c"]
    xs = []
    for i, c in enumerate(ORDER):
        d = summary["corpora"].get(c)
        if not d or d["p_multi_model"] is None:
            continue
        dist = d["n_models_dist"]
        n = d["n_attributable"]
        vals = [dist.get(k, 0) / n for k in ("1", "2", "3", "4")]
        vals[0] += dist.get("0", 0) / n
        bottom = 0
        for v, col, lab in zip(vals, cols, ["1 model", "2 models", "3 models", "≥4 models"]):
            ax.bar(i, v, bottom=bottom, color=col, edgecolor="white", label=lab if i == 0 else None)
            bottom += v
        p, lb, ub = d["p_multi_model"], d["p_multi_model_lb"], d["p_multi_model_ub"]
        ax.errorbar(i, 1 - p + 0.0, yerr=[[max(0, ub - p)], [max(0, p - lb)]], fmt="none", ecolor="black", capsize=4, lw=1.2)
        ax.text(i, 1.02, f"n={n}\n≥2 models: {100 * p:.1f}%\n[{100 * lb:.1f}–{100 * ub:.1f}]", ha="center", va="bottom", fontsize=8.5)
        xs.append((i, LABELS[c]))
    ax.set_xticks([i for i, _ in xs])
    ax.set_xticklabels([l for _, l in xs], fontsize=9)
    ax.set_ylim(0, 1.32)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylabel("share of LLM workflows")
    ax.set_title("Distinct models per workflow in public agent/workflow templates", fontsize=11, pad=4)
    ax.legend(loc="upper right", fontsize=8, ncol=4, frameon=False, bbox_to_anchor=(1.0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_prevalence.png", dpi=180)
    fig.savefig(FIGURES / "fig1_prevalence.svg")
    plt.close(fig)


def fig2(df, summary):
    prim = df[(df["corpus"] == "n8n_api") & (~df["dup"]) & (df["n_sites"] >= 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    # (a) sites per workflow
    ax = axes[0]
    vals = prim["n_sites"].clip(upper=10)
    ax.hist(vals, bins=np.arange(0.5, 11.5, 1), color="#3182bd", edgecolor="white")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(i) for i in range(1, 10)] + ["10+"])
    ax.set_xlabel("LLM call sites per workflow")
    ax.set_ylabel("workflows")
    ax.set_title(f"(a) n8n official: call sites (median {prim['n_sites'].median():.0f}, p90 {prim['n_sites'].quantile(.9):.0f})", fontsize=9.5)
    # (b) sites by single vs multi model
    ax = axes[1]
    attr = prim[prim["n_explicit"] + prim["n_default"] >= 1]
    single = attr[attr["n_models"] < 2]["n_sites"].clip(upper=10)
    multi = attr[attr["n_models"] >= 2]["n_sites"].clip(upper=10)
    bins = np.arange(0.5, 11.5, 1)
    ax.hist([single, multi], bins=bins, color=["#d9d9d9", "#08519c"], label=[f"1 model (n={len(single)})", f"≥2 models (n={len(multi)})"], density=True)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(i) for i in range(1, 10)] + ["10+"])
    ax.set_xlabel("LLM call sites per workflow")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) call sites: single- vs multi-model workflows", fontsize=9.5)
    # (c) tier pair heatmap
    ax = axes[2]
    tiers = ["xs", "s", "m", "l"]
    M = np.zeros((4, 4))
    for tp, n in summary["pooled"]["tier_pairs"]:
        a, b = tp.split("-")
        i, j = tiers.index(a), tiers.index(b)
        M[i, j] += n
        if i != j:
            M[j, i] += n
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(tiers); ax.set_yticklabels(tiers)
    for i in range(4):
        for j in range(4):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_title("(c) size-tier pairs in ≥2-model workflows\n(xs ≤2B, s ≤9B/mini/flash, m, l = pro/opus/4o/>35B)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_mix.png", dpi=180)
    fig.savefig(FIGURES / "fig2_mix.svg")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(PARSED / "workflows.csv", keep_default_na=False)
    df["dup"] = df["dup"].astype(str).str.lower() == "true"
    for c in ("n_sites", "n_models", "n_explicit", "n_default"):
        df[c] = pd.to_numeric(df[c])
    summary = json.load(open(PARSED / "summary.json"))
    fig1(df, summary)
    fig2(df, summary)
    log("figures written")
