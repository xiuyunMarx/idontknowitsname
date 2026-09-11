"""Per-concurrency CDFs of the recomputed fraction of each LLM call (recomputed / input tokens).

One figure per concurrency, five arms, all measured calls (warmup sessions excluded, first turn per call).
  coding  -> benchmark/records/codeAgent/recompute_cdf_c<N>.{pdf,png}
  fact    -> benchmark/records/FactCheck/recompute_cdf_c<N>.{pdf,png}
The per-call rows behind every curve go to <dir>/recompute_cdf_calls.csv (one row per LLM call).
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_figs import load  # noqa: E402

OUT = {"coding": "benchmark/records/codeAgent", "fact": "benchmark/records/FactCheck"}
CALL_FIELDS = ["concurrency", "arm", "session_pid", "call_index", "site", "site_invocation",
               "prompt_tokens", "cached_device", "cached_host", "recomputed", "recomputed_fraction", "ttft_ms"]
ARMS = [("lruraw", "SGLang (LRU)", "#6E6E6E", (0, (1, 1))),
        ("cachescout", "CacheScout", "#B5589A", (0, (4, 1.5))),
        ("kvonly", "Planner only", "#009E73", (0, (3, 1, 1, 1))),
        ("lru", "Re-layout only", "#0072B2", (0, (6, 2))),
        ("ours", "Ours", "#D55E00", "-")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    for wl, out in OUT.items():
        os.makedirs(out, exist_ok=True)
        runs = load(wl)
        with open(os.path.join(out, "recompute_cdf_calls.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CALL_FIELDS); w.writeheader()
            for (c, arm), ss in sorted(runs.items()):
                for s in ss:
                    site = s["site"] if arm != "cachescout" else ""   # CacheScout logs carry no call-site names
                    w.writerow(dict(concurrency=c, arm=arm, session_pid=s["pid"], call_index=s["call"], site=site,
                                    site_invocation=s["inv"] if site else "", prompt_tokens=int(s["prompt_tokens"]),
                                    cached_device=int(s["cached_device"]), cached_host=int(s["cached_host"]),
                                    recomputed=int(s["recompute"]),
                                    recomputed_fraction=round(s["recompute"] / s["prompt_tokens"], 4),
                                    ttft_ms=s["ttft_ms"]))
        for c in sorted({c for c, _ in runs}):
            fig, ax = plt.subplots(figsize=(3.3, 2.2))
            for arm, label, color, ls in ARMS:
                ss = runs.get((c, arm))
                if not ss:
                    continue
                x = np.sort([s["recompute"] / s["prompt_tokens"] for s in ss])
                y = np.arange(1, len(x) + 1) / len(x)
                ax.step(np.concatenate([[0], x]), np.concatenate([[0], y]), where="post",
                        color=color, ls=ls, label=label)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.set_xlabel("recomputed fraction")
            ax.set_ylabel("CDF of LLM calls")
            ax.grid(True, lw=0.4, alpha=0.3)
            ax.legend(loc="upper left", handlelength=2.4, labelspacing=0.3, borderaxespad=0.4)
            ax.text(0.98, 0.04, f"$c={c}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
            fig.tight_layout(pad=0.3)
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out, f"recompute_cdf_c{c}.{ext}"), dpi=300)
            plt.close(fig)
        print(wl, "->", out, sorted({c for c, _ in runs}))


if __name__ == "__main__":
    main()
