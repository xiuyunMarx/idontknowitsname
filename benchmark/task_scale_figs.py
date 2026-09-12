"""Task-scale figures: does the cache policy change how much work the agent does?

Per (workload, concurrency, arm): first-turn output tokens per session (typed-output retry turns excluded)
and rounds per session, mean with 95% CI over sessions. One figure per workload, two panels.
  coding -> benchmark/records/codeAgent/task_scale.{pdf,png,csv}
  fact   -> benchmark/records/FactCheck/task_scale.{pdf,png,csv}
"""
import csv, glob, os, re, sys, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fact_check"))
from fact_bench import CALL_RE  # noqa: E402

DEC = re.compile(r"^\[decode\] (\d+)-(\d+)t(\d+)-\S+ decode_ms=([\d.]+) output_tokens=(\d+)")
SESS = re.compile(r"\] session \d+ rc=(\d+) wall=[\d.]+s rounds=(\d+)")
WORKLOADS = {"coding": ("results/coding_sweep", "benchmark/records/codeAgent", "Coding agent"),
             "fact": ("results/fact_sweep_h16_inputfirst", "benchmark/records/FactCheck", "Fact check")}
ARMS = [("lruraw", "SGLang (LRU)", "#6E6E6E", "o"), ("cachescout", "CacheScout", "#B5589A", "v"),
        ("kvonly", "Planner only", "#009E73", "^"), ("lru", "Re-layout only", "#0072B2", "s"),
        ("ours", "Ours", "#D55E00", "D"), ("continuum", "Continuum", "#E69F00", "P")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def per_session(log, out):
    order, tok = [], {}
    for ln in open(log):
        m = CALL_RE.match(ln)
        if m and m.group(1) not in order:
            order.append(m.group(1))
        m = DEC.match(ln)
        if m and m.group(3) == "1":
            tok[m.group(1)] = tok.get(m.group(1), 0) + int(m.group(5))
    warm = set(order[:2])
    first_turn = [v for p, v in tok.items() if p not in warm]
    rounds = [int(m.group(2)) for m in (SESS.search(l) for l in open(out)) if m and m.group(1) == "0"]
    return first_turn, rounds


def ci(v):
    return 1.96 * st.pstdev(v) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def main():
    for wl, (src, dst, title) in WORKLOADS.items():
        data = {}
        for out in sorted(glob.glob(os.path.join(src, "*_c*.out"))):
            arm, _, c = os.path.basename(out)[:-4].rpartition("_c")
            if arm not in dict((a, 1) for a, *_ in ARMS) or not c.isdigit():
                continue
            data[(int(c), arm)] = per_session(out[:-4] + ".log", out)
        cs = sorted({c for c, _ in data})
        rows = []
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
        for ax, idx, ylabel in ((axes[0], 0, "output tokens per session"), (axes[1], 1, "rounds per session")):
            for arm, label, color, mk in ARMS:
                ys = [np.mean(data[(c, arm)][idx]) if (c, arm) in data else np.nan for c in cs]
                es = [ci(data[(c, arm)][idx]) if (c, arm) in data else 0 for c in cs]
                ax.errorbar(cs, ys, yerr=es, color=color, marker=mk, capsize=1.5, elinewidth=0.6, label=label)
            ax.set_xlabel("concurrency"); ax.set_ylabel(ylabel); ax.set_xticks(cs)
            ax.set_ylim(bottom=0); ax.grid(True, lw=0.4, alpha=0.3)
        for (c, arm), (ft, rd) in sorted(data.items()):
            rows.append(dict(concurrency=c, arm=arm, sessions=len(ft), first_turn_output_mean=round(np.mean(ft)),
                             first_turn_output_ci95=round(ci(ft)), first_turn_output_median=round(np.median(ft)),
                             rounds_mean=round(np.mean(rd), 2), rounds_ci95=round(ci(rd), 2)))
        axes[1].legend(loc="lower right", ncol=2, handlelength=1.8, columnspacing=0.8)
        fig.suptitle(title, fontsize=8, y=1.0)
        fig.tight_layout(pad=0.3)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(dst, f"task_scale.{ext}"), dpi=300)
        plt.close(fig)
        with open(os.path.join(dst, "task_scale.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(wl, "->", dst, cs)


if __name__ == "__main__":
    main()
