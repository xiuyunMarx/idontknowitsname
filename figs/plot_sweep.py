"""Three comparison figures (LRU vs ours) for the Qwen3-8B rows of sweep_results.csv: TTFT, JCT, hit rate.
usage: plot8b.py sweep_results.csv OUT_DIR [run_label=q8b]"""
import csv, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV, OUT = sys.argv[1], sys.argv[2]; RUN = sys.argv[3] if len(sys.argv) > 3 else "q8b"; NAME = sys.argv[4] if len(sys.argv) > 4 else RUN
rows = [r for r in csv.DictReader(open(CSV)) if r["run"] == RUN]
CS = sorted({int(r["concurrency"]) for r in rows})
data = {(r["arm"], int(r["concurrency"])): r for r in rows}
model = rows[0]["model"]; dev_tok = int(rows[0]["device_tokens"]); host_tok = int(rows[0]["host_tokens"])
nsess = "/".join(data[("lru", c)]["sessions"] for c in CS)
STEP = min(b - a for a, b in zip(CS, CS[1:])) if len(CS) > 1 else 1   # x spacing between levels
ARMS = [("lru", "LRU", "#2a78d6"), ("ours", "Ours", "#eb6834")]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DEV, HOST, MISS = "#1c5cab", "#86b6ef", "#dcdbd5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
sub = f"{model}, device pool {dev_tok/1000:.1f}k tokens, host {host_tok/1000:.0f}k tokens, {nsess} sessions per level"

def style(ax, ylabel):
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5); ax.set_xlabel("Concurrent sessions"); ax.set_ylabel(ylabel)
    ax.set_xticks(CS); ax.set_xlim(CS[0] - STEP, CS[-1] + 1.6 * STEP)

def lines(ax, col_name, scale, ylabel, title, pct_at=None, pp=False):
    pct_at = CS[-2:] if pct_at is None else pct_at
    ys = {}
    for arm, label, col in ARMS:
        y = np.array([float(data[(arm, c)][col_name]) * scale for c in CS]); ys[arm] = y
        ax.plot(CS, y, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, solid_joinstyle="round", solid_capstyle="round", label=label, zorder=3)
        ax.annotate(label, (CS[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    for c in pct_at:  # selective direct label: ours vs LRU at the loaded end
        i = CS.index(c); d = ys["ours"][i] - ys["lru"][i] if pp else (ys["ours"][i] / ys["lru"][i] - 1) * 100
        txt = f"{d:+.1f} pp" if pp else f"{d:+.0f}%"
        ax.annotate(txt, (c, ys["ours"][i]), xytext=(0, -13 if d < 0 else 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    style(ax, ylabel); ax.set_title(title); ax.set_ylim(0, None)
    return ys

# ---------- Figure 1: TTFT ----------
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
lines(axs[0], "ttft_p50_ms", 1e-3, "TTFT (s)", "Median TTFT, all calls")
lines(axs[1], "ttft_mean_ms", 1e-3, "TTFT (s)", "Mean TTFT, all calls")
lines(axs[2], "ttft_p95_ms", 1e-3, "TTFT (s)", "p95 TTFT, all calls")
axs[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Time to first token: " + sub, fontsize=9.5, color=INK2, ha="left", x=0.01)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/{NAME}_ttft.{ext}", dpi=200)

# ---------- Figure 2: JCT ----------
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
ys = lines(axs[0], "jct_mean_s", 1, "JCT (s)", "Mean job completion time")
lines(axs[1], "jct_p95_s", 1, "JCT (s)", "p95 job completion time")
axs[0].legend(loc="upper left", fontsize=8.5)
ax = axs[2]
d = (ys["ours"] / ys["lru"] - 1) * 100
ax.bar(CS, d, width=0.55 * STEP, color="#eb6834", edgecolor=SURF, linewidth=1, zorder=3); ax.axhline(0, color=AXIS, lw=0.8)
for c, v in zip(CS, d):
    ax.annotate(f"{v:+.1f}%", (c, v), xytext=(0, -10 if v < 0 else 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
style(ax, "Change in mean JCT (%)"); ax.set_title("Ours relative to LRU, lower is better"); ax.grid(False)
ax.set_ylim(min(d.min() * 1.6, -2), max(d.max() * 1.6, 2)); ax.set_xlim(CS[0] - 0.6 * STEP, CS[-1] + 0.6 * STEP)
fig.suptitle("Job completion time: " + sub, fontsize=9.5, color=INK2, ha="left", x=0.01)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/{NAME}_jct.{ext}", dpi=200)

# ---------- Figure 3: hit rate ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]; w = 0.34 * STEP
for j, (arm, label, col) in enumerate(ARMS):
    x = np.array(CS) + (j - 0.5) * (w + 0.04 * STEP)
    dev = np.array([float(data[(arm, c)]["device_pct"]) for c in CS]); host = np.array([float(data[(arm, c)]["host_pct"]) for c in CS]); miss = np.array([float(data[(arm, c)]["miss_pct"]) for c in CS])
    kw = dict(width=w, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.bar(x, dev, color=DEV, **kw); ax.bar(x, host, bottom=dev, color=HOST, **kw); ax.bar(x, miss, bottom=dev + host, color=MISS, **kw)
    for xi, v in zip(x, dev):  # label the segment the comparison is about
        ax.annotate(f"{v:.0f}", (xi, v / 2), ha="center", va="center", fontsize=7.5, color=SURF if v >= 7 else INK2)
    for xi in x: ax.annotate("L" if arm == "lru" else "O", (xi, 98), ha="center", va="top", fontsize=8, color=INK2)
style(ax, "Share of prompt tokens (%)"); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xlim(CS[0] - 0.7 * STEP, CS[-1] + 0.7 * STEP)
ax.set_title("Source of each prompt token (L = LRU, O = ours; number = device hit %)")
ax.legend(handles=[Patch(color=DEV, label="Device hit"), Patch(color=HOST, label="Host hit"), Patch(color=MISS, label="Miss (recomputed)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8.5)
ax = axs[1]
lines(ax, "miss_pct", 1, "Miss (%)", "Miss rate, all calls", pp=True)
ax.set_ylim(0, 100); ax.legend(loc="upper left", fontsize=8.5)
fig.suptitle("Prefix cache hit rate: " + sub, fontsize=9.5, color=INK2, ha="left", x=0.01)
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/{NAME}_hitrate.{ext}", dpi=200)
print("written:", sorted(os.listdir(OUT)))
