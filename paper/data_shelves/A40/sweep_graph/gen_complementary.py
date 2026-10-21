# Summurizing data for supporting complementary effect of mechanisms
import pandas as pd
import numpy as np
import os
import csv
SAMPLE_AGENT = "coding"

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = {"fact_check": "fact", "coding": "coding", "BFCL": "bfcl", "finance": "finance"}
# 2x2 factorial over the two pillars: (relayout on?, scheduling on?) -> arm
ARMS = {
    (False, False): "vanilla",
    (True, False): "relayout",
    (False, True): "kvonly",
    (True, True): "ours",
}


def fetch_data(agent) -> pd.DataFrame:
    """Long table: one row per (c, relayout, scheduling) with JCT and throughput speedup over vanilla."""
    frames = {}
    for key, arm in ARMS.items():
        path = os.path.join(HERE, agent, f"{PREFIX[agent]}_sweep_{arm}.csv")
        frames[key] = pd.read_csv(path).set_index("c")[["jct_mean_s", "prompt_ktokens_per_min"]]
    base = frames[(False, False)]
    rows = []
    for (relayout, sched), df in frames.items():
        for c in df.index:
            rows.append({
                "agent": agent,
                "c": int(c),
                "relayout": relayout,
                "scheduling": sched,
                "jct_speedup": base.loc[c, "jct_mean_s"] / df.loc[c, "jct_mean_s"],
                "tp_speedup": df.loc[c, "prompt_ktokens_per_min"] / base.loc[c, "prompt_ktokens_per_min"],
            })
    return pd.DataFrame(rows)


def gmean(x):
    return float(np.exp(np.log(x).mean()))


def factorial(df) -> pd.DataFrame:
    """2x2 of geomean speedup over every (agent, c) in df; cells are (jct, tp)."""
    g = df.groupby(["relayout", "scheduling"]).agg(jct=("jct_speedup", gmean), tp=("tp_speedup", gmean))
    return g


def interaction(g) -> tuple:
    """Superadditivity: both / (relayout-only x scheduling-only), per metric."""
    both, r, s = g.loc[(True, True)], g.loc[(True, False)], g.loc[(False, True)]
    return both.jct / (r.jct * s.jct), both.tp / (r.tp * s.tp)


def to_latex(g, caption_levels: str) -> str:
    """Two side-by-side 2x2 tabulars, one per metric."""
    def tabular(metric):
        def cell(relayout, sched):
            return f"{g.loc[(relayout, sched)][metric]:.2f}$\\times$"
        return [
            "\\begin{tabular}{lcc}",
            "\\toprule",
            " & No scheduling & Scheduling \\\\",
            "\\midrule",
            f"No re-layout & {cell(False, False)} & {cell(False, True)} \\\\",
            f"Re-layout & {cell(True, False)} & {cell(True, True)} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ]
    return "\n".join(
        [f"% geomean over {caption_levels}", "% JCT speedup"] + tabular("jct")
        + ["\\hfill", "% Throughput speedup"] + tabular("tp")
    )


def report(df, name):
    g = factorial(df)
    ij, it = interaction(g)
    n = len(df) // len(ARMS)
    print(f"== {name}: geomean over {n} (agent, c) points per arm")
    print(g.round(2))
    print(f"interaction both/(relayout x scheduling): JCT {ij:.2f}, throughput {it:.2f}\n")
    print(to_latex(g, f"{n} concurrency points, {name}"))
    print()


def main():
    sample = fetch_data(SAMPLE_AGENT)
    report(sample, SAMPLE_AGENT)
    pooled = pd.concat(fetch_data(a) for a in PREFIX)
    report(pooled, "all four agents")


if __name__ == "__main__":
    main()
