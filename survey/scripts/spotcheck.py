"""Draw a stratified sample of workflows for manual verification of the parsers.

Writes data/spotcheck/sample.csv with the parser's view (sites, models) and empty manual columns;
`--score` compares filled manual columns with the parser output.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, PARSED, RAW, log, read_jsonl  # noqa: E402

OUT = DATA / "spotcheck" / "sample.csv"


def draw(seed=20260830):
    df = pd.read_csv(PARSED / "workflows.csv", keep_default_na=False)
    df = df[df["dup"].astype(str).str.lower() != "true"]
    rnd = random.Random(seed)
    picks = []
    n8n = df[(df["corpus"] == "n8n_api") & (df["n_sites"] >= 1)]
    for label, sub, k in [("n8n_single", n8n[(n8n["n_models"] < 2) & (n8n["n_unknown"] == 0)], 10),
                          ("n8n_multi", n8n[n8n["n_models"] >= 2], 10),
                          ("n8n_unknown_or_default", n8n[(n8n["n_unknown"] > 0) | (n8n["n_default"] > 0)], 10),
                          ("dify", df[(df["corpus"] == "dify") & (df["n_sites"] >= 1)], 10),
                          ("flowise", df[(df["corpus"] == "flowise") & (df["n_sites"] >= 1)], 10),
                          ("langflow", df[(df["corpus"] == "langflow")], 5)]:
        ids = list(sub["workflow_id"])
        rnd.shuffle(ids)
        for wid in ids[:k]:
            r = sub[sub["workflow_id"] == wid].iloc[0]
            picks.append(dict(stratum=label, corpus=r["corpus"], workflow_id=wid, name=r["name"], parser_n_sites=r["n_sites"],
                              parser_n_models=r["n_models"], parser_models=r["models"], parser_n_unknown=r["n_unknown"],
                              parser_n_default=r["n_default"], manual_n_sites="", manual_n_models="", manual_models="", note=""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(picks).to_csv(OUT, index=False)
    log(f"wrote {len(picks)} sample rows to {OUT}")


def score():
    df = pd.read_csv(OUT, keep_default_na=False)
    # re-join the parser's current view so re-runs of the pipeline are scored, not the stale sample columns
    cur = pd.read_csv(PARSED / "workflows.csv", keep_default_na=False).set_index(["corpus", "workflow_id"])
    for i, r in df.iterrows():
        key = (r["corpus"], r["workflow_id"])
        if key in cur.index:
            df.at[i, "parser_n_sites"] = cur.loc[key, "n_sites"]
            df.at[i, "parser_n_models"] = cur.loc[key, "n_models"]
            df.at[i, "parser_models"] = cur.loc[key, "models"]
    done = df[df["manual_n_sites"].astype(str).str.strip() != ""]
    if not len(done):
        log("no manual labels yet")
        return
    s_ok = (pd.to_numeric(done["manual_n_sites"]) == pd.to_numeric(done["parser_n_sites"])).mean()
    m_ok = (pd.to_numeric(done["manual_n_models"]) == pd.to_numeric(done["parser_n_models"])).mean()
    multi_ok = ((pd.to_numeric(done["manual_n_models"]) >= 2) == (pd.to_numeric(done["parser_n_models"]) >= 2)).mean()
    res = dict(n_labelled=int(len(done)), site_count_agreement=round(float(s_ok), 3), model_count_agreement=round(float(m_ok), 3),
               multi_model_agreement=round(float(multi_ok), 3),
               per_stratum={k: int(v) for k, v in done["stratum"].value_counts().items()})
    json.dump(res, open(OUT.parent / "score.json", "w"), indent=1)
    log(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    score() if a.score else draw()
