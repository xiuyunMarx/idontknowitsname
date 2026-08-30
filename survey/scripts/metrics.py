"""Per-workflow metrics, de-duplication, and summary.json for the report."""
import collections
import itertools
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARSED, log, read_jsonl  # noqa: E402

GEN_ROLES = {"chat", "agent", "classifier", "extractor"}
CORPORA = ["n8n_api", "n8n_hf", "dify", "flowise", "langflow"]


def load():
    wf = []
    for f in sorted(PARSED.glob("*_workflows.jsonl")):
        wf += list(read_jsonl(f))
    cs = list(read_jsonl(PARSED / "callsites_norm.jsonl"))
    return wf, cs


def per_workflow(w, sites):
    gen = [s for s in sites if s["role"] in GEN_ROLES]
    site_key = lambda s: s.get("node_id") or s.get("node_name")  # noqa: E731
    n_sites_distinct = len({site_key(s) for s in gen})
    n_sites_all_distinct = len({site_key(s) for s in sites if s["role"] != "dangling"})
    explicit = [s for s in gen if s["resolution"] == "explicit"]
    defaults = [s for s in gen if s["resolution"] == "default"]
    unknown = [s for s in gen if s["resolution"] in ("unknown", "expression")]
    E = {s["model_norm"] for s in explicit}
    E_prov = {s["provider"] for s in explicit}
    # headline: a default site is assumed to equal an explicit model of the same provider if one exists,
    # otherwise it contributes one "<provider>:default" model
    D = {f"{s['provider']}:default" for s in defaults if s["provider"] not in E_prov}
    n_models = len(E | D)
    n_models_lb = len(E) if E else (1 if gen else 0)
    n_models_ub = len(E) + len({(s["provider"], s.get("lm_node") or s["node_type"]) for s in defaults}) + len(unknown)
    vendors = {s["vendor"] for s in explicit if s["vendor"]} | {s["provider"] for s in defaults if s["provider_class"] == "closed_api"}
    tiers = {s["tier"] for s in explicit if s["tier"]}
    open_models = {s["model_norm"] for s in explicit if s["open_weight"] is True}
    families = {(s["vendor"], s["family"]) for s in explicit if s["family"]}
    tier_by_model = {s["model_norm"]: s["tier"] for s in explicit if s["tier"]}
    return dict(
        corpus=w["corpus"], source=w["source"], workflow_id=w["workflow_id"], name=w["name"],
        n_nodes=w["n_nodes"], graph_hash=w["graph_hash"], chain_len=w["chain_len"], has_router=bool(w.get("has_router")),
        views=w.get("views"),
        n_sites=n_sites_distinct, n_sites_all=n_sites_all_distinct, n_site_model_pairs=len(gen), n_explicit=len(explicit), n_default=len(defaults), n_unknown=len(unknown),
        n_models=n_models, n_models_lb=n_models_lb, n_models_ub=n_models_ub, n_families=len(families),
        n_vendors=len(vendors), n_tiers=len(tiers), n_open_models=len(open_models),
        any_open=bool(open_models), any_self_host=any(s["provider_class"] == "self_host" for s in gen),
        any_open_hosted=any(s["provider_class"] == "open_hosted" for s in gen),
        has_embedding=any(s["role"] == "embedding" for s in sites), has_reranker=any(s["role"] == "reranker" for s in sites),
        models="|".join(sorted(E | D)), tiers="|".join(sorted(tiers)),
        tier_pairs="|".join(f"{a}-{b}" for a, b in itertools.combinations(sorted(tier_by_model.values()), 2)) if len(tier_by_model) >= 2 else "",
    )


def dedup(df):
    """Within corpus: keep one workflow per graph_hash (highest views first, then first seen)."""
    df = df.copy()
    df["views_f"] = pd.to_numeric(df["views"], errors="coerce").fillna(-1)
    df = df.sort_values(["corpus", "graph_hash", "views_f"], ascending=[True, True, False])
    df["dup"] = df.duplicated(["corpus", "graph_hash"], keep="first")
    return df.drop(columns=["views_f"])


def rate(mask, denom):
    return round(float(mask.sum()) / denom, 4) if denom else None


def summarise(df, cs):
    out = {"corpora": {}, "pooled": {}}
    for corpus, g_all in df.groupby("corpus"):
        g = g_all[~g_all["dup"]]
        llm = g[g["n_sites"] >= 1]
        multi_site = g[g["n_sites"] >= 2]
        attributable = llm[llm["n_explicit"] + llm["n_default"] >= 1]  # at least one site with a resolvable model
        d = dict(
            n_all_raw=int(len(g_all)), n_all=int(len(g)), n_dup_removed=int(g_all["dup"].sum()),
            n_llm=int(len(llm)), n_multi_site=int(len(multi_site)), n_attributable=int(len(attributable)),
            p_llm=rate(g["n_sites"] >= 1, len(g)),
            p_multi_site=rate(llm["n_sites"] >= 2, len(llm)),
            p_multi_model=rate(attributable["n_models"] >= 2, len(attributable)),
            p_multi_model_lb=rate(attributable["n_models_lb"] >= 2, len(attributable)),
            p_multi_model_ub=rate(attributable["n_models_ub"] >= 2, len(attributable)),
            p_multi_model_of_multi_site=rate(multi_site[multi_site["n_explicit"] + multi_site["n_default"] >= 1]["n_models"] >= 2,
                                             int((multi_site["n_explicit"] + multi_site["n_default"] >= 1).sum())),
            p_multi_vendor=rate(attributable["n_vendors"] >= 2, len(attributable)),
            p_multi_tier=rate(attributable["n_tiers"] >= 2, len(attributable)),
            p_multi_family=rate(attributable["n_families"] >= 2, len(attributable)),
            p_any_open=rate(attributable["any_open"], len(attributable)),
            p_multi_open=rate(attributable["n_open_models"] >= 2, len(attributable)),
            p_any_self_host=rate(attributable["any_self_host"], len(attributable)),
            p_multi_model_given_self_host=rate(attributable[attributable["any_self_host"]]["n_models"] >= 2, int(attributable["any_self_host"].sum())),
            p_router=rate(llm["has_router"], len(llm)),
            p_erg=rate(llm["has_embedding"] & llm["has_reranker"], len(llm)),
            n_models_dist={str(k): int(v) for k, v in attributable["n_models"].clip(upper=4).value_counts().sort_index().items()},
            unknown_site_rate=rate(llm["n_unknown"] > 0, len(llm)),
            default_site_rate=rate(llm["n_default"] > 0, len(llm)),
            sites_median=float(llm["n_sites"].median()) if len(llm) else None,
            sites_p90=float(llm["n_sites"].quantile(0.9)) if len(llm) else None,
            sites_max=int(llm["n_sites"].max()) if len(llm) else None,
            chain_median=float(llm["chain_len"].median()) if len(llm) else None,
            chain_p90=float(llm["chain_len"].quantile(0.9)) if len(llm) else None,
            sites_median_single=float(attributable[attributable["n_models"] < 2]["n_sites"].median()) if len(attributable) else None,
            sites_median_multi=float(attributable[attributable["n_models"] >= 2]["n_sites"].median()) if (attributable["n_models"] >= 2).any() else None,
        )
        out["corpora"][corpus] = d
    # pooled detail on primary corpus
    prim = df[(df["corpus"] == "n8n_api") & (~df["dup"])]
    prim_ids = set(prim["workflow_id"])
    gen = [s for s in cs if s["corpus"] == "n8n_api" and s["workflow_id"] in prim_ids and s["role"] in GEN_ROLES]
    expl = [s for s in gen if s["resolution"] == "explicit"]
    top_models = collections.Counter(s["model_norm"] for s in expl).most_common(20)
    top_open = collections.Counter(s["model_norm"] for s in expl if s["open_weight"] is True).most_common(15)
    top_prov = collections.Counter(s["provider"] for s in gen).most_common(20)
    multi = prim[prim["n_models"] >= 2]
    pairs = collections.Counter()
    for ms in multi["models"]:
        for a, b in itertools.combinations(sorted(ms.split("|")), 2):
            pairs[f"{a} + {b}"] += 1
    tier_pairs = collections.Counter()
    for tp in multi["tier_pairs"]:
        for t in filter(None, tp.split("|")):
            tier_pairs[t] += 1
    examples = multi.sort_values("views", ascending=False).head(15)[["workflow_id", "name", "n_sites", "n_models", "models", "views"]]
    out["pooled"] = dict(
        top_models=top_models, top_open_models=top_open, top_providers=top_prov, top_pairs=pairs.most_common(20),
        tier_pairs=tier_pairs.most_common(), examples=examples.to_dict("records"),
        n_multi_model=int(len(multi)), n_router=int(prim["has_router"].sum()),
        distinct_open_models_self_host=len({s["model_norm"] for s in expl if s["provider_class"] == "self_host"}),
    )
    # HF overlap
    api_h = set(df[(df["corpus"] == "n8n_api")]["graph_hash"])
    hf = df[(df["corpus"] == "n8n_hf") & (~df["dup"])]
    if len(hf):
        out["hf_overlap"] = dict(n_hf_dedup=int(len(hf)), n_hf_in_api=int(hf["graph_hash"].isin(api_h).sum()),
                                 p_hf_in_api=rate(hf["graph_hash"].isin(api_h), len(hf)))
    fails = {}
    for f in PARSED.glob("*_failures.jsonl"):
        fails[f.stem] = sum(1 for _ in read_jsonl(f))
    out["parse_failures"] = fails
    # unmatched node types with model-ish names (n8n) for review
    return out


if __name__ == "__main__":
    wf, cs = load()
    by = collections.defaultdict(list)
    for s in cs:
        by[(s["corpus"], s["workflow_id"])].append(s)
    rows = [per_workflow(w, by.get((w["corpus"], w["workflow_id"]), [])) for w in wf]
    df = dedup(pd.DataFrame(rows))
    df.to_csv(PARSED / "workflows.csv", index=False)
    summary = summarise(df, cs)
    json.dump(summary, open(PARSED / "summary.json", "w"), indent=1, ensure_ascii=False, default=str)
    for c, d in summary["corpora"].items():
        log(f"{c:9s} all={d['n_all']:5d} llm={d['n_llm']:5d} multi-site={d['p_multi_site']} "
            f"multi-model={d['p_multi_model']} [{d['p_multi_model_lb']},{d['p_multi_model_ub']}] "
            f"vendors>=2={d['p_multi_vendor']} tiers>=2={d['p_multi_tier']} open>=2={d['p_multi_open']} self-host={d['p_any_self_host']}")
    if "hf_overlap" in summary:
        log("hf overlap", summary["hf_overlap"])
