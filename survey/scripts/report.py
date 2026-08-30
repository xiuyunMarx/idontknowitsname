"""Render REPORT.md from summary.json (numbers) + authored prose."""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, PARSED, ROOT, log  # noqa: E402


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def main():
    S = json.load(open(PARSED / "summary.json"))
    C = S["corpora"]
    P = S["pooled"]
    api = C["n8n_api"]
    score = None
    if (DATA / "spotcheck" / "score.json").exists():
        score = json.load(open(DATA / "spotcheck" / "score.json"))
    fetched = datetime.date.today().isoformat()
    names = {"n8n_api": "n8n official template library (AI category)", "dify": "Dify DSL collections", "flowise": "Flowise marketplace",
             "langflow": "Langflow starter projects", "n8n_hf": "n8n HF dumps (cross-check)"}

    def row(c):
        d = C[c]
        return (f"| {names[c]} | {d['n_all_raw']} / {d['n_all']} | {d['n_llm']} | {pct(d['p_multi_site'])} | "
                f"**{pct(d['p_multi_model'])}** [{pct(d['p_multi_model_lb'])}–{pct(d['p_multi_model_ub'])}] | "
                f"{pct(d['p_multi_model_of_multi_site'])} | {pct(d['p_multi_vendor'])} | {pct(d['p_multi_tier'])} | "
                f"{pct(d['p_any_open'])} | {pct(d['p_multi_open'])} | {pct(d['p_any_self_host'])} | "
                f"{d['sites_median']:.0f} / {d['sites_p90']:.0f} | {pct(d['default_site_rate'])} / {pct(d['unknown_site_rate'])} |")

    tbl = "\n".join(row(c) for c in ["n8n_api", "dify", "flowise", "n8n_hf"])
    lf = C["langflow"]
    top_models = "\n".join(f"| {m} | {n} |" for m, n in P["top_models"][:15])
    top_open = "\n".join(f"| {m} | {n} |" for m, n in P["top_open_models"][:10])
    top_pairs = "\n".join(f"| {m} | {n} |" for m, n in P["top_pairs"][:15])
    tier_pairs = ", ".join(f"{t}: {n}" for t, n in P["tier_pairs"])
    examples = "\n".join(
        f"| [{e['workflow_id']}](https://n8n.io/workflows/{e['workflow_id']}) | {e['name']} | {e['n_sites']} | {e['n_models']} | {e['models'].replace('|', ', ')} | {int(float(e['views'])) if e['views'] not in (None, '', 'None') else ''} |"
        for e in P["examples"][:12])
    hf = S.get("hf_overlap", {})
    dist = api["n_models_dist"]
    spot = ("not yet scored" if not score else
            f"{score['n_labelled']} workflows labelled by hand from the raw files; agreement with the parser on call-site count "
            f"{pct(score['site_count_agreement'])}, on distinct-model count {pct(score['model_count_agreement'])}, "
            f"on the ≥2-models decision {pct(score['multi_model_agreement'])}. The first labelling pass found three parser rules wrong "
            f"(vision calls through the OpenAI vendor node counted as image generation; unconnected model sub-nodes counted as invoked; "
            f"one site counted per attached model rather than per consumer when two models feed one agent as a fallback) — all fixed "
            f"before the final run; the numbers above come from the corrected parsers.")

    md = f"""# How often does one agent workflow use more than one model?

*Measured on public workflow-template corpora, fetched {fetched}. Every number below is in `data/parsed/summary.json`; regenerate with `scripts/run_all.sh`.*

## TL;DR

- In the **n8n official template library** (the largest public corpus of deployable agent workflows; {api['n_all']} AI-category templates after de-duplication, {api['n_llm']} of which invoke an LLM), **{pct(api['p_multi_model'])} of LLM workflows invoke ≥2 distinct models** (bounds {pct(api['p_multi_model_lb'])}–{pct(api['p_multi_model_ub'])} depending on how unresolvable model fields are counted). {pct(api['p_multi_site'])} have ≥2 LLM call sites; among those, **{pct(api['p_multi_model_of_multi_site'])}** use ≥2 models.
- The same measurement on **Dify** community/official DSL gives **{pct(C['dify']['p_multi_model'])}**, on the **Flowise** marketplace **{pct(C['flowise']['p_multi_model'])}**, and on the HF n8n dumps {pct(C['n8n_hf']['p_multi_model'])} (cross-check, all categories).
- What the mixes are: overwhelmingly a **small model paired with a large one** (size-tier pairs among multi-model n8n workflows: {tier_pairs}); the most common pairs are {"; ".join(m for m, _ in P['top_pairs'][:3])}. Multi-model workflows have a median of {api['sites_median_multi']:.0f} call sites vs {api['sites_median_single']:.0f} for single-model ones.
- Mixing across **vendors** is less common ({pct(api['p_multi_vendor'])} of n8n LLM workflows); mixing **≥2 open-weight models** is rare today ({pct(api['p_multi_open'])} n8n, {pct(C['dify']['p_multi_open'])} Dify) because templates default to hosted closed APIs; {pct(api['p_any_self_host'])} of n8n and {pct(C['dify']['p_any_self_host'])} of Dify LLM workflows use a self-host provider (Ollama / OpenAI-compatible endpoint).
- Explicit model routers exist as a first-class construct: n8n's `Model Selector` node appears in {P['n_router']} templates; Dify's `question-classifier` and Flowise's condition-agent play the same role.

## 1. Question and scoping

The serving-side question is whether a single agent *request* fans out over several models, so that a platform hosting the models must decide **which model to load next** for in-flight requests. We count, per workflow template, the number of **distinct model identifiers** among its LLM call sites (any vendor). Secondary columns split that by vendor, size tier, open-weight availability and self-host provider:

- the multi-model rate **regardless of provider** is the design-practice number (how often authors compose models);
- the **≥2 open-weight** and **self-host** columns are the population where the load-sequence problem is realised *today*;
- the **≥2 size tiers** column is the addressable population: a small+large pair on closed APIs (gpt-4o-mini + gpt-4.1, gemini-flash + gemini-pro) becomes a self-hosted two-model chain the moment the closed models are swapped for open equivalents of the same tiers — which is exactly what a platform serving open models offers.

## 2. Method

**Corpora.** Templates, not deployments (see limitations):

| corpus | what | size |
|---|---|---|
| n8n official template library | `api.n8n.io/api/templates/search?category=AI` index (8,185 templates) → full workflow JSON for the {api['n_all_raw']} that contain a LangChain node | primary |
| Dify DSL collections | official `tmpl.dify.ai` feed (28) + GitHub `svcvit/Awesome-Dify-Workflow` (46), `wwwzhouhui/dify-for-dsl` (92), `difyhub/workflows` (11), `Winson-030/dify-DSL` (3) | {C['dify']['n_all_raw']} raw / {C['dify']['n_all']} de-duplicated |
| Flowise marketplace | `FlowiseAI/Flowise` `packages/server/marketplaces/{{chatflows,agentflows,agentflowsv2}}` | {C['flowise']['n_all_raw']} |
| Langflow starter projects | `langflow-ai/langflow` `initial_setup/starter_projects` | {lf['n_all_raw']} (site counts only: exported templates carry no model ids) |
| n8n HF dumps | `mbakgun/n8nbuilder-n8n-workflows-dataset`, `ruh-ai/n8n-workflow-dataset`, `npv2k1/n8n-workflow` | {C['n8n_hf']['n_all_raw']} raw / {C['n8n_hf']['n_all']} de-duplicated; all categories; cross-check only |

**Call sites.** A call site is a node that invokes a model at runtime (n8n: `agent`, `chainLlm`, `informationExtractor`, `textClassifier`, vendor nodes such as `openAi`; Dify: `llm`, `agent`, `question-classifier`, `parameter-extractor`; Flowise: chains/agents fed by a chat-model node, or v2 agent/LLM nodes; Langflow: `Agent`/`LanguageModelComponent`). Embedding, reranker and image/audio nodes are recorded but excluded from the headline. In n8n the model is read from the LM sub-node wired in through an `ai_languageModel` connection (following `modelSelector` routers); model fields come as strings, resource-locator objects, `=`-prefixed literals, or `models/…` Gemini ids.

**Model resolution.** Each site is `explicit` (a literal model id), `default` (field omitted → node default, which for several n8n nodes changed over time without a version bump — see `data/node_defaults.csv`), `expression` (set at runtime) or `unknown` (no model wired). Ids are normalised by a regex table (`scripts/normalise_models.py`) to vendor / family / size / tier / open-weight; unmatched ids stay distinct raw strings (`data/parsed/unmatched_models.json`).

**Metrics.** Per workflow: `n_sites`, `n_models` (distinct explicit ids; a default site counts as the same model as an explicit id of the same provider if one exists, else as one `<provider>:default` model), with bounds `lb` (explicit ids only) and `ub` (every default/unknown site distinct). Denominator for the multi-model rate: LLM workflows with ≥1 attributable site. `n_tiers` uses xs ≤2B, s ≤9B or nano/mini/flash/haiku, m, l = pro/opus/sonnet/gpt-4o/gpt-4.1/>35B.

**De-duplication.** Structural hash over the multiset of node types/versions and typed edges (names, positions, ids and sticky notes dropped); one workflow kept per hash and corpus (highest view count first). Dify collections had {C['dify']['n_dup_removed']} duplicates (cn/en copies, cross-repo re-posts).

## 3. Results

![fig1](figures/fig1_prevalence.png)

| corpus | templates raw / dedup | LLM workflows | ≥2 call sites | **≥2 models** [lb–ub] | ≥2 models given ≥2 sites | ≥2 vendors | ≥2 size tiers | any open-weight | ≥2 open-weight | any self-host provider | sites median / p90 | default / unknown site rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{tbl}
| Langflow starter projects | {lf['n_all_raw']} / {lf['n_all']} | {lf['n_llm']} | {pct(lf['p_multi_site'])} | n/a (no model ids in exports) | | | | | | | {lf['sites_median']:.0f} / {lf['sites_p90']:.0f} | |

Distribution of distinct models per attributable n8n LLM workflow: {", ".join(f"{k if k != '4' else '≥4'}: {v}" for k, v in dist.items())}.

![fig2](figures/fig2_mix.png)

**What is mixed (n8n official, {P['n_multi_model']} multi-model workflows).** Size-tier pairs: {tier_pairs}. Top model pairs:

| pair | workflows |
|---|---|
{top_pairs}

Top models over all explicit n8n generation sites:

| model | sites |
|---|---|
{top_models}

Top open-weight models (n8n): {", ".join(f"{m} ({n})" for m, n in P['top_open_models'][:8]) or "none"}. Distinct open-weight models declared on self-host providers: {P['distinct_open_models_self_host']}.

**Named examples (n8n official, by views):**

| id | template | sites | models | model set | views |
|---|---|---|---|---|---|
{examples}

Other fixed points used to validate the parsers: Flowise "Supervisor Worker" (agentflow v2) = 4 sites / 3 models (gpt-4o-mini supervisor-workers, gemini-2.5-flash, gpt-4.1); Dify "Deep Researcher" = 19 LLM nodes / 2 models (gemini-2.0-flash-exp + deepseek-r1-distill-llama-8b); n8n 9247 "Smart chat routing between Gemini and GPT models" = `Model Selector` fanning gpt-4.1-nano / gemini-2.5-pro / gemini-2.0-flash into one agent.

## 4. The other population: single-model, multi-call workflows

{pct(api['p_multi_site'])} of n8n LLM workflows have ≥2 call sites and the p90 is {api['sites_p90']:.0f} sites (Dify: median {C['dify']['sites_median']:.0f}, p90 {C['dify']['sites_p90']:.0f}; Dify "Deep Researcher" alone has 19). Even when all sites share one model, the chain length is what a load scheduler's lookahead is worth: a request whose remaining chain calls the same model again (p90: {api['sites_p90']:.0f} sites in total) is a reason to keep that model resident. So the single-model majority is not irrelevant to load scheduling — only the *cross-model* reordering benefit is limited to the multi-model share.

## 5. Limitations

- **Templates ≠ deployments.** Counts reflect what authors publish, not traffic. Two biases in opposite directions: showcase templates over-represent multi-vendor demos; vendor marketplaces (Flowise, Langflow, n8n's own examples) and free-tier defaults over-represent a single OpenAI model.
- **Defaults.** {pct(api['default_site_rate'])} of n8n LLM workflows have at least one site whose model is the node default; the default for the Gemini and OpenAI chat-model nodes changed over time without a version bump, so those sites cannot be named exactly. The lb/ub bounds bracket the effect; the headline assumes a default site equals an explicit model of the same provider when one exists.
- **Open-weight share is a floor.** Templates are written against hosted APIs because that is what template authors have; the size-tier split (§1) is the better proxy for what a self-hosting platform would see.
- **Langflow** exports strip model choices entirely; only site counts are reported.
- **Parser verification.** {spot}

## 6. Reproduction

`bash scripts/run_all.sh` (fetch → parse → normalise → metrics → figures → report). Raw n8n detail crawl and HF dumps are gitignored and re-fetched by the scripts; everything else is committed. HF-dump overlap with the API crawl (structural hash): {hf.get('n_hf_in_api', 'n/a')} of {hf.get('n_hf_dedup', 'n/a')} ({pct(hf.get('p_hf_in_api'))}) — the HF dumps span all template categories, the crawl only the AI category. Parse failures: {S['parse_failures']}.
"""
    (ROOT / "REPORT.md").write_text(md, encoding="utf-8")
    log(f"wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
