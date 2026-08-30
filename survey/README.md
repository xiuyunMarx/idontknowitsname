# Survey: how often does one agent workflow use more than one model?

Quantitative measurement over public agent/workflow-template corpora (n8n official template
library, Dify DSL collections, Flowise marketplace, Langflow starters, plus HF n8n dumps as a
cross-check). The deliverable is `REPORT.md` with `figures/`.

Reproduce (Python ≥3.10 with `pandas`, `matplotlib`, `pyyaml`, `huggingface_hub`; network for the fetch steps):

```bash
bash scripts/run_all.sh
```

Or step by step: `scripts/fetch_*.py` → `scripts/parse_*.py` → `normalise_models.py` → `metrics.py`
→ `figures.py` → `spotcheck.py` (+ manual labels, `--score`) → `report.py`.

Layout:

- `data/raw/` cached downloads (`hf/` and the n8n detail crawl are gitignored; re-fetch with the scripts)
- `data/parsed/` one row per LLM call site (`callsites_norm.jsonl`), one row per workflow (`workflows.csv`), and every number in the report (`summary.json`)
- `data/models.csv`-style normalisation lives in `scripts/normalise_models.py` (regex table); node defaults in `data/node_defaults.csv`
- `data/spotcheck/` manual-verification sample and score
- `figures/` `fig1_prevalence`, `fig2_mix`
