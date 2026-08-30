#!/usr/bin/env bash
# Reproduce the survey end-to-end. Network needed for the fetch steps (idempotent; skips cached files).
set -euo pipefail
cd "$(dirname "$0")"
python fetch_n8n_api.py --index
for i in 0 1 2 3; do python fetch_n8n_api.py --details --sleep 0.2 --shard $i --nshards 4 & done; wait   # ~25 min; resumable
python fetch_hf_n8n.py
python fetch_dify.py
python fetch_github_flows.py
python parse_n8n.py api
python parse_n8n.py hf
python parse_dify.py
python parse_flowise.py
python parse_langflow.py
python normalise_models.py
python metrics.py
python figures.py
python spotcheck.py            # draws the sample; label data/spotcheck/sample.csv by hand, then: python spotcheck.py --score
python report.py
