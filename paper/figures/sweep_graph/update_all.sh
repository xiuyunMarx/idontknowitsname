#!/usr/bin/env bash
# Regenerate all sweep figures: the speedup panels of each workload,
# the shared legend and the hit-rate heatmap.
set -euo pipefail
cd "$(dirname "$0")"

# draw.py reads the CSVs and writes the PDFs of its own directory.
for workload in fact_check coding BFCL; do
    (cd "$workload" && python draw.py)
done
python draw_legend.py
python draw_hit_rate.py
