#!/bin/bash
# Pull benchmark/results/ (every level's logs and summaries) and the SLURM outputs back here.
#   ./greatlakes/pull_results.sh <uniqname> [--dry-run]
# The cluster writes every level flat under benchmark/results/<arm>/; here they land as
#   <arm>/per-agent/{BFCL,coding,fact_check,finance}/sweep_<program>_c<N>/
#   <arm>/mixed/mix_L<N>/
# Then: python -m benchmark.summarize
set -euo pipefail
cd "$(dirname "$0")/.."
USER_=${1:?usage: $0 <uniqname> [--dry-run]}; shift || true
HOST=${GL_HOST:-greatlakes.arc-ts.umich.edu}
SRC=${GL_REPO:-"~/idontknowitsname"}
REMOTE="$USER_@$HOST:$SRC/benchmark/results"

for arm in $(ssh "$USER_@$HOST" "ls $SRC/benchmark/results"); do
    [ -d "benchmark/results/$arm" ] || mkdir -p "benchmark/results/$arm"
    rsync -az --info=name1 "$@" --exclude='*/' "$REMOTE/$arm/" "benchmark/results/$arm/"
    for map in "mix_L*:mixed" "sweep_BFCL_agent_c*:per-agent/BFCL" "sweep_coding_agent_c*:per-agent/coding" \
               "sweep_fact_check_c*:per-agent/fact_check" "sweep_doc_analysis_c*:per-agent/finance"; do
        pat=${map%%:*}; sub=${map##*:}
        mkdir -p "benchmark/results/$arm/$sub"
        rsync -az --info=name1 "$@" --include="$pat/" --include="$pat/**" --exclude='*' \
              "$REMOTE/$arm/" "benchmark/results/$arm/$sub/"
    done
done
rsync -az --info=name1 "$@" "$USER_@$HOST:$SRC/greatlakes/slurm-*.out" "$USER_@$HOST:$SRC/greatlakes/jobs.txt" greatlakes/ || true
rsync -az --info=name1 "$@" "$USER_@$HOST:$SRC/model/*.profile.json" model/ || true
