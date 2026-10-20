#!/bin/bash
# Pull benchmark/results/ (every level's logs and summaries) and the SLURM outputs back here.
#   ./greatlakes/pull_results.sh [uniqname] [--dry-run]
# Goes through the `gl` ssh alias by default so the ControlMaster socket is reused and Duo
# is asked once, not per rsync. Set GL_HOST to a full hostname (then uniqname is required).
# The cluster writes every level flat under benchmark/results/<arm>/; here they land as
#   <arm>/per-agent/{BFCL,coding,fact_check,finance}/sweep_<program>_c<N>/
#   <arm>/mixed/mix_L<N>/
# Then: python -m benchmark.summarize
set -euo pipefail
cd "$(dirname "$0")/.."
USER_=""; case ${1:-} in ""|--*) ;; *) USER_=$1; shift ;; esac
HOST=${GL_HOST:-gl}
case $HOST in
    *.*) TARGET="${USER_:?usage: $0 <uniqname> [--dry-run] (uniqname required with a full GL_HOST)}@$HOST" ;;
    *)   TARGET=$HOST ;;   # ssh alias: user comes from ~/.ssh/config
esac
SRC=${GL_REPO:-"~/idontknowitsname"}
REMOTE="$TARGET:$SRC/benchmark/results"

for arm in $(ssh "$TARGET" "ls $SRC/benchmark/results"); do
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
rsync -az --info=name1 "$@" "$TARGET:$SRC/greatlakes/slurm-*.out" "$TARGET:$SRC/greatlakes/jobs.txt" greatlakes/ || true
rsync -az --info=name1 "$@" --include='*/' --include='*.profile.json' --exclude='*' "$TARGET:$SRC/model/" model/ || true
