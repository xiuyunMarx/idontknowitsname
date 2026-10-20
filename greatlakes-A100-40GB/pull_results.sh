#!/bin/bash
# Pull the A100 run back here: greatlakes-A100-40GB/benchmark_results/ (every level's logs and summaries),
# the SLURM outputs and jobs.txt of this directory, and the slice's profile.
#   ./greatlakes-A100-40GB/pull_results.sh [uniqname] [--dry-run]
# Goes through the `gl` ssh alias by default so the ControlMaster socket is reused and Duo is asked
# once, not per rsync. Set GL_HOST to a full hostname (then uniqname is required).
# Only copies from the cluster to here; nothing is sent, nothing on the cluster is touched.
#
# The cluster writes every level flat under benchmark_results/<arm>/; here they land as
#   <arm>/per-agent/{BFCL,coding,fact_check,finance}/sweep_<program>_c<N>/
#   <arm>/mixed/mix_L<N>/
# the layout benchmark/results/generate_table.py reads (copy it into benchmark_results/ to build the tables).
set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh
source greatlakes-A100-40GB/config.sh

USER_=""; case ${1:-} in ""|--*) ;; *) USER_=$1; shift ;; esac
HOST_=${GL_HOST:-gl}
case $HOST_ in
    *.*) TARGET="${USER_:?usage: $0 <uniqname> [--dry-run] (uniqname required with a full GL_HOST)}@$HOST_" ;;
    *)   TARGET=$HOST_ ;;   # ssh alias: user comes from ~/.ssh/config
esac
# The checkout on the cluster. Not GL_REPO: config.sh builds that from $HOME, which here is the local home.
SRC=${GL_REMOTE_REPO:-"~/idontknowitsname"}
REMOTE="$TARGET:$SRC/$A100_RESULTS"

for arm in $(ssh "$TARGET" "ls $SRC/$A100_RESULTS" 2>/dev/null); do
    mkdir -p "$A100_RESULTS/$arm"
    # the arm's own files: progress.log, plan.json, server.<n>.log, requests.jsonl
    rsync -az --info=name1 "$@" --exclude='*/' "$REMOTE/$arm/" "$A100_RESULTS/$arm/"
    # the levels, sorted into per-agent/<agent>/ and mixed/
    for map in "mix_L*:mixed" "sweep_BFCL_agent_c*:per-agent/BFCL" "sweep_coding_agent_c*:per-agent/coding" \
               "sweep_fact_check_c*:per-agent/fact_check" "sweep_doc_analysis_c*:per-agent/finance" \
               "smoke_*:smoke"; do
        pat=${map%%:*}; sub=${map##*:}
        mkdir -p "$A100_RESULTS/$arm/$sub"
        rsync -az --info=name1 "$@" --include="$pat/" --include="$pat/**" --exclude='*' \
              "$REMOTE/$arm/" "$A100_RESULTS/$arm/$sub/"
    done
done
rsync -az --info=name1 "$@" "$TARGET:$SRC/$A100_DIR/slurm-*.out" "$TARGET:$SRC/$A100_JOBS" "$A100_DIR/" || true
rsync -az --info=name1 "$@" "$TARGET:$SRC/model/$GL_DEVICE_DIR/" "model/$GL_DEVICE_DIR/" || true
