#!/bin/bash
# Submit the TP=4 case study as one job (greatlakes-A40-TP4/arms.sbatch): four arms in sequence, mixed levels.
#   ./greatlakes-A40-TP4/submit_all.sh                     ours vanilla relayout ours_noprefetch, in one job
#   ./greatlakes-A40-TP4/submit_all.sh ours vanilla        a subset; resubmitting resumes (finished levels skipped)
#   RUN_ARM_OPTS="--only mix_L6" ./greatlakes-A40-TP4/submit_all.sh ours   extra benchmark/run_arm.py options
set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh
source greatlakes-A40-TP4/config.sh

COMMON=(--account="$GL_ACCOUNT" --partition="$GL_PARTITION" --gres="$GL_GRES"
        --cpus-per-task="$GL_CPUS" --mem="$GL_MEM" --nodes=1
        --output="$TP4_DIR/slurm-%x-%j.out" --export=ALL,GUARD_PORT=9101)
[[ -n "$GL_MAIL" ]] && COMMON+=(--mail-user="$GL_MAIL" --mail-type=END,FAIL)

JID=$(sbatch --parsable "${COMMON[@]}" --time="$TP4_TIME" greatlakes-A40-TP4/arms.sbatch "$@")
echo "arms job $JID: ${*:-ours vanilla relayout ours_noprefetch}" | tee -a "$TP4_JOBS"
echo "queue: squeue -u $USER -o '%.10i %.24j %.8T %.10M %.9l %R'"
