#!/bin/bash
# Submit the plan on the A100-40GB slices: the prepare job, then one job per arm depending on it.
#   ./greatlakes-A100-40GB/submit_all.sh                      all 12 arms after a fresh prepare job
#   ./greatlakes-A100-40GB/submit_all.sh ours vanilla         a subset
#   PREP=12345678 ./greatlakes-A100-40GB/submit_all.sh ours   depend on an existing prepare job instead
#   PREP=none ./greatlakes-A100-40GB/submit_all.sh ours       no dependency (prepare already finished): resume a dead arm
#   RUN_ARM_OPTS="--only mix_L6" PREP=none ./greatlakes-A100-40GB/submit_all.sh cachescout
#                                                             extra benchmark/run_arm.py options for every arm
#
# The arm job is greatlakes/arm.sbatch, unchanged: the A100 settings reach it through the environment
# (sbatch --export=ALL, and greatlakes/config.sh keeps values that are already set), and the results
# directory is appended to run_arm's arguments, where the last --results wins.
# Resources come from greatlakes-A100-40GB/config.sh. Job ids are appended to greatlakes-A100-40GB/jobs.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh
source greatlakes-A100-40GB/config.sh

ARMS=${*:-"vanilla adapter kvonly relayout ours continuum cachescout kvflow continuum_relayout cachescout_relayout kvflow_relayout ours_noprefetch"}
COMMON=(--account="$GL_ACCOUNT" --partition="$GL_PARTITION" --gres="$GL_GRES"
        --cpus-per-task="$GL_CPUS" --mem="$GL_MEM" --export=ALL
        --output="$A100_DIR/slurm-%x-%j.out")
[[ -n "$GL_MAIL" ]] && COMMON+=(--mail-user="$GL_MAIL" --mail-type=END,FAIL)

PREP=${PREP:-}
if [[ -z "$PREP" ]]; then
  PREP=$(sbatch --parsable "${COMMON[@]}" --time="$GL_PREP_TIME" "$A100_DIR/prepare.sbatch")
  echo "prepare job $PREP" | tee -a "$A100_JOBS"
fi
DEP=()
[[ "$PREP" != none ]] && DEP=(--dependency=afterok:"$PREP")

# shellcheck disable=SC2086   # RUN_ARM_OPTS is meant to split into words
for ARM in $ARMS; do
  JID=$(sbatch --parsable "${COMMON[@]}" "${DEP[@]}" --time="$GL_TIME" --job-name="mortis-a100-$ARM" \
        greatlakes/arm.sbatch "$ARM" --results "$A100_RESULTS" ${RUN_ARM_OPTS:-})
  echo "$ARM job $JID" | tee -a "$A100_JOBS"
done
echo "queue: squeue -u $USER -o '%.10i %.24j %.8T %.10M %.9l %R'"
