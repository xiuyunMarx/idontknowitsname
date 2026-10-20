#!/bin/bash
# Submit the plan on the A100-40GB slices: the prepare job, then one job per arm depending on it.
#   ./greatlakes-A100-40GB/submit_all.sh                      all 12 arms after a fresh prepare job
#   ./greatlakes-A100-40GB/submit_all.sh ours vanilla         a subset
#   PREP=12345678 ./greatlakes-A100-40GB/submit_all.sh ours   depend on an existing prepare job instead
#   PREP=none ./greatlakes-A100-40GB/submit_all.sh ours       no dependency (prepare already finished): resume a dead arm
#   RUN_ARM_OPTS="--only mix_L6" PREP=none ./greatlakes-A100-40GB/submit_all.sh cachescout
#                                                             extra benchmark/run_arm.py options for every arm
#
# Uses greatlakes/arm.sbatch unchanged: the A100 settings go in through the environment (--export=ALL) and
# the results directory is appended to run_arm's arguments (last --results wins). Job ids go to jobs.txt.
# Arms share nodes (8 slices each), so every arm gets its own server port via GUARD_PORT; prepare uses 9000.
set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh
source greatlakes-A100-40GB/config.sh

ARMS=${*:-"vanilla adapter kvonly relayout ours continuum cachescout kvflow continuum_relayout cachescout_relayout kvflow_relayout ours_noprefetch"}
declare -A PORT=([vanilla]=9001 [adapter]=9002 [kvonly]=9003 [relayout]=9004 [ours]=9005 [continuum]=9006
                 [cachescout]=9007 [kvflow]=9008 [continuum_relayout]=9009 [cachescout_relayout]=9010
                 [kvflow_relayout]=9011 [ours_noprefetch]=9012)
COMMON=(--account="$GL_ACCOUNT" --partition="$GL_PARTITION" --gres="$GL_GRES"
        --cpus-per-task="$GL_CPUS" --mem="$GL_MEM"
        --output="$A100_DIR/slurm-%x-%j.out")
[[ -n "$GL_MAIL" ]] && COMMON+=(--mail-user="$GL_MAIL" --mail-type=END,FAIL)

PREP=${PREP:-}
if [[ -z "$PREP" ]]; then
  PREP=$(sbatch --parsable "${COMMON[@]}" --export=ALL,GUARD_PORT=9000 --time="$GL_PREP_TIME" "$A100_DIR/prepare.sbatch")
  echo "prepare job $PREP" | tee -a "$A100_JOBS"
fi
DEP=()
[[ "$PREP" != none ]] && DEP=(--dependency=afterok:"$PREP")

# shellcheck disable=SC2086   # RUN_ARM_OPTS is meant to split into words
for ARM in $ARMS; do
  [[ -n "${PORT[$ARM]:-}" ]] || { echo "unknown arm: $ARM" >&2; exit 2; }
  JID=$(sbatch --parsable "${COMMON[@]}" "${DEP[@]}" --export=ALL,GUARD_PORT="${PORT[$ARM]}" \
        --time="$GL_TIME" --job-name="mortis-a100-$ARM" \
        greatlakes/arm.sbatch "$ARM" --results "$A100_RESULTS" ${RUN_ARM_OPTS:-})
  echo "$ARM job $JID port ${PORT[$ARM]}" | tee -a "$A100_JOBS"
done
echo "queue: squeue -u $USER -o '%.10i %.24j %.8T %.10M %.9l %R'"
