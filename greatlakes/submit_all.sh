#!/bin/bash
# Submit the whole plan: the prepare job, then one job per arm depending on it.
#   ./greatlakes/submit_all.sh                      all 12 arms after a fresh prepare job
#   ./greatlakes/submit_all.sh ours vanilla         a subset
#   PREP=12345678 ./greatlakes/submit_all.sh ours   depend on an existing prepare job instead
#   PREP=none ./greatlakes/submit_all.sh ours       no dependency (prepare already finished): resume a dead arm
# Resources come from greatlakes/config.sh. Job ids are appended to greatlakes/jobs.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh
ARMS=${*:-"vanilla adapter kvonly relayout ours continuum cachescout kvflow continuum_relayout cachescout_relayout kvflow_relayout ours_noprefetch"}
COMMON=(--account="$GL_ACCOUNT" --partition="$GL_PARTITION" --gres="$GL_GRES" --cpus-per-task="$GL_CPUS" --mem="$GL_MEM" --export=ALL)
[[ -n "$GL_MAIL" ]] && COMMON+=(--mail-user="$GL_MAIL" --mail-type=END,FAIL)
mkdir -p greatlakes

PREP=${PREP:-}
if [[ -z "$PREP" ]]; then
  PREP=$(sbatch --parsable "${COMMON[@]}" --time="$GL_PREP_TIME" greatlakes/prepare.sbatch)
  echo "prepare job $PREP" | tee -a greatlakes/jobs.txt
fi
DEP=()
[[ "$PREP" != none ]] && DEP=(--dependency=afterok:"$PREP")

for ARM in $ARMS; do
  JID=$(sbatch --parsable "${COMMON[@]}" "${DEP[@]}" --time="$GL_TIME" --job-name="mortis-$ARM" greatlakes/arm.sbatch "$ARM")
  echo "$ARM job $JID" | tee -a greatlakes/jobs.txt
done
echo "queue: squeue -u $USER -o '%.10i %.14j %.8T %.10M %.9l %R'"
