#!/bin/bash
# The TP=4 case study in one allocation: the four arms one after another, each with its own server
# (benchmark/run_arm.py), mixed levels only. One job because the account's gres/gpu=5 cap fits only one
# 4-GPU job anyway, so one queue wait instead of four. Resubmitting resumes: finished levels are skipped.
#   sbatch [resources] greatlakes-A40-TP4/arms.sbatch [arm ...]      (submit_all.sh adds the resources)
#SBATCH --job-name=mortis-tp4
#SBATCH --nodes=1 --ntasks=1
#SBATCH --signal=B:USR1@600
#SBATCH --output=greatlakes-A40-TP4/slurm-%x-%j.out
set -uo pipefail
cd "${GL_REPO:-$SLURM_SUBMIT_DIR}"
source greatlakes/config.sh
source greatlakes-A40-TP4/config.sh
gl_load_modules
export PYTHON=$GL_PYTHON OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
gl_stage_model
export JAC_DATA_PATH=/tmp/$USER/jacdata-$SLURM_JOB_ID; mkdir -p "$JAC_DATA_PATH"   # node-local jac memory
export GUARD_PORT=${GUARD_PORT:-9101}
ARMS=${*:-"ours vanilla relayout ours_noprefetch"}
echo "arms=$ARMS node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "tp=$TP_SIZE kv=$KV host=${HOST}GB port=$GUARD_PORT results=$TP4_RESULTS"

# The TP=4 profile, if no smoke at TP=4 made it yet (same step as smoke.sbatch).
PROFILE="model/$GL_DEVICE_DIR/$(echo "${MODEL:-Qwen/Qwen3-32B-AWQ}" | sed 's#/#--#').profile.json"
if [[ ! -s "$PROFILE" ]]; then
  echo "== profile $(date '+%F %T')"
  mkdir -p "$(dirname "$PROFILE")"
  # KV=31830 for the profiler only: it pushes 1.3x the pool of junk to the host, which overflows 25 GB at 61440.
  KV=31830 REPROFILE=1 ./start_server.bash profile 2>&1 | tee "$TP4_DIR/profile.out"
  [[ -s "$PROFILE" ]] || { echo "PROFILE FAILED: no $PROFILE"; exit 1; }
fi
echo "profile: $(tr -d '\n ' < "$PROFILE")"

# USR1 (10 min before the wall time) and TERM go to the arm in progress, which stops cleanly; the loop then ends.
DRIVER=""; STOP=0
forward() { STOP=1; echo "$(date '+%F %T') signal $1 -> driver $DRIVER"; [[ -n "$DRIVER" ]] && kill -"$1" "$DRIVER" 2>/dev/null; }
trap 'forward USR1' USR1
trap 'forward TERM' TERM INT
RC=0
for ARM in $ARMS; do
  (( STOP )) && break
  echo "== arm $ARM $(date '+%F %T')"
  $PYTHON -m benchmark.run_arm "$ARM" --results "$TP4_RESULTS" --python "$PYTHON" --part mixed ${RUN_ARM_OPTS:-} &
  DRIVER=$!
  wait "$DRIVER"; rc=$?
  if (( rc > 128 )); then wait "$DRIVER"; rc=$?; fi   # the first wait returns early when a trapped signal arrives
  echo "== arm $ARM rc=$rc $(date '+%F %T')"
  (( rc != 0 )) && RC=$rc
done
echo "arms done rc=$RC end=$(date '+%F %T')"
exit $RC
