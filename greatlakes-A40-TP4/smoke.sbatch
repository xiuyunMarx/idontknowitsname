#!/bin/bash
# TP=4 check on one spgpu node: measure the TP=4 profile if missing, then one short fact_check level
# and a flush on vanilla and on ours through the real jac -> server path.
#
#   submit from the repo root:  sbatch --account=... --partition=spgpu --gres=gpu:a40:4 --cpus-per-task=16 --mem=160G \
#                                      --time=02:00:00 --export=ALL,GUARD_PORT=9100 greatlakes-A40-TP4/smoke.sbatch
#SBATCH --job-name=mortis-tp4-smoke
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=greatlakes-A40-TP4/slurm-%x-%j.out
set -uo pipefail
cd "${GL_REPO:-$SLURM_SUBMIT_DIR}"
source greatlakes/config.sh
source greatlakes-A40-TP4/config.sh
gl_load_modules
export PYTHON=$GL_PYTHON OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
gl_stage_model
export JAC_DATA_PATH=/tmp/$USER/jacdata-$SLURM_JOB_ID; mkdir -p "$JAC_DATA_PATH"   # node-local jac memory
export GUARD_PORT=${GUARD_PORT:-9100}
echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} python=$PYTHON"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m | head -8        # how the four GPUs are linked (NVLink bridge or PCIe)
echo "tp=$TP_SIZE kv=$KV host=${HOST}GB port=$GUARD_PORT results=$TP4_RESULTS"

PROFILE="model/$GL_DEVICE_DIR/$(echo "${MODEL:-Qwen/Qwen3-32B-AWQ}" | sed 's#/#--#').profile.json"
if [[ ! -s "$PROFILE" ]]; then
  echo "== profile $(date '+%F %T')"
  mkdir -p "$(dirname "$PROFILE")"
  # KV=31830 for the profiler only: it pushes 1.3x the pool of junk to the host, which overflows 25 GB at 61440.
  KV=31830 REPROFILE=1 ./start_server.bash profile 2>&1 | tee "$TP4_DIR/profile.out"
  [[ -s "$PROFILE" ]] || { echo "PROFILE FAILED: no $PROFILE"; exit 1; }
fi
echo "profile: $(tr -d '\n ' < "$PROFILE")"

for ARM in vanilla ours; do
  echo "== smoke $ARM $(date '+%F %T')"
  $PYTHON -m benchmark.run_arm "$ARM" --smoke --results "$TP4_RESULTS" --python "$PYTHON" || echo "smoke $ARM FAILED rc=$?"
done
echo "smoke done $(date '+%F %T')"
