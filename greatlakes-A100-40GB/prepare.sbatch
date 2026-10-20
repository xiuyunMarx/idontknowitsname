#!/bin/bash
# End-to-end check on one A100 slice before the arm jobs (submit_all.sh chains them with afterok):
# one short fact_check level and a flush on vanilla and on cachescout, through the real jac -> server path.
# Refuses to run without the slice's profile (made by probe_and_profile.bash).
#
#   submit from the repo root:  sbatch [resources] greatlakes-A100-40GB/prepare.sbatch   (submit_all.sh does this)
#SBATCH --job-name=mortis-a100-prepare
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=greatlakes-A100-40GB/slurm-%x-%j.out
set -uo pipefail
cd "${GL_REPO:-$SLURM_SUBMIT_DIR}"
source greatlakes/config.sh
source greatlakes-A100-40GB/config.sh
gl_load_modules
export PYTHON=$GL_PYTHON OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
gl_stage_model
# jac's per-program sqlite memory: node-local and per job, never on the NFS home (see greatlakes/arm.sbatch)
export JAC_DATA_PATH=/tmp/$USER/jacdata-$SLURM_JOB_ID; mkdir -p "$JAC_DATA_PATH"
echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} python=$PYTHON jac=$JAC"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | head -2; nproc
export GUARD_PORT=${GUARD_PORT:-9000}   # own server port, the node is shared with arm jobs
echo "kv=$KV host=${HOST}GB mem_fraction=$MEM_FRACTION chunked_prefill=$CHUNKED_PREFILL_SIZE cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS port=$GUARD_PORT results=$A100_RESULTS"

PROFILE="model/$GL_DEVICE_DIR/$(echo "${MODEL:-Qwen/Qwen3-32B-AWQ}" | sed 's#/#--#').profile.json"
[[ -s "$PROFILE" ]] || { echo "no $PROFILE: run greatlakes-A100-40GB/probe_and_profile.bash first" >&2; exit 1; }
echo "profile: $(tr -d '\n ' < "$PROFILE")"

for ARM in vanilla cachescout; do
  echo "== smoke $ARM $(date '+%F %T')"
  $PYTHON -m benchmark.run_arm "$ARM" --smoke --results "$A100_RESULTS" --python "$PYTHON" || echo "smoke $ARM FAILED rc=$?"
done
echo "prepare done $(date '+%F %T')"
