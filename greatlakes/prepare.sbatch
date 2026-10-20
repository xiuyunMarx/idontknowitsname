#!/bin/bash
# The end-to-end check before the arm jobs (submit_all.sh chains them with afterok): one short
# fact_check level and a flush on vanilla and on cachescout (the baseline controllers with the new
# flush handler), through the real jac -> server path. The model profile (device rates, H2D
# bandwidth) and the JIT kernels were made by setup_env.sh step 6; this job refuses to run
# without the profile, since twelve arm jobs starting together would each measure their own.
#
#   submit from the repo root:  sbatch [resources] greatlakes/prepare.sbatch   (submit_all.sh does this)
#SBATCH --job-name=mortis-prepare
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=greatlakes/slurm-%x-%j.out
set -uo pipefail
cd "${GL_REPO:-$SLURM_SUBMIT_DIR}"
source greatlakes/config.sh
gl_load_modules
export PYTHON=$GL_PYTHON OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
gl_stage_model
# jac's persistent memory is a sqlite file per program under $JAC_DATA_PATH/.jac/data (default: the
# repo, i.e. the NFS home, where concurrent sessions on several nodes corrupt it): node-local, per job
export JAC_DATA_PATH=/tmp/$USER/jacdata-$SLURM_JOB_ID; mkdir -p "$JAC_DATA_PATH"
echo "node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} python=$PYTHON jac=$JAC"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | head -2; nproc
PROFILE="model/$GL_DEVICE_DIR/$(echo "${MODEL:-Qwen/Qwen3-32B-AWQ}" | sed 's#/#--#').profile.json"
[[ -s "$PROFILE" ]] || { echo "no $PROFILE: run step 6 of greatlakes/setup_env.sh first (the model profile)" >&2; exit 1; }
echo "profile: $(cat "$PROFILE" | tr -d '\n ')"

for ARM in vanilla cachescout; do
  echo "== smoke $ARM $(date '+%F %T')"
  $PYTHON -m benchmark.run_arm "$ARM" --smoke --results benchmark/results --python "$PYTHON" || echo "smoke $ARM FAILED rc=$?"
done
ls -la model/*/*.profile.json
echo "prepare done $(date '+%F %T')"
