#!/bin/bash
# One arm, all its experiments, one server (benchmark/run_arm.py). Submit from the repo root:
#   sbatch [resources] --job-name=mortis-<arm> greatlakes/arm.sbatch <arm> [run_arm options]
# submit_all.sh adds the resources from config.sh. Re-submitting the same arm resumes: finished
# levels are skipped. Ten minutes before the wall time SLURM sends USR1, which the driver turns
# into a clean stop (sessions and server killed, the level in progress redone on resume).
#SBATCH --nodes=1 --ntasks=1
#SBATCH --signal=B:USR1@600
#SBATCH --output=greatlakes/slurm-%x-%j.out
set -uo pipefail
ARM=${1:?arm}; shift || true
cd "${GL_REPO:-$SLURM_SUBMIT_DIR}"
source greatlakes/config.sh
gl_load_modules
export PYTHON=$GL_PYTHON OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
gl_stage_model
# jac's persistent memory is a sqlite file per program under $JAC_DATA_PATH/.jac/data (default: the
# repo, i.e. the NFS home, where concurrent sessions on several nodes corrupt it): node-local, per job
export JAC_DATA_PATH=/tmp/$USER/jacdata-$SLURM_JOB_ID; mkdir -p "$JAC_DATA_PATH"
echo "arm=$ARM node=$(hostname) job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$PYTHON -m benchmark.run_arm "$ARM" --results benchmark/results --python "$PYTHON" "$@" &
DRIVER=$!
forward() { echo "$(date '+%F %T') signal $1 -> driver $DRIVER"; kill -"$1" "$DRIVER" 2>/dev/null; }
trap 'forward USR1' USR1
trap 'forward TERM' TERM INT
wait "$DRIVER"; RC=$?
if (( RC > 128 )); then wait "$DRIVER"; RC=$?; fi   # the first wait returns early when a trapped signal arrives
echo "arm=$ARM rc=$RC end=$(date '+%F %T')"
exit $RC
