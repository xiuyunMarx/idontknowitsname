#!/bin/bash
# Probe the A100-40GB MIG slice on Great Lakes, then profile it.
#
#   ./greatlakes-A100-40GB/probe_and_profile.bash      run from the repo root on a login node
#
# Step 1: start the "ours" server with a 15 GiB device pool and a 25 GB host tier.
#         Pass = the server comes up and sglang keeps the full pool (it clamps a pool that
#         does not fit and logs the smaller number).
# Step 2: only if step 1 passed, measure the device profile (model/device_profiler.py:
#         prefill tps idle and under load, host->device promote tps) and save it here.
#         The A40 profile in model/ is moved aside while the server runs and put back after.
#
# Files written to this directory:
#   probe.out            everything this script prints
#   probe_server.log     server log of step 1
#   profile.out          server output of step 2
#   <model>.profile.json the A100 profile

set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh

# ---- settings. config.sh describes the A40; these override it for the slice.
GL_PARTITION=gpu_mig40
GL_GRES=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
GL_CPUS=8
export KV=61440     # device pool in tokens: 15 GiB at 256 KiB/token
export HOST=25      # host tier in GB
export MODEL=${MODEL:-Qwen/Qwen3-32B-AWQ}

OUT=greatlakes-A100-40GB
PROFILE_NAME="$(echo "$MODEL" | sed 's#/#--#').profile.json"
A40_PROFILE=model/$PROFILE_NAME
A40_BACKUP=$OUT/$PROFILE_NAME.a40

# ---- on the login node: take one slice and run this script on it
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  srun --account="$GL_ACCOUNT" --partition="$GL_PARTITION" --gres="$GL_GRES" \
       --cpus-per-task="$GL_CPUS" --mem="$GL_MEM" --time=01:30:00 \
       --job-name=mortis-probe-a100 --export=ALL \
       bash "$0" 2>&1 | tee "$OUT/probe.out"
  exit "${PIPESTATUS[0]}"
fi

# ---- on the compute node
gl_load_modules
export PYTHON=$GL_PYTHON OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
gl_stage_model
echo "node=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "model=$MODEL kv=$KV tokens host=${HOST}GB"

# ================================================================ step 1: probe
echo
echo "== step 1: start the ours server with KV=$KV HOST=${HOST}GB  $(date '+%T')"
export LOG=$OUT/probe_server.log
export PIDFILE=$OUT/server.pid

if ! ./start_server.bash ours; then
  echo "PROBE FAILED: the server did not come up, see $LOG"
  exit 1
fi

pool=$(grep -oE 'max_total_num_tokens=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
host_line=$(grep -oE 'Allocating [0-9.]+ GB host memory' "$LOG" | tail -1)
echo "memory lines from the server log:"
grep -E 'avail mem=|mem usage=|host memory|max_total_num_tokens' "$LOG" | tail -6
./start_server.bash stop

if [[ "${pool:-0}" -ne "$KV" ]]; then
  echo "PROBE FAILED: asked for $KV tokens, sglang kept ${pool:-none}."
  echo "15 GiB does not fit under sglang's default mem_fraction_static on this slice."
  exit 1
fi
echo "PROBE OK: device pool $pool tokens, $host_line"

# ================================================================ step 2: profile
echo
echo "== step 2: device profile  $(date '+%T')"

# The server writes model/<model>.profile.json. Keep the A40 one out of the way and restore
# it whatever happens, including a SLURM kill.
restore_a40_profile() {
  if [[ -f "$A40_BACKUP" ]]; then
    mv -f "$A40_BACKUP" "$A40_PROFILE"
  fi
}
trap restore_a40_profile EXIT
if [[ -f "$A40_PROFILE" ]]; then
  mv "$A40_PROFILE" "$A40_BACKUP"
fi

set +e
./start_server.bash profile 2>&1 | tee "$OUT/profile.out"
status=${PIPESTATUS[0]}
set -e

if [[ -f "$A40_PROFILE" ]]; then
  mv "$A40_PROFILE" "$OUT/$PROFILE_NAME"
fi

if [[ $status -ne 0 || ! -s "$OUT/$PROFILE_NAME" ]]; then
  echo "PROFILE FAILED: rc=$status, see $OUT/profile.out"
  exit 1
fi
echo "PROFILE OK: $OUT/$PROFILE_NAME"
cat "$OUT/$PROFILE_NAME"
