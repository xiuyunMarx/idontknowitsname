#!/bin/bash
# Probe the A100-40GB MIG slice on Great Lakes, then profile it.
#
#   ./greatlakes-A100-40GB/probe_and_profile.bash      run from the repo root on a login node
#
# Step 1: start the "ours" server with a 15 GiB device pool and a 25 GB host tier.
#         Pass = the server comes up and sglang keeps the full pool (it clamps a pool that
#         does not fit and logs the smaller number).
# Step 2: only if step 1 passed, measure the device profile (model/device_profiler.py:
#         prefill tps idle and under load, host->device promote tps). The server writes it
#         to model/<device name>/<model>.profile.json.
#
# Files written to this directory:
#   probe.out            everything this script prints
#   probe_server.log     server log of step 1
#   profile.out          server output of step 2

set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh

# ---- settings. config.sh describes the A40; these override it for the slice.
GL_PARTITION=gpu_mig40
GL_GRES=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
GL_CPUS=8
export KV=61440     # device pool in tokens: 15 GiB at 256 KiB/token
export HOST=25      # host tier in GB
PROFILE_KV=31830    # pool for step 2 only. The profiler pushes the 16k target to host with 1.3x the
                    # pool of junk; at 61440 that overflows the 25 GB host tier and promote_tps is wrong.
                    # The rates it measures do not depend on the pool size.
export MEM_FRACTION=${MEM_FRACTION:-0.89}   # share of GPU memory for weights + KV; sglang's default (~0.83) clamps KV to ~13 GiB here
export MODEL=${MODEL:-Qwen/Qwen3-32B-AWQ}

OUT=greatlakes-A100-40GB
PROFILE_NAME="$(echo "$MODEL" | sed 's#/#--#').profile.json"

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

# The server names the profile directory after torch.cuda.get_device_name, which on a MIG slice
# includes the slice ("NVIDIA A100 80GB PCIe MIG 3g.40gb") while nvidia-smi names the parent card.
# Ask torch, with the same substitutions as Engine.get_device_name.
DEVICE_DIR=$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))' | sed 's/[ .-]/_/g')
PROFILE=model/$DEVICE_DIR/$PROFILE_NAME
echo "profile dir: model/$DEVICE_DIR"

# ================================================================ step 1: probe
echo
echo "== step 1: start the ours server with KV=$KV HOST=${HOST}GB  $(date '+%T')"
export LOG=$OUT/probe_server.log
export PIDFILE=$OUT/server.pid

# --engine-log info makes sglang print its memory accounting (avail mem, max_total_num_tokens).
if ! ./start_server.bash ours --engine-log info; then
  echo "PROBE FAILED: the server did not come up, see $LOG"
  exit 1
fi

# The pool the engine ended up with: our ledger line (always printed) and sglang's own line.
# Every grep may legitimately miss, so none of them is allowed to end the script.
pool=$(grep -oE '\[ledger\] device=[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+' || true)
sglang_pool=$(grep -oE 'max_total_num_tokens=[0-9]+' "$LOG" | tail -1 | cut -d= -f2 || true)
host_line=$(grep -oE 'Allocating [0-9.]+ GB host memory' "$LOG" | tail -1 || true)
echo "memory lines from the server log:"
grep -E 'avail mem=|mem usage=|host memory|max_total_num_tokens|\[ledger\]' "$LOG" | tail -8 || true
./start_server.bash stop

if [[ "${pool:-0}" -ne "$KV" ]]; then
  echo "PROBE FAILED: asked for $KV tokens, the engine has ${pool:-none} (sglang: ${sglang_pool:-?})."
  exit 1
fi
echo "PROBE OK: device pool $pool tokens, ${host_line:-host tier size not logged}"

# ================================================================ step 2: profile
echo
echo "== step 2: device profile  $(date '+%T')"

# save_profile opens the path directly, so the directory has to exist first.
mkdir -p "$(dirname "$PROFILE")"

set +e
KV=$PROFILE_KV REPROFILE=1 ./start_server.bash profile 2>&1 | tee "$OUT/profile.out"
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 || ! -s "$PROFILE" ]]; then
  echo "PROFILE FAILED: rc=$status, expected $PROFILE, see $OUT/profile.out"
  exit 1
fi
echo "PROFILE OK: $PROFILE"
cat "$PROFILE"
