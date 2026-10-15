#!/bin/bash
# Great Lakes settings, sourced by every script in this directory. Edit this file only.

# ---- where things live on the cluster
export GL_REPO=${GL_REPO:-$HOME/idontknowitsname}            # the synced checkout (rsync target)
export GL_CONDA_ROOT=${GL_CONDA_ROOT:-$HOME/miniforge3}       # installed by setup_env.sh if absent
export GL_ENV=${GL_ENV:-mortis}                               # conda env with torch + the sglang fork
export GL_ZIG_DIR=${GL_ZIG_DIR:-$HOME/.local/zig}             # Zig 0.16.0, for building jac from jaseci/
export GL_ZIG_URL=${GL_ZIG_URL:-https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz}

# ---- SLURM: spgpu is the A40 48 GB partition, the one the account can use
export GL_ACCOUNT=${GL_ACCOUNT:-lingjia0}
export GL_PARTITION=${GL_PARTITION:-spgpu}
export GL_GRES=${GL_GRES:-gpu:a40:1}
export GL_CPUS=${GL_CPUS:-12}                                 # spgpu nodes: 32 cores per 8 GPUs; c=80 means 80 jac processes
export GL_MEM=${GL_MEM:-96G}                                  # host KV tier (25 GB pinned) + weights staging + jac sessions
# Server pools for the A40 (start_server.bash defaults are the old local-GPU values). sglang's own
# profile at mem_fraction_static=0.851 leaves 19.25 GiB for KV after the 18.27 GiB of AWQ weights, i.e. 78841
# tokens at 256 KiB/token; anything above is clamped to that profile.
export KV=${KV:-61440}                                        # device KV pool, tokens: 15 GiB, 4.25 GiB of the profile left idle so host/device stays near 1.5
export HOST=${HOST:-25}                                       # host KV tier, GB: must stay smaller than the live working set at the top of the grid so host eviction happens
export GL_TIME=${GL_TIME:-72:00:00}                           # partition max 14 d; a dead arm resumes
export GL_PREP_TIME=${GL_PREP_TIME:-03:00:00}
export GL_MAIL=${GL_MAIL:-xiaoyma@umich.edu}                                   # e.g. you@umich.edu for --mail-type=END,FAIL

# The HF cache goes on the account's scratch; logs and results stay in the repo (benchmark/results/).
export HF_HOME=${HF_HOME:-/scratch/lingjia_root/$GL_ACCOUNT/$USER/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}                    # setup_env.sh downloads; jobs may set 1

# ---- modules for the sglang JIT kernels (nvcc 12.x + a C++20 host compiler). The env also carries
# conda-forge gcc 12, which start_server.bash prefers when present.
export GL_MODULES=${GL_MODULES:-"cuda/12.8.1 gcc/13.2.0"}

# ---- derived
export GL_PYTHON=$GL_CONDA_ROOT/envs/$GL_ENV/bin/python
export PATH=$GL_ZIG_DIR:$GL_REPO/jaseci/jac/zig-out/bin:$PATH
export JAC=$GL_REPO/jaseci/jac/zig-out/bin/jac

# Loading safetensors from the GPFS scratch stalls (sglang mmaps the shards; the page faults crawl),
# so a job first copies the model to the node's local disk (sequential, minutes) and serves it from there.
gl_stage_model() {
  local m=${MODEL:-Qwen/Qwen3-32B-AWQ} local_home=/tmp/$USER/hf
  mkdir -p "$local_home/hub"
  echo "staging $m -> $local_home $(date '+%T')"
  # the model dir holds symlinks into the shared hub/blobs store; copy both, links stay relative
  rsync -a "$HF_HOME/hub/models--${m//\//--}" "$HF_HOME/hub/blobs" "$local_home/hub/"
  echo "staged $(du -sh "$local_home" | cut -f1) $(date '+%T')"
  export HF_HOME=$local_home HF_HUB_OFFLINE=1
}

gl_load_modules() {
  if command -v module >/dev/null 2>&1 && [[ -n "$GL_MODULES" ]]; then
    # shellcheck disable=SC2086
    module load $GL_MODULES || echo "WARNING: module load $GL_MODULES failed; check GL_MODULES in greatlakes/config.sh" >&2
  fi
}
