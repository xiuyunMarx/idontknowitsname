#!/bin/bash
# One-time environment build on a Great Lakes LOGIN node (needs internet; no GPU needed):
#   bash greatlakes/setup_env.sh
# Idempotent: every step checks whether it is already done. The GPU-dependent first start
# (JIT kernels, device profile) is greatlakes/prepare.sbatch.
#
# What it builds, following memory/build-forks-from-local-source: everything from the trees in
# this repo, never a prebuilt jac.
#   1. miniforge at $GL_CONDA_ROOT (if no conda), env $GL_ENV with python 3.12 + conda-forge gcc 12
#   2. torch 2.9.1 cu128, then the vendored sglang fork editable (pip install -e sglang/python)
#   3. Zig 0.16.0 at $GL_ZIG_DIR, jac built from jaseci/jac (zig build -Dno-ninja -Ddev), and the
#      byllm runtime deps installed into jac's materialized runtime (jac -m pip)
#   4. the model weights into $HF_HOME
set -euo pipefail
command -v sbatch >/dev/null || { echo "run this on a Great Lakes login node (no sbatch here)" >&2; exit 1; }
source "$(dirname "$0")/config.sh"
cd "$GL_REPO"
gl_load_modules

echo "== 1. conda env $GL_ENV"
if [[ ! -x "$GL_CONDA_ROOT/bin/conda" ]]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$GL_CONDA_ROOT"
fi
CONDA="$GL_CONDA_ROOT/bin/conda"
if [[ ! -x "$GL_PYTHON" ]]; then
  "$CONDA" create -y -n "$GL_ENV" python=3.12
fi
"$CONDA" install -y -n "$GL_ENV" -c conda-forge "gcc=12" "gxx=12"
PIP="$GL_PYTHON -m pip"

echo "== 2. torch + sglang fork"
$GL_PYTHON -c "import torch; assert torch.__version__.startswith('2.9.1')" 2>/dev/null || \
  $PIP install "torch==2.9.1" "torchvision==0.24.1" "torchaudio==2.9.1" --index-url https://download.pytorch.org/whl/cu128
# "import sglang" alone passes in the repo root (the sglang/ directory is a namespace package): test a submodule
$GL_PYTHON -c "import sglang.srt" 2>/dev/null || $PIP install -e sglang/python
$PIP install aiohttp huggingface_hub   # the guard server's HTTP front; the downloader
$GL_PYTHON - <<'EOF'
import torch, sglang.srt, aiohttp
print("torch", torch.__version__, "cuda", torch.version.cuda, "| sglang", sglang.srt.__file__)
EOF

echo "== 3. zig + jac"
if [[ ! -x "$GL_ZIG_DIR/zig" ]]; then
  mkdir -p "$GL_ZIG_DIR"
  curl -fsSL "$GL_ZIG_URL" | tar -xJ --strip-components=1 -C "$GL_ZIG_DIR"
fi
"$GL_ZIG_DIR/zig" version
# Zig writes build outputs with renameat2(RENAME_NOREPLACE), which the NFS home rejects (EINVAL), so the
# build runs on the login node's local /tmp. -Djaclang-dir makes the binary serve the compiler from
# the synced tree at run time (it must hold the LLVM shim), and -Dshim-bin bundles the synced prebuilt
# shim instead of fetching LLVM (3.4 GB) and linking it with the system C++ compiler.
SHIM=jaclang/compiler/passes/native/llvm/libjacllvm.so
if [[ ! -x "$JAC" ]]; then
  B=/tmp/$USER/jacbuild
  mkdir -p "$B"
  rsync -a --delete --exclude .zig-cache --exclude zig-out --exclude "$SHIM" jaseci/jac/ "$B/"
  cp jaseci/jac/$SHIM "/tmp/$USER/libjacllvm.so"
  # -Dtarget pins glibc so zig links its own CRT: the system crt1.o (glibc 2.28) references
  # __libc_csu_init, which zig's libc does not provide.
  ( cd "$B" && "$GL_ZIG_DIR/zig" build -Dno-ninja --global-cache-dir "/tmp/$USER/zig-global" \
      -Dtarget=x86_64-linux-gnu.2.28 -Djaclang-dir="$GL_REPO/jaseci/jac" -Dshim-bin="/tmp/$USER/libjacllvm.so" )
  mkdir -p "$(dirname "$JAC")" && cp "$B/zig-out/bin/jac" "$JAC"
fi
"$JAC" --version
# byllm's connector deps plus what the agents' tools use; they live in jac's own runtime
"$JAC" -m pip install litellm httpx loguru pillow pytest requests
mkdir -p "$HOME/.local/bin"; ln -sfn "$JAC" "$HOME/.local/bin/jac"

echo "== 4. model weights -> $HF_HOME"
mkdir -p "$HF_HOME"
$GL_PYTHON -m huggingface_hub.commands.huggingface_cli download "${MODEL:-Qwen/Qwen3-32B-AWQ}" 2>/dev/null || \
  $GL_PYTHON -c "from huggingface_hub import snapshot_download; print(snapshot_download('${MODEL:-Qwen/Qwen3-32B-AWQ}'))"
du -sh "$HF_HOME"

echo "== 5. static analysis of the four agents (warms static_analysis/.cache; runs under jac, whose runtime has jaclang)"
for p in fact_check coding_agent BFCL_agent doc_analysis; do
  "$JAC" -m static_analysis.agent_launcher --no-register --dry-run "benchmark/applications/$p.jac" 2>&1 | tail -1
done
echo "== 6. model profile on an A40 (prefill tps idle / under load, host->device promote tps) -> model/*.profile.json"
# The login node has no GPU: this step takes one A40 through a blocking srun (waits in the queue like a job).
# The first start also JIT-compiles sglang's kernels, so the arm jobs never do. REPROFILE=1 re-measures an existing profile.
PROFILE="model/$(echo "${MODEL:-Qwen/Qwen3-32B-AWQ}" | sed 's#/#--#').profile.json"
if [[ -s "$PROFILE" && "${REPROFILE:-0}" != 1 ]]; then
  echo "$PROFILE exists (REPROFILE=1 to re-measure):"; cat "$PROFILE"
else
  srun --account="$GL_ACCOUNT" --partition="$GL_PARTITION" --gres="$GL_GRES" --cpus-per-task="$GL_CPUS" \
       --mem="$GL_MEM" --time=01:30:00 --job-name=mortis-profile --export=ALL \
       bash -c 'source greatlakes/config.sh; gl_load_modules; export PYTHON=$GL_PYTHON
                echo "profiling on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
                gl_stage_model
                REPROFILE=${REPROFILE:-0} ./start_server.bash profile' 2>&1 | tee greatlakes/profile.out
  [[ -s "$PROFILE" ]] || { echo "profiling failed: no $PROFILE; see greatlakes/profile.out" >&2; exit 1; }
  echo "profile written:"; cat "$PROFILE"
fi
echo "setup done. Next: ./greatlakes/submit_all.sh"
