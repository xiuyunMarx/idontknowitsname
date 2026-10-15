#!/bin/bash
# Push the repo to Great Lakes with rsync, including the git-ignored search caches
# (benchmark/applications/bench_data/HoVer/wiki_cache, .../BFCL/web_cache): the agents read tool
# results from them, so no compute node ever queries Wikipedia and gets rate-limited.
#
#   ./greatlakes/sync_to_greatlakes.sh <uniqname> [--dry-run]
#   GL_HOST=greatlakes-xfer.arc-ts.umich.edu ./greatlakes/sync_to_greatlakes.sh <uniqname>
#
# Not sent: build caches (jaseci/jac/.zig-cache, .llvm-build, .pbs-build, zig-out: jac is rebuilt
# there by setup_env.sh), bytecode, the paper, and benchmark/results (pull those back with
# pull_results.sh). .git is sent so the jobs can record provenance. The 1.2 GB LLVM shim
# (jaclang/compiler/passes/native/llvm/libjacllvm.so) is sent: `zig build -Ddev` needs it and
# would otherwise require `zig build fetch-llvm` plus a C++ toolchain on the login node.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_=${1:?usage: $0 <uniqname> [--dry-run]}; shift || true
HOST=${GL_HOST:-greatlakes.arc-ts.umich.edu}
DEST=${GL_REPO:-"~/idontknowitsname"}

for d in benchmark/applications/bench_data/HoVer/wiki_cache benchmark/applications/bench_data/BFCL/web_cache; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  (( n > 0 )) || { echo "$d is empty: warm it first (benchmark/applications/warm_web_cache.py)" >&2; exit 1; }
  echo "$d: $n files"
done

rsync -avz --info=progress2 "$@" \
  --exclude '.zig-cache/' --exclude '.llvm-build/' --exclude '.pbs-build/' --exclude 'zig-out/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.jac/' --exclude '*.jir' \
  --exclude 'static_analysis/.cache/' --exclude 'benchmark/results/' --exclude 'paper/' \
  --exclude '.venv/' --exclude '*.egg-info/' --exclude 'sglang/python/build/' \
  ./ "$USER_@$HOST:$DEST/"
echo "synced to $USER_@$HOST:$DEST"
