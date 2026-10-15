#!/bin/bash
# Pull benchmark/results/ (every level's logs and summaries) and the SLURM outputs back here.
#   ./greatlakes/pull_results.sh <uniqname> [--dry-run]
# Then: python -m benchmark.summarize
set -euo pipefail
cd "$(dirname "$0")/.."
USER_=${1:?usage: $0 <uniqname> [--dry-run]}; shift || true
HOST=${GL_HOST:-greatlakes.arc-ts.umich.edu}
SRC=${GL_REPO:-"~/idontknowitsname"}
rsync -avz --info=progress2 "$@" "$USER_@$HOST:$SRC/benchmark/results/" benchmark/results/
rsync -avz "$@" "$USER_@$HOST:$SRC/greatlakes/slurm-*.out" "$USER_@$HOST:$SRC/greatlakes/jobs.txt" greatlakes/ || true
rsync -avz "$@" "$USER_@$HOST:$SRC/model/*.profile.json" model/ || true
