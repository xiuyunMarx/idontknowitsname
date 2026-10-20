#!/bin/bash
# Settings for the A100-40GB run (one 3g.40gb MIG slice, partition gpu_mig40: 8 slices and 64 cores per node).
# Sourced after greatlakes/config.sh; anything not set here keeps that file's value.

export GL_PARTITION=gpu_mig40
export GL_GRES=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
export GL_DEVICE_DIR=NVIDIA_A100_80GB_PCIe_MIG_3g_40gb   # torch's name of the slice: model/<it>/ holds the profile
export GL_CPUS=8
export GL_MEM=96G

export KV=61440                      # device KV pool, tokens (15 GiB)
export HOST=25                       # host KV tier, GB
export MEM_FRACTION=0.89             # sglang mem_fraction_static; its default clamps the pool to ~13 GiB here
export CHUNKED_PREFILL_SIZE=4096     # sglang chunked_prefill_size, the A40 value (the slice reports 80 GB and would pick 8192)
export CUDA_GRAPH_MAX_BS=32          # sglang cuda_graph_max_bs, the A40 value (the slice would pick 256)

# This run's files stay in this directory.
export A100_DIR=greatlakes-A100-40GB
export A100_RESULTS=$A100_DIR/benchmark_results      # run_arm.py --results, on the cluster and here after a pull
export A100_JOBS=$A100_DIR/jobs.txt
