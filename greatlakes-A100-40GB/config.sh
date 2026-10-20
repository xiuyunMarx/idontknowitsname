#!/bin/bash
# Settings for the A100-40GB run. Sourced after greatlakes/config.sh by every script in this directory;
# everything not listed here keeps the value from greatlakes/config.sh (account, conda env, modules, HF cache).
#
# The GPU is a 3g.40gb MIG slice of an A100 80GB PCIe (partition gpu_mig40: 2 nodes, 8 slices and 64 cores each).
# What this run needs beyond the A40 settings was worked out by probe_and_profile.bash:
#   - mem_fraction_static 0.89, or sglang clamps the 15 GiB device pool to about 13 GiB
#   - chunked prefill 4096 and CUDA graphs up to batch 32, the values sglang picks on the A40;
#     on the slice it reads the parent card's 80 GB and would pick 8192 / 256
#   - the profile lives in model/NVIDIA_A100_80GB_PCIe_MIG_3g_40gb/ (torch's name of the slice)

export GL_PARTITION=gpu_mig40
export GL_GRES=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
export GL_DEVICE_DIR=NVIDIA_A100_80GB_PCIe_MIG_3g_40gb
export GL_CPUS=8                     # 64 cores / 8 slices per node
export GL_MEM=96G                    # 1 TB / 8 slices = 128 GB per slice

export KV=61440                      # device KV pool, tokens: 15 GiB, same as the A40 run
export HOST=25                       # host KV tier, GB, same as the A40 run
export MEM_FRACTION=0.89             # read by model.Engine -> sglang mem_fraction_static
export CHUNKED_PREFILL_SIZE=4096     # read by model.Engine -> sglang chunked_prefill_size
export CUDA_GRAPH_MAX_BS=32          # read by model.Engine -> sglang cuda_graph_max_bs

# Everything of this run stays in this directory, apart from the A40 run in greatlakes/ and benchmark/results/.
export A100_DIR=greatlakes-A100-40GB
export A100_RESULTS=$A100_DIR/benchmark_results      # run_arm.py --results, on the cluster and here after a pull
export A100_JOBS=$A100_DIR/jobs.txt
