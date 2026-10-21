#!/bin/bash
# Settings for the A40 TP=4 case study (4 A40 on one spgpu node, one server with tensor parallelism 4).
# Sourced after greatlakes/config.sh; anything not set here keeps that file's value.

export GL_GRES=gpu:a40:4
export GL_CPUS=16                    # 32 cores per 8-GPU node: the fair share of 4 GPUs
export GL_MEM=160G                   # 25 GB pinned host tier + weights staging + jac sessions
export TP_SIZE=${TP_SIZE:-4}         # TP_SIZE=2 in --export: a 2-GPU smoke of the same code path (2-GPU holes are common)
export GL_DEVICE_DIR=NVIDIA_A40_tp$TP_SIZE   # model/<it>/ holds that TP degree's profile (Engine.profile_path)
export KV=61440                      # device KV pool, tokens (15 GiB in total; each rank holds 1/TP of every token)
export HOST=25                       # host KV tier, GB in total (model.py gives sglang HOST/TP per rank)

# This run's files stay in this directory.
export TP4_DIR=greatlakes-A40-TP4
export TP4_RESULTS=${TP4_RESULTS:-$TP4_DIR/benchmark_results}
export TP4_JOBS=$TP4_DIR/jobs.txt
export TP4_TIME=${TP4_TIME:-44:00:00}   # four arms x six mixed levels at TP=4, about 36 h; a dead job resumes
