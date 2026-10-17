# expriment plan on Great Lakes
Device: A40 48GB VRAM, partition spgpu, account lingjia0 (Great Lakes has no H100; the 72B-AWQ engine needs ~55 GB)
Model: Qwen3-32B-AWQ, 19.5 GB (64 layers, 8 KV heads, 262 KB/token)
Device Pool= 31830 tokens, 8.3 GB    (--kv 31830)
Host Pool= 76k tokens, 20 GB          (--host 20)
CLIP=4096 (sglang default)
mem_fraction_static, attention backend: sglang defaults
context_length: 32768

## SLURM per job
1 A40, 12 cores, --mem=96G, 72 h. One server per arm: device KV pool 61440 tokens (15 GiB; sglang profiles 19.25 GiB after the AWQ weights, the rest is left idle so host/device stays near 1.5), host tier 25 GB, session timeout 7200 s. The four sweeps run back to back with no flush; then one KV flush, then the six mixed levels with a KV flush between them (learned statistics are never reset).

## Per-agent concurrency sweep
sessions per lane k is fixed per workload. Inputs wrap when c * k > dataset size (a repeat comes >=100 sessions later; the pools turn over within a minute, so it will miss). Requires wrap=True for fixed-k lanes in mixed_workload.py.


| workload | inputs | k | c levels |
|---|---|---|---|
| fact_check | 120 | 4 | 4, 8, 12, 16, 24|
| coding | 157 | 3 | 8, 16, 32, 48, 64, 80|
| BFCL | 100 | 3 | 4, 8, 12, 16, 20, 24|
| finance | 150 | 4 | 4, 6, 8, 12, 16, 20 |

Inputs listed in `benchmark/applications/bench_data/dirty_inputs.json` are skipped by every arm (coding 11, 16, 26, 51, 52, 56, 111; finance 53; BFCL 47, 71): greedy decoding runs into max_tokens inside a JSON field on them under at least one prompt layout. A session that still hits max_tokens counts in throughput, JCT and TTFT but not in accuracy (`excluded` in summary.json). Data from before this filter is under `greatlakes/stale/results-dirty-inputs/`.

## Mixed workload
lanes:  [2, 3, 4, 5, 6, 8]
4 agents, equal lanes per agent; coding fixed at 6 sessions per lane, the others follow.

vanilla, adapter, kvonly, relayout, ours, continuum, cachescout, kvflow, continuum_relayout, cachescout_relayout, kvflow_relayout, ours_noprefetch;

## Arms
For per-agent sweep: vanilla, adapter, kvonly, relayout, ours, continuum, cachescout, kvflow, continuum_relayout, cachescout_relayout, kvflow_relayout; 

For mixed sweep: vanilla, adapter, kvonly, relayout, ours, continuum, cachescout, kvflow, continuum_relayout, cachescout_relayout, kvflow_relayout, ours_noprefetch;

## NOTE:
Keep all logs to ensure we can calculate data later. 
Metrics we need: JCT, TTFT, Cache hit on host, Cache hit on host, Cache miss; Thoroughput
