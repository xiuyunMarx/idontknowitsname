# 16-lane mixed workload — 2026-09-16

Ran fresh servers sequentially for `lru` (reordering only) and `nextvisit`
(ours with the next-visit planning fix). No source changes were made for this
experiment. Both use Qwen/Qwen3-8B, FCFS, 8 GB host cache, and the default
device KV pool. Lanes: fact_check=4, coding_agent=6, doc_analysis=3,
BFCL_agent=3. Four sessions per lane, one warmup per application.

Both arms completed all 64 measured sessions with exit code zero. Input tuples
(program, phase, index, input) match exactly. No opaque requests, template
mismatches, or tracebacks occurred. Both servers were stopped after completion;
GPU memory returned to 41 MiB.

## Results

Warmup is excluded from request/session metrics. TTFT is averaged over measured
requests; cache percentages are weighted by prompt tokens. Full-run throughput
includes the draining period and is not steady-state throughput.

| Metric | 16-lane lru | 16-lane ours |
|---|---:|---:|
| Measured workload span (s) | 864.65 | 810.91 |
| Full-run sessions/min | 4.441 | 4.735 |
| Mean TTFT (ms) | 4875 | 4301 |
| Cache miss (%) | 52.24 | 43.58 |
| Device hit (%) | 31.71 | 30.39 |
| Host hit (%) | 16.05 | 26.03 |
| Sessions done by 350 s | 21 | 25 |
| Sessions done by 600 s | 48 | 51 |
| Model requests | 879 | 888 |
| Output tokens | 105005 | 106592 |
| Output tokens/s over full run | 121.44 | 131.45 |
| Mean requests awaiting first token at a first-token event | 3.93 | 3.95 |
| Mean requests decoding at a first-token event | 7.41 | 7.40 |

All lanes remain active through the common 350-second window: each run's first
lane finishes at 413.2 s and 367.7 s respectively. Their independently computed
full-load-window throughput numbers use different intervals and should not be
directly interpreted as a fixed-window throughput comparison.

| Application | lru mean JCT (s) | ours mean JCT (s) |
|---|---:|---:|
| coding_agent | 201.0 | 187.4 |
| fact_check | 169.8 | 155.7 |
| doc_analysis | 107.2 | 96.3 |
| BFCL_agent | 127.9 | 122.4 |

Relative to lru, ours has 6.6% higher full-run session throughput, 8.2% higher
output-token throughput, 11.8% lower mean TTFT, and 8.66 percentage points less
cache miss. Output volume is 1.5% larger, so unlike the earlier old-ours versus
fixed-ours comparison, this run's improvement is not explained by fewer output
tokens. Agent trajectories still differ, and this is one run per arm.

## Comparison with the earlier 24-lane runs

| Metric | 24-lane lru | 24-lane fixed ours | 16-lane lru | 16-lane fixed ours |
|---|---:|---:|---:|---:|
| Full-run sessions/min | 4.052 | 4.300 | 4.441 | 4.735 |
| Mean TTFT (ms) | 11118 | 9966 | 4875 | 4301 |
| Cache miss (%) | 59.95 | 56.07 | 52.24 | 43.58 |
| Mean requests awaiting first token | 9.22 | 8.96 | 3.93 | 3.95 |

The relative session-throughput gain of ours over lru is similar: 6.1% at
24 lanes versus 6.6% at 16 lanes. Cache-miss improvement is larger at 16 lanes
(8.66 versus 3.88 percentage points). This is consistent with reduced memory
and admission pressure making cache preservation more useful, but not with
all KV mechanisms being disabled at 24 lanes.

Speculative loading still has limited room: the 16-lane ours server records
2,258 no-room deferrals and only 12 actual promotion starts (10,196 tokens),
including warmup. Most of the extra measured cache hits are still on the host,
not the device.

Cross-concurrency comparisons are descriptive: 16 lanes run 64 sessions and
24 lanes run 96, and their application proportions differ slightly. A repeated
experiment with identical total inputs would be needed to isolate concurrency
as the cause of the differences.

## Artifacts

- `mix16_lru_20260916/summary.json` and `sessions.jsonl`
- `mix16_nextvisit_20260916/summary.json` and `sessions.jsonl`
- `mix16_lru_20260916.out` and `mix16_lru_20260916_server.log`
- `mix16_nextvisit_20260916.out` and `mix16_nextvisit_20260916_server.log`
