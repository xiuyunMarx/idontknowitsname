# Next-visit planning verification — 2026-09-16

## Change

`Program.predict_tree()` now aggregates only first future arrivals at each
callsite. A path that has already visited the target after the current call
does not contribute another occurrence to that target's probability, timing,
or value-flow paths. Search expansion through that occurrence still continues,
including loop counters and predictions of downstream exits.

This keeps nested SCC analysis and multi-hop prediction. It is not a depth-one
forecast: each callsite gets a prediction of its next visit. Alternative paths
to that visit still merge conservatively. Later visits are reconsidered when
the session advances and replans.

## Functional verification

- Added three regression tests covering next-visit probability and self loops,
  downstream reachability through repeated visits, unequal-length alternative
  paths, and the token target produced by `Controller._plan_call()`.
- Before the fix, two new tests failed: later loop iterations truncated the
  next Coder prompt and inflated the next-visit probability.
- After the fix, `python -m unittest discover -s tests -v` passed all 27 tests,
  including existing nested-SCC, iteration-counter, and branch-statistics tests.
- Replayed the real cached coding template with identical synthetic live values
  and the old versus new prediction method. The old Coder forecast merged 15
  paths and reconstructed only 25 characters (`complete=False`); the new one
  retained one first-arrival path and reconstructed all 1,573 characters
  (`complete=True`), including spec, tries, plan, advice, and the header.

## Mixed workload

Ran a fresh fixed server with Qwen/Qwen3-8B, FCFS, and 8 GB host cache, using
the `run_mix.bash` defaults: fact_check=6, coding_agent=8, doc_analysis=5,
BFCL_agent=5; four sessions per lane and one warmup per application. All 96
measured sessions finished with exit code zero. The server was stopped after
completion, and GPU memory returned to 41 MiB.

New tag: `mix24_nextvisit_20260916`. Existing baseline files were preserved.
Comparison is against the earlier runs, not newly repeated controls.
The (program, phase, input index, input) tuples match the old ours run exactly.
Recompiled coding, fact-check, and document registration payloads also match
their prior cached payloads.

Metrics below exclude warmup unless specified. Overall cache percentages are
weighted by prompt tokens; overall TTFT is weighted by requests.

| Metric | Existing lru | Old ours | Next-visit ours |
|---|---:|---:|---:|
| Measured-phase elapsed (s) | 1421.6 | 1441.7 | 1339.6 |
| Mean TTFT (ms) | 11118 | 11292 | 9966 |
| Prompt cache miss (%) | 59.95 | 55.82 | 56.07 |
| Device hit (%) | 26.40 | 25.92 | 25.45 |
| Host hit (%) | 13.65 | 18.26 | 18.49 |
| Sessions finished by 600 s | 40 | 33 | 40 |
| Sessions finished by 900 s | 65 | 63 | 68 |
| Sessions finished by 1200 s | 87 | 85 | 92 |
| Model requests | 1276 | 1319 | 1281 |
| Output tokens | 161152 | 170914 | 161618 |

| Application | Old ours mean JCT (s) | Next-visit mean JCT (s) |
|---|---:|---:|
| coding_agent | 324.1 | 300.8 |
| fact_check | 284.0 | 263.7 |
| doc_analysis | 170.9 | 142.8 |
| BFCL_agent | 208.7 | 190.9 |

For measured Coder arrivals with a logged plan, mean target length increased
from 372 to 1,216 tokens. This is target size, not achieved cache reuse:
mean planner-recorded completed length was 364 versus 206 tokens.
Including warmup, actual promotion starts increased from 14 to 25, while
no-room deferrals remained high (3,266 versus 3,197). No opaque-request,
template-mismatch, or traceback messages occurred in the new server log.

## Interpretation

The specific cross-iteration prefix truncation is fixed and covered by tests.
This single mixed run completed 7.1% sooner than old ours, with 11.7% lower
mean TTFT. It also outperformed the existing lru run on elapsed time and mean
TTFT. However, overall cache miss did not improve, and output tokens fell
5.4%, so this run does not establish that the performance change comes solely
from better KV reuse. Admission and promotion-space limitations remain.

The benchmark's full-load window is different in each run (old ours 601.1 s,
new ours 513.4 s), so its 3.39 versus 3.97 sessions/min values should not be
treated as throughput over an identical interval. Fixed-window counts and
full-workload elapsed are reported above instead.

Artifacts: `mix24_nextvisit_20260916/summary.json`,
`mix24_nextvisit_20260916/sessions.jsonl`,
`mix24_nextvisit_20260916.out`, and
`mix24_nextvisit_20260916_server.log` in this directory.
