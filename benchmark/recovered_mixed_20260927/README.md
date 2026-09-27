# Mixed workload recovery — 2026-09-27

Recovered from Claude's surviving scratchpads and saved tool results. Existing
benchmark results were not overwritten. These files are a partial recovery of
historical results, not newly measured data.

## Recovered 4×4 run

- `mix4x4_h10_programs_recovered.csv`: all 44 per-program summary rows for all
  11 arms, parsed from the original `mix4x4_h10.out` driver log.
- `mix4x4_h10_recovered.csv`: 11 arm summaries. JCT, TTFT, cache percentages,
  session/failure counts, and each arm's own full-load throughput are available.
- Common-window throughput (`throughput`) and completion counts (`done_window`)
  survive for only five arms: vanilla, kvonly, relayout, ours, ours_noprefetch.
  These use the original common window of 3527 seconds. The corresponding
  fields are blank for the other six arms; do not treat them as zeros or
  substitute `own_throughput`, which uses different durations across arms.
- `common_window_counts_recovered` identifies the five complete arm rows.
- `mix4x4_throughput_fixed_windows.csv` provides comparable throughput for
  **all 11 arms** over 600, 900, and 1200 seconds after measurement starts,
  excluding warmup. These checkpoint completion counts survived in the driver
  log. Throughput is completed sessions divided by window length in minutes.
  Each count was checked against the sum of the four per-program counts, and
  each window is within every arm's full-load period. These shorter windows
  are valid comparisons but do not reproduce the original 3527-second metric.
- `mix4x4_throughput_600_1200s.csv` reports completions during (600, 1200]
  seconds, calculated as `done_1200s - done_600s`, divided by 10 minutes.
  It excludes the first ten measured minutes and lies within all arms' full-load
  periods. Sessions started before 600 seconds count if they finish in this
  interval. Constant lane count does not by itself establish steady performance;
  this interval was selected retrospectively during recovery.
- All 20 per-program rows and all five arm rows in the surviving partial CSVs
  were checked against the recovered values. Original copies are retained.

The driver log records successful completion of all 11 arms and ends with
`09-27 09:33:22 MIX4X4-DONE`. A successful driver exit does not mean every
individual session succeeded; the recovered `failed` counts preserve failures.

Configuration: Qwen/Qwen3-14B-AWQ, KV=31830 tokens, HOST=10 GB, CLIP=4096,
xgrammar, four lanes each for fact_check, BFCL_agent, coding_agent, doc_analysis.
Coding runs six sessions per lane; the other programs follow until coding
finishes. One sequential warmup session per program precedes measurement.
This differs from the new `mix16h10_ours_20260927_124528` run, which used five
fixed sessions per lane for every program.

## Preserved evidence and other configurations

`originals/5fd3f06b-1ebf-43ed-9785-032fe725c3a1/` contains the original 4×4
driver script/log and five-arm partial CSVs, plus 3×4 partial CSVs for two
arms and 4×5 partial CSVs for three arms, with their surviving scripts/logs.

`originals/102d1070-bb96-4068-8499-9e2724387a06/` contains surviving older
mixed-run scripts and output logs, including the original three-program
`mix20h10` experiments. Historical protocols differ; do not merge these runs.

`manifest.json` records the 34 source paths, copy paths, byte counts, and
SHA-256 checksums. The source files were copied byte-for-byte and verified.
`claude_mix4x4_tool_evidence.json` preserves relevant historical tool commands
and results with transcript source paths and line numbers.

Main source:
`/tmp/claude-1007/-home-xiaoyu-idontknowitsname/5fd3f06b-1ebf-43ed-9785-032fe725c3a1/scratchpad/`

Claude memory:
`/home/xiaoyu/.claude/projects/-home-xiaoyu-idontknowitsname/memory/project-results-current.md`

## Missing data

No original historical `sessions.jsonl`, per-session logs, or full server logs
were found in the searched project scratchpads/history. No still-open deleted
benchmark files were found through `/proc/*/fd`. Therefore arbitrary common
windows, per-session distributions, and raw call traces cannot be reconstructed
from these summaries alone. No synthetic session records have been created.

`recover.py` reproduces the CSV extraction and validation from the surviving
scratchpads. It does not launch any workloads or execute the copied drivers.
