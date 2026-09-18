# mix20: mixed workload, 2026-09-18 01:42-04:41

## What was run
Three programs on one server at the same time, one server per arm, arms run one after another
on the same machine: vanilla (plain SGLang), relayout (re-layout only, LRU eviction), ours
(re-layout + KV planner with the 0.8 token-usage gate on prefetch).

| program      | lanes | sessions per lane | inputs                          |
|--------------|-------|-------------------|---------------------------------|
| fact_check   | 4     | follow            | HoVer claims, 120               |
| BFCL_agent   | 4     | follow            | BFCL v4 web search, 100 (wraps) |
| coding_agent | 12    | 6 (fixed)         | HumanEval bundles, 157          |

A lane runs sessions back to back. The 12 coding lanes are "fixed": each runs exactly 6 sessions
and then stops. The fact and BFCL lanes "follow": they keep starting new sessions until the last
coding lane has finished, so all 20 lanes stay busy for the whole run. Lane counts were chosen so
the three programs' live prefixes together (~104k tokens) exceed the host tier (54k) by about 1.9x,
the regime where the planner has something to keep.

## Server / engine
- Qwen3-8B, one 24 GB GPU, device KV pool 31,830 tokens.
- HOST=8: 8 GB host (CPU) KV tier, about 54k tokens.
- CLIP=128 (SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION): admission ignores future output growth, so the
  pool fills with prompts; the engine ran ~13 requests with ~7 queued, token usage ~0.96.
- All three agents: max_tokens=1024 (stop condition only). Constrained JSON decoding on
  (response_format forwarded to sglang), so no typed-output retries except runaway strings.
- BFCL: full tool results kept in history (BF_EXCERPT_CHARS=0), tool delay lognormal mean 2 s,
  std 2 s (BF_TOOL_DELAY_S=2, BF_TOOL_DELAY_STD_S=2); fact and coding tool delay fixed 2 s.
- Each arm starts with one warm-up session per program (not measured).

## How the numbers are defined
- `full_load_s`: from the first measured session start until the first fixed (coding) lane
  finishes its 6 sessions; during this window all 20 lanes are busy.
- `sessions_per_min_full_load`: sessions (all programs) completed inside that window / minutes.
- `done_600s/900s/1200s`: sessions completed within that many seconds of the start, same clock
  for every arm; these fixed windows are the like-for-like throughput comparison.
- Per-program rows: sessions completed in the whole run (follow lanes complete a different
  number per arm, so per-program means are session-weighted), JCT = session wall time,
  TTFT = server-side time to first token per request, device/host/miss = share of prompt
  tokens served from GPU cache / loaded from host / recomputed.
- The row "all three programs together" is the sum over the three programs.

## Files
- mix20.csv: this summary. mix.log / mix20_<arm>.out: driver output.
- ../cache/mix20_<arm>/sessions.jsonl: one line per session. summary.json: same numbers as the CSV.
- ../calls/mix20_<arm>.csv: one line per HTTP request (prompt, device/host hit, recomputed, TTFT).
- mix20_<arm>_server.log: server + engine log (--engine-log info).
