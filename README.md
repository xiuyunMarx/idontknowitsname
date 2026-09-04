# Workflow-aware KV-cache management for Jac/byllm agents on SGLang

This repository serves multi-agent [Jac](https://www.jac-lang.org/) programs (byllm
`by llm()` call sites) through one SGLang engine and manages the two-tier KV cache
(GPU radix tree + HiCache host tier) from a model of the agent workflow that is
recovered online from the request stream. The baseline is SGLang's own LRU with
the same HiCache configuration.

Contents

1. [System design](#1-system-design)
2. [Algorithms](#2-algorithms)
3. [Experiment design](#3-experiment-design)
4. [Results](#4-results)
5. [Reproducing](#5-reproducing)

## 1. System design

```
  jac program (byllm InterceptorLLM)          server/  (asyncio, port 8964)             model/ + sglang fork
  ┌──────────────┐  POST /v1/chat/completions ┌──────────────────────────────┐        ┌──────────────────────┐
  │ ClaimAnalyzer│ ─────────────────────────▶ │ HttpServer ─▶ Controller     │ ─────▶ │ Engine (SGLang)      │
  │ EvidenceScout│ ◀───────────────────────── │   decompiler.parser.decompose│ ◀───── │  radix tree (device) │
  │ EvidenceAnal.│  POST /v1/sessions/close   │   Program (per agent graph)  │        │  HiCache host tier   │
  └──────────────┘                            │   KVPlanner (jobs, priorities)│  RPC   │  promote / kv_priority│
                                              └──────────────────────────────┘        └──────────────────────┘
```

| Layer | Files | Role |
|---|---|---|
| Proxy and controller | `server/http_server.py`, `server/server.py` | Accepts OpenAI-style chat requests, attributes each to a live session, decompiles it into a call site, forwards it to the engine, and feeds the observation back into the workflow model. Sessions end on `/v1/sessions/close` or an idle timeout. |
| Decompiler | `decompiler/parser.py`, `decompiler/primitives.py` | Turns a request body back into the byllm call site that produced it (function signature, `sem` strings, schema rows), and keeps one `Program` per agent: call graph, value-flow rules, timing statistics, and the next-call predictor. |
| Planner | `server/kv_planner.py` | Converts predicted calls into KV jobs with deadlines (promotion of host-resident prefixes, speculative creation) and pushes the eviction priority map to the engine on every plan change. |
| Engine wrapper | `model/model.py`, `model/promote.py`, `model/device_profiler.py` | Wraps the SGLang `Engine`; adds a KV ledger, speculative-request budgets, the profiled prefill and host-load rates, and two scheduler-side RPCs (`hicache_promote`, `kv_priority`) that steer HiCache. |
| SGLang fork | `sglang/` (branch `kvflow-prefetch`) | `HiRadixCache.prefetch_prefix`: non-blocking speculative host-to-device loads on a low-priority stream; band-ordered eviction with `TreeNode.priority`; per-layer waits when a batch admits a prefix whose copy is still landing. |
| Benchmark | `benchmark/fact_check/` | The HoVer-style fact-check workload, the closed-loop driver, and the CSV consolidation. |

Two serving modes come from one binary: `--no-spec` is the LRU baseline (no planner,
`radix_eviction_policy=lru`), `--manage-only` is the managed arm (planner on, promotion
and eviction steering, no bulk speculative prefill). Both run the same decompiler, so
the only difference between arms is what the engine does with the cache.

### KV tiers

The device pool is what SGLang can allocate after model weights (31.8k tokens for
Qwen3-8B on a 24 GB RTX 3090); the host tier is HiCache with write-through
(`--host 16` GB, 108k tokens). A miss on both tiers re-prefills the whole prompt;
a host hit reloads at PCIe speed (measured 0.03 to 0.05 ms per token with the kernel
copy backend, hidden behind the forward pass layer by layer).

## 2. Algorithms

### 2.1 Decompiling requests into call sites

byllm renders every `by llm()` call the same way: a system prompt, then the function
signature, `sem` descriptions, a schema block, then the argument values, then a
hint. `decompose()` parses that text back into a `Callsite` keyed by the structural
part (everything that does not depend on argument values) and an `Extras` record of
the values, candidate lists (`visit` routing), and the owning object. A typed retry
or the next ReAct turn of the same call is recognised as a continuation, not a new
call.

### 2.2 Learning the workflow

Each agent gets a `Program` (`decompiler/primitives.py`):

* **Call graph and automaton.** Completed sessions are recorded as call sequences.
  `Program.rebuild()` runs an Alergia-style red/blue state merge (`_compat`, Hoeffding
  bound with `alpha=0.05`) over those sequences and rebuilds the probabilistic finite
  automaton; it fires when the session count reaches 8, 16, 32, ... and replays the
  recorded per-call timing onto the new states so state-specific gap and duration
  quantiles survive a rebuild.
* **Next-call prediction.** A k=4 context trie with Witten-Bell backoff gives the
  next-symbol distribution and the end-of-session probability; `predict_tree()` expands
  the top-3 continuations per step to depth 8 and reports, per future call site, its
  probability mass and arrival-time quantiles (t50, t90), built from the automaton's
  gap and duration statistics.
* **Value flow.** For every argument of every call site the program learns where the
  value comes from: a session constant (`copy`), an accumulating list that extends the
  previous call's value (`extend`), or fresh. Once the rules are stable the binding
  layout is frozen and requests are re-emitted in that order (`_relayout`), so the
  r-th call of a site is a byte-level extension of the (r-1)-th: `header | claim |
  history-so-far` is the shared prefix, the fresh bindings and the reply are the tail.
  This is what lets the planner name the exact reusable prefix, not just the agent.

### 2.3 Planning KV work

Every predicted call becomes a `Job` with a deadline (`anchor + t50 - 0.5 * (t90 - t50)`),
its uncached and host-resident token counts, and a value `p * (uncached + 0.8 * host)`.
Jobs run earliest deadline first within a profiled budget:

* **Promotion** (host to device) uses only free device slots plus retired cache, never
  displaces active cache, keeps one prefill chunk free, and is deferred with a
  cooldown when there is no room (the conservative rule from PBKV, see below).
* **Creation** (speculative prefill of a predicted prompt) is disabled in the managed
  arm used for the experiments (`--manage-only`).

### 2.4 Eviction steering

`model/promote.py` puts every radix node in one of four bands on the device tier,
lowest evicted first, LRU inside a band:

| band | contents |
|---|---|
| RETIRED (-2) | private bytes of sessions that have ended (lifecycle tier) |
| TRANSIENT (-1) | the one-off tail of a served prompt: the bytes past the head the flow rules can rebuild (fresh bindings, generated tokens) |
| 0 | anything unclassified |
| PLAN_BASE + s | prefixes a planned job needs |

`s` is the planner's reuse score of a prefix, `p(call) * exp(-dt / 60 s)` where `dt`
is the predicted time until the call arrives; the engine sums the scores of every
plan that covers a node, so a header shared by many sessions outranks any single
session's private history, and among private prefixes the sooner and surer use is
kept longest (`kv_planner.push_priorities`, `promote.kv_priority`).

The host tier is deliberately not steered: `HostStrategy` evicts retired cache first
and everything else by recency, so a private prefix the device gave up stays
reloadable until recency retires it.

### 2.5 What changed and why (Sep 2026)

The first version protected planned prefixes above everything with a rank derived
from a 120 s horizon, demoted every served prompt past the site *header*, and applied
the same bands to the host tier. Under overload (8 sessions' plans larger than the
pool) that sacrificed the same two prefixes of every session and purged them from
host too; in the last session of each lane it lost 53 s against LRU. Diagnostic runs
showed that the host tier, not the ranking, was the culprit. The current design follows
PBKV's split (deterministic lifecycle tier, continuous score for active cache,
host untouched) while keeping two things PBKV does not have: the exact reusable byte
range from the value-flow rules, and the predicted arrival time in the score.

## 3. Experiment design

### 3.1 Workload

`benchmark/fact_check/fact_check.jac` is a HoVer-style multi-hop fact checker with a
cycle: `ClaimAnalyzer -> EvidenceScout x2 -> EvidenceAnalyzer -> (NEED_MORE) ->
ClaimAnalyzer`. Every LLM call takes `(claim, evidences, ...)`; `evidences` grows per
round and carries the retrieved passage (`FC_PASSAGE_CHARS=4000`), so each call site's
prompt extends its previous one by about 1k tokens per finding. Settings used:

| knob | value | why |
|---|---|---|
| rounds | 2 to 6, stop when the verdict is decided (`FC_MIN_ROUNDS=2`, `FC_ROUNDS=6`) | variable session length; retired cache appears mid-run |
| search latency | every web search costs at least 2 s (`FC_TOOL_DELAY_S=2`, cache hits included) | idle windows in which a session's private cache must survive |
| retrieval | Wikipedia TextExtracts, cached per query in `wiki_cache/` | offline and byte-identical across arms |
| claims | 120 HoVer dev claims (`hover_claims_120.tsv`), balanced over hops and label | one distinct claim per session, no cross-session prefix sharing |
| sampling | `temperature=0`, `max_tokens=4096` per call, context length 16384 | greedy; round-6 prompts occasionally exceed the context limit and fall back the same way in both arms |

### 3.2 Driver and methodology

`benchmark/fact_check/fact_bench.py` runs `concurrency` lanes; each lane runs
`--sessions` sessions back to back (total = concurrency x sessions), so concurrency
stays constant until the last session of each lane drains. Two warm-up sessions
precede the measured ones. A fresh server is started for every (arm, level) so the
cache is cold. Both arms see the same claims in the same launch order, which allows
pairing by session index.

Metrics: prefix-cache hit rate (token share served from device, from host, or
recomputed, from the server's per-request log), TTFT per call, and job completion
time (JCT, wall clock of a session). `JCT steady` drops the first and last
`concurrency` sessions. Two paired views are used to separate policy effect from
noise: JCT paired by session index, and TTFT paired on calls whose cache state
(uncached, host, device token counts) is identical in both arms.

Hardware: one RTX 3090 (24 GB), Qwen3-8B, SGLang fork, kernel copy backend for
HiCache, host tier 16 GB. Single run per point; six sessions per lane.

## 4. Results

`qwen8_sweep.csv` holds one row per (concurrency, arm); the figures are
`figs/q8b_v2_hitrate.png`, `figs/q8b_v2_ttft.png`, `figs/q8b_v2_jct.png`.

### 4.1 Hit rate and TTFT

| c | sessions | miss % LRU | miss % ours | host hit % LRU | host hit % ours | TTFT p50 LRU | TTFT p50 ours | TTFT p95 LRU | TTFT p95 ours |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 48 | 45.7 | 46.2 | 35.0 | 34.5 | 0.75 s | 1.00 s | 3.9 s | 4.3 s |
| 10 | 60 | 58.2 | 53.3 | 27.8 | 31.6 | 2.79 s | 2.42 s | 7.9 s | 8.2 s |
| 12 | 72 | 66.7 | 58.1 | 20.5 | 28.1 | 5.06 s | 4.50 s | 11.9 s | 10.5 s |
| 14 | 84 | 73.6 | 61.2 | 15.7 | 25.6 | 8.52 s | 6.49 s | 15.5 s | 11.5 s |
| 16 | 96 | 74.8 | 64.2 | 14.5 | 24.6 | 10.68 s | 9.80 s | 18.1 s | 16.1 s |
| 18 | 108 | 74.1 | 63.8 | 14.7 | 23.8 | 12.87 s | 10.95 s | 19.7 s | 17.6 s |

Where the gain comes from: the managed arm keeps the `EvidenceAnalyzer` prefix
resident (its miss rate drops from 64 to 88 % under LRU to 37 to 46 % at c=10 to 18,
mostly as host hits) while `ClaimAnalyzer` and `EvidenceScout` stay within a few points of LRU.
TTFT on calls with identical cache state is still lower in the managed arm at every
loaded level (c=14: -1.3 s at the median over 592 pairs), because the other lanes
recompute less and the queue is shorter.

### 4.2 JCT

| c | JCT mean LRU | JCT mean ours | change | JCT p95 LRU | JCT p95 ours | steady-state change |
|---|---|---|---|---|---|---|
| 8 | 101.6 s | 108.0 s | +6 % | 184.7 s | 179.3 s | +4 % |
| 10 | 141.6 s | 137.2 s | -3 % | 251.6 s | 239.2 s | -5 % |
| 12 | 185.8 s | 170.1 s | -8 % | 323.7 s | 304.5 s | -10 % |
| 14 | 228.4 s | 185.3 s | -19 % | 405.6 s | 317.2 s | -21 % |
| 16 | 257.4 s | 230.5 s | -10 % | 491.4 s | 434.5 s | -8 % |
| 18 | 283.3 s | 253.7 s | -10 % | 516.0 s | 431.1 s | -14 % |

Reading the curve:

* **c=8 is below the pressure knee.** With 2 s search delays and 2 to 6 rounds the
  live set (about 29k tokens) fits the 31.8k pool; LRU already retains 86 to 98 % of
  the reusable prefixes and any policy ties. The +6 % is workload drift (the managed
  run generated 5.8 % more output tokens) plus timing noise; TTFT on identical cache
  states differs by 9 ms at the median, so the manager adds no fixed per-request cost.
* **c=10 to 14: gains grow with pressure** and peak at c=14 (miss -12.4 pp, TTFT p50
  -24 %, JCT -19 %; the managed run also generated 4.8 % fewer tokens, worth 3 to 4 of
  those percentage points).
* **c=16 to 18: both tiers are oversubscribed.** LRU's miss rate saturates near 74 %,
  the managed arm holds about 10 pp below it, and JCT stays 10 % lower with a larger
  p95 gain (-14 to -16 %).

Single-run JCT means carry about ±6 % noise from greedy-decoding path divergence and
queue position; hit rate and TTFT are the primary evidence, JCT the supporting one.
The client-side residual per session (search waits and program overhead) is identical
across arms at every level (18.6 to 19.7 s), so search latency does not contribute to
any difference.

### 4.3 Relation to PBKV

PBKV (arXiv 2605.06472) reports gains only in the band where LRU thrashes but a
good policy can still hold the live set (concurrency 60 to 84 for their HoVer +
LangChain setup, all policies converge at 96). This system shows the same shape:
nothing at c=8, a peak at c=14, convergence toward a constant offset at c=16 to 18.
The mechanisms shared with PBKV are the lifecycle (retired) tier and conservative
prefetching; the additions are the byte-exact reusable prefix from the value-flow
rules and the arrival-time term in the score.

## 5. Reproducing

```sh
P=~/miniconda3/envs/sglang/bin/python          # env with the sglang fork installed editable

# one arm, one level, by hand
$P -m server.server Qwen/Qwen3-8B --no-spec --host 16 > lru.log 2>&1 &     # LRU baseline
$P -m server.server Qwen/Qwen3-8B --manage-only --host 16 > ours.log 2>&1 & # managed
export FC_CACHE_DIR=$PWD/benchmark/fact_check/wiki_cache FC_MIN_ROUNDS=2 FC_TOOL_DELAY_S=2
$P -m benchmark.fact_check.fact_bench --sessions 6 --concurrency 12 --warmup 2 \
   --claims benchmark/fact_check/hover_claims_120.tsv --tag lru --server-log lru.log \
   --logdir results/run_lru_c12 > lru_c12.out

# the sweep used for the tables above (fresh server per arm and level, CSV at the end)
LEVELS="8 10 12 14 16 18" ARMS="lru ours" ./qwen8_sweep.bash
$P -m benchmark.fact_check.sweep_csv qwen8_sweep.csv "q8b:Qwen/Qwen3-8B:results/qwen8_sweep_v2"
~/miniconda3/envs/jaseci/bin/python figs/plot_sweep.py qwen8_sweep.csv figs q8b q8b_v2

# decompiler self-check
$P -m decompiler.test_history
```

Notes for a clean run: the device pool size is printed as `[ledger] device=N tokens`
when the server starts; any other process holding GPU memory shrinks it and makes the
run incomparable. The sweep script kills leftover `sglang::`, `server.server`, `jac run`
and `fact_bench` processes between levels; do not run unrelated shells whose command
line contains those strings while it is active.
