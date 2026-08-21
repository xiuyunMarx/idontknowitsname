# Story 
Because a significant portion of byLLM program's prompt is compile-time-determined, a compiler can recover the call topology, argument provenance, and invariant/variant split with zero annotations and zero training, and a runtime can exploit that to prefill KV in idle windows and cut cold-start TTFT.

Static analysis for security: https://arxiv.org/pdf/2607.01640; KVcache based: KVFlow, PBKV, Pythia

# Difference Axis

Everyone in this space speculates on *which agent runs next*. The axes that separate us:

| axis | KVFlow / PBKV | CacheScout / Pythia | Ours |
|---|---|---|---|
| **where the knowledge comes from** | app-supplied step graph (KVFlow); offline-trained predictor, per workload (PBKV) | online Markov over observed dispatches (CacheScout, ~50 dispatches to converge); developer-supplied workflow interface (Pythia) | **compiler** static analysis, zero annotation, zero training, zero observation |
| **what is known** | next agent id / next-K agent ids | next agent id (top-1) | full may-happen topology (BFS-layered reachable set) + **binding provenance**: which value feeds which future parameter |
| **what the system does to KV** | *move* KV that already exists (CPU→GPU prefetch, survival-scored eviction) | *create* anchor KV via a synthetic warmup request | **create the longest knowable prefix**: invariant + graph ctx + **speculated argument values** + xgrammar precompile |
| **how deep the speculation goes** | none — dynamic parts declared "less valuable for caching" (KVFlow §2); cross-workflow content reuse declared infeasible (PBKV) | fixed anchor only; anchors are 53–62% of prompt tokens *by their own measurement* — that number is their coverage ceiling | Spculation is compile-time derived dependency level |
| **authority over the prompt** | none — prompt is taken as given | none — "no changes to either component" | **the compiler rewrites the layout** (cache-aware zone order, arrival-order binding zone). A serving-layer system structurally cannot do this |
| **cold start** | out of scope (eval pre-warms every agent's fixed prompt) | blind for the first ~50 dispatches | live on request #1 — our measurements *are* first requests |
| **correctness model** | no content speculation → no risk | synthetic warmup prefix, divergence from the real served prompt never discussed | advisory + byte-verified: served prompt is byLLM's own rendering; a wrong guess costs a cache miss, never a wrong prompt |

One-sentence positioning: **prior systems *predict* what the serving layer will be asked for; a language-integrated compiler *knows* it — including the argument values, which is the part everyone else writes off.**

Complementary, not competing: KVFlow/PBKV govern the steady-state *residency* of KV; we govern its *creation*. Say so explicitly — it defuses "already done" and costs nothing.

TODO before posting: read IdleSpec (arXiv 2605.22154, "Exploiting Idle Time via Speculative Planning for LLM Agents") — closest title found, content unread. Verify CacheScout quotes against the PDF.

# Contribution
1. we built a compiler-based static analysis mechanisim to recover the structure for the runtime and Automatically lays out LLM prompt for better KV reuse.
2. We design and implemented a runtime system that proactively and speculatively prefill kvcache which significantly reduce TTFT.
3. We conducted comprehensive evaluations on our framework and prove that this kind of compiler-runtime co-design makes sense in a broader agent applications. (On going, we have only few evluation yet.)

# Motivation
For byLLM, most of its runtime prompt can be determined during compile time, and the remaining part, which are the parameter values, is released during runtime. 

# Who Cares — why TTFT is worth optimizing at all

1. **TTFT is a provisioning metric.** Serving systems are sized and evaluated against TTFT SLO attainment. Cutting TTFT at fixed hardware means more requests served inside the same SLO.
2. **Prefill is the only part of latency still addressable at the systems layer.** Decode time is set by memory bandwidth and by how many tokens the model chooses to emit; a serving system cannot shorten it without changing the model or the output.
3. **Prefill removed from the critical path is throughput, not just latency.** Prefill and decode contend for the same GPU. Work that is done during an idle window is work the loaded engine no longer has to interleave. From serving's view, goodput is the underlying quantity above TTFT.
4. In many agent applications, many LLM calls inside them are consuming large context and egress a small structured output. TTFT is vital for them. 

Who concretely benefits: the app developer (perceived responsiveness), the operator (SLO attainment per GPU), and single-GPU / local deployments.

# Threats to validity / scope of the claim

- We report **ΣTTFT**, the quantity the mechanism acts on. We do not claim a proportional end-to-end win.
- The **Oracle column** upper-bounds what any KV-reuse system can achieve on this hardware. 
- On a 3090, 4B model is too big. No prefill can be paralleled with decode without harming decode latency. 
- We build our system on vLLM. Baseline hence adopts vLLM's APC mechanism. 

# Design
1. Static Analysis: Deduce the topology of byLLM calls, provenance of each paramters and invariant in byLLM's runtime prompt, Automatically lays out every LLM prompt for maximal KV reuse(Which is originally fragmented in byLLM runtime)
2. Runtime: Prefill the byLLM's prompt in the engine idle. (Both spatial idle and temporal idle, based on Chunked Prefill of vLLM)
3. For the parameters that are prepared before calling that byLLM function, proactively prefill them to reduce TTFT.
4. In supervisor agent, truncate the reasoning and fork a probe decode to determine which node to pre-prefill first. (Or using a draft model to do speculate decoding, in short it is training free)

# Evaluation: A case study
An multi-intent agent, with a supervisor dominating 4 subagents.
```
normalize ──▶ route (visit [-->] by llm, select=1, 4 candidates)
                 │
                 ├─▶ MathDesk  : analyze ─[RAG]─▶ solve     ─[RAG]─▶ finalize : MathBrief
                 ├─▶ CodeDesk  : analyze ─[RAG]─▶ implement ─[RAG]─▶ finalize : CodeBrief
                 ├─▶ DocDesk   : analyze ─[RAG]─▶ answer    ─[RAG]─▶ finalize : DocBrief
                 └─▶ WriteDesk : analyze ─[RAG]─▶ draft     ─[RAG]─▶ finalize : WriteBrief
```
Run on Qwen3-4B, 3090, with 20 problems sampled from public datasets like (GSM8K)

Tool calling provides a gap for proactive prefilling. Prefill-Decode aggregation is disabled here because 3090 cannot handle. 

## Results 
Compare the gain for each stage.
1. Baseline: Same ByLLM runtime, with vLLM's Adaptive prefetch caching.
2. Oracle: everything is cached. No prefill needs.
3. Ours: compiler-derived speculative prefill (no probe, no deploy-time prewarm).

| component | n | prompt tok | baseline | ours | oracle | ours vs baseline | headroom captured |
|---|---|---|---|---|---|---|---|
| `normalize (entry)` | 20 | 193 | 37.3 ms | **37.6 ms** | 35.3 ms | 0.99x | n/a (already at floor) |
| `route (visit-by-llm)` | 20 | 549 | 65.8 ms | **65.9 ms** | 37.8 ms | 1.00x | 0% |
| `MathDesk.analyze` | 8 | 500 | 65.8 ms | **66.1 ms** | 37.8 ms | 1.00x | 0% |
| `CodeDesk.analyze` | 4 | 423 | 64.0 ms | **64.3 ms** | 34.5 ms | 1.00x | 0% |
| `DocDesk.analyze` | 3 | 283 | 52.3 ms | **52.3 ms** | 35.2 ms | 1.00x | 0% |
| `WriteDesk.analyze` | 5 | 294 | 52.3 ms | **52.5 ms** | 36.6 ms | 1.00x | 0% |
| `MathDesk.solve` | 8 | 879 | 119.8 ms | **67.8 ms** | 37.7 ms | 1.77x | 63% |
| `CodeDesk.implement` | 4 | 835 | 117.9 ms | **49.7 ms** | 36.2 ms | 2.37x | 83% |
| `DocDesk.answer` | 3 | 707 | 102.3 ms | **62.3 ms** | 37.1 ms | 1.64x | 61% |
| `WriteDesk.draft` | 5 | 676 | 102.0 ms | **50.7 ms** | 37.9 ms | 2.01x | 80% |
| `MathDesk.finalize` | 8 | 1453 | 216.3 ms | **42.7 ms** | 40.1 ms | 5.07x | 99% |
| `CodeDesk.finalize` | 4 | 1292 | 187.0 ms | **42.6 ms** | 37.2 ms | 4.39x | 96% |
| `DocDesk.finalize` | 3 | 1296 | 185.0 ms | **41.5 ms** | 39.2 ms | 4.46x | 98% |
| `WriteDesk.finalize` | 5 | 1258 | 171.1 ms | **41.9 ms** | 39.6 ms | 4.09x | 98% |
| **sum per request** | 20 | ~3300 | **469.9 ms** | **265.1 ms** | **186.2 ms** | **1.77x** | **72%** |

Median over 20 cold-start requests; Oracle is measured by replaying one identically-sized prompt twice and reading the second serve to get 100% cache hit.

# Evaluation 2: ReAct tool-loop QA agent (second topology)

A single-agent ReAct loop, which is a **data-dependent loop**.

```
reason ──[SEARCH]──▶ retrieve (RAG, ~0.3s gap) ──▶ integrate ──┐
   ▲                                                            │
   └──────────────── loop, 2-5 turns ───────────────────────────┘
   └──[ANSWER]──▶ finalize : QABrief
```

The compiler recovers the loop back-edge (`integrate -> {reason, integrate, finalize}`) and, crucially, a **self-referential ret across it**: the running notes are the return of `integrate` and the argument of the next `reason`, `integrate` and `finalize`. Qwen3-4B, 3090, 20 multi-hop questions over a closed synthetic corpus (37 passages, no passage contains both ends of a hop, so each hop needs its own retrieval).

## Results

| call site | Calling times (base/ours) | baseline | ours | ratio | cached tok |
|---|---|---|---|---|---|
| `reason` | 74 / 84 | 39.3 ms | 39.3 ms | 1.00x | 448 → 448 |
| `integrate` | 58 / 71 | 39.4 ms | 38.3 ms | 1.03x | 368 → 544 |
| **`finalize`** | **20 / 20** | **66.8 ms** | **39.2 ms** | **1.71x** | **304 → 496** |

1. In this testcase, ``reason`` has no headroom for reducing TTFT, because there is no gap between integrate and reason.
2. ``integrate`` got a TTFT improment because it consumes the output cumulation of ``integrate``, and the ``integrate``'s output is prefilled during gap.
3. ``integrate`` also consumes value from `reason`, and proactively prefill it in the gap. But due to that the incremental prompt is small, gain is minor.
