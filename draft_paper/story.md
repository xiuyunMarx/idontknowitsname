# Story 
Because a significant portion of byLLM program's prompt is compile-time-determined, a compiler can recover the call topology, argument provenance, and invariant/variant split with zero annotations and zero training, and a runtime can exploit that to prefill KV in idle windows and cut cold-start TTFT.

Static analysis for security: https://arxiv.org/pdf/2607.01640; KVcache based: KVFlow, PBKV, Pythia
# Contribution
1. we built a compiler-based static analysis mechanisim to recover the structure for the runtime and Automatically lays out LLM prompt for better KV reuse.
2. We design and implemented a runtime system that proactively and speculatively prefill kvcache which significantly reduce TTFT.
3. We conducted comprehensive evaluations on our framework and prove that this kind of compiler-runtime co-design makes sense in a broader agent applications. (On going, we have only one evluation yet.)

# Motivation
For byLLM, most of its runtime prompt can be determined during compile time, and the remaining part, which are the parameter values, is released during runtime. 

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
