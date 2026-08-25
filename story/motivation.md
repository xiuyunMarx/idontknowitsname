# Motivation

LLM-integrated programming languages offer a complementary source of predictability that is available in this regime: the program itself. We use byLLM as a canonical setting. ByLLM's compiler materializes program semantics in an intermediate representation and its runtime constructs prompts by binding dynamic program values to that representation. This design exposes information that a trace-driven serving layer must otherwise infer.

In particular, we have the following observations.
1. A major part of prompt in byLLM framework can be duduced during compile time.
2. Arguments for a LLM call might be ready a long time before the call site. 

An agent workflow alternates model calls with program execution. Program execution creates two forms of unused server slack.

Temporal idle occurs during tool execution and interpreter movement between call sites. During this period, the workflow has no request in the engine. Across five agentic benchmarks served on local vLLM, tool execution takes 2-29% of workflow runtime [AgenticWorkloads]. For deep-research and coding agents with network-bound tools, tool execution takes 45-57% [PASTE]. Individual tool calls average 0.9 s on SWE-Bench and 1.9 s on web search, with long tails [Continuum].

Computational slack is idle compute while the engine is busy. Decode accounts for 91.0-98.6% of model time on those same benchmarks and prefill for only 1.4-9.0% [AgenticWorkloads]. Across our three workloads, the split is 92.3% and 7.7%. Memory-bound decode leaves compute idle. On one RTX 3090 serving Qwen2.5-7B, adding 48 extra uncached prefill tokens to a decode step increases time between tokens by 2.1% at every decode concurrency from 1 to 16.

Serving systems already multiplex both forms of slack, but only among requests that have been issued. Temporal multiplexing fills idle periods across requests. It serves one workflow's decode during another workflow's tool window. Spatial multiplexing fills idle compute within one engine step. Chunked prefill splits a prompt into chunks and schedules a chunk alongside running decodes. This pairing raises the arithmetic intensity of a memory-bound decode step [Sarathi, SarathiServe, MuxServe]. The 48-token budget above measures this spatial capacity on our hardware.

Load affects the two forms differently. At higher load, other workflows fill temporal gaps. Computational slack remains within memory-bound decode steps. This slack can support prefill before a request arrives. Using it requires selecting what to prefill without an execution history.

<!-- refs:
AgenticWorkloads = Agentic AI Workload Characteristics, arXiv 2605.26297
PASTE = Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving, arXiv 2603.18897
Continuum = Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live, arXiv 2511.02230
Sarathi = SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills, arXiv 2308.16369
SarathiServe = Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, OSDI 2024, arXiv 2403.02310
MuxServe = MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving, arXiv 2404.02015
-->
