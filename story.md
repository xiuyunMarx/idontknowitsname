# Intro

The integration of large language models (LLMs) into programming languages has emerged as an active research direction. Rather than treating model invocation as an external API call surrounded by ad hoc string manipulation, systems such as byLLM, Cotlins, and NVIDIA OOOA expose LLM-backed computation through dedicated language constructs, type information, and runtime support [byLLM, Cotlins, NVIDIA OOOA]. These abstractions address several limitations of conventional LLM application development: prompts assembled through string concatenation are opaque to program analysis, model outputs require defensive parsing, and control flow that interleaves deterministic computation with probabilistic generation is difficult to express, test, and optimize. By making the relationship between program state and model invocation explicit, LLM-integrated languages improve programmability and recover semantic structure that is unavailable in an ordinary API request.

This structure, however, is typically lost at the boundary between the language runtime and the model-serving system. A compiler may know the identity and type of an LLM-backed function, the callsites that can execute after it, and the program values that will flow into downstream calls. The language runtime may further know exactly how those elements are serialized into a prompt. Yet the serving layer receives only a completed token sequence and schedules it as an opaque request. Consequently, information extracted to make an LLM application easier to program is not reused to make that application faster to serve.

Recent agent-serving systems recover part of this missing structure from execution history. PBKV trains a workload-specific, multi-step predictor on offline workflow traces and uses its predictions for KV-cache eviction and prefetching. Pythia mines annotated historical traces into workflow and prompt profiles, and explicitly routes new or changed workflows through a reactive shadow-profiling phase until those profiles become reliable. CacheScout avoids offline training but learns agent-transition probabilities online; before sufficient transitions have been observed, its predictive signal is weak and its cache policy relies primarily on recency. These systems retain deterministic safeguards, but their prediction-driven optimizations require an execution-derived prior [PBKV, Pythia, CacheScout]. When a workflow is first deployed, or when its control flow, agent roster, prompt schema, or coordination policy changes, such a prior may be unavailable or stale. We call this regime *history-free workflow cold start*: the first executions of a workflow after the model is already serving but before a reliable execution profile has been accumulated.

LLM-integrated programming languages offer a complementary source of predictability that is available in this regime: the program itself. We use byLLM as a canonical setting. ByLLM's compiler materializes program semantics in an intermediate representation and its runtime constructs prompts by binding dynamic program values to that representation. This design exposes information that a trace-driven serving layer must otherwise infer. In particular, we make three observations.

1. **A substantial prompt prefix is known before execution.** The system persona, function signature, output schema, tool definitions, semantic annotations, and compile-time constant arguments of an LLM call are invariant across executions of a compiled program. For downstream calls, these components often constitute a large fraction of the final prompt.

2. **Static provenance predicts when dynamic prompt content becomes available.** A return value is not known at compile time, but the compiler can determine the argument provenance of a byLLM function. Once the producer returns, the concrete value may be available. The serving runtime can therefore materialize downstream prompts proactively.

3. **Compiler metadata provides a version-aligned, zero-history prior.** A compiler can enumerate next LLM callsites and can associate speculative prediction results probe with the exact program.

These observations motivate **compiler-guided proactive prefill**, a compiler-to-serving contract for carrying program structure across the abstraction boundary. At compilation time, our analysis extracts LLM declarations and callsites, a conservative may-run-next topology, prompt layout, and parameter provenance. At runtime, the server tracks which values have materialized, constructs the longest known byte-exact prefix of likely successor calls, and submits proactive prefill work during temporal idle and spatial idle.

We make the following contributions:
1. We identify a previously unused compiler-to-serving interface and characterize how compiler-derived value readiness exposes KV-cache reuse opportunities without execution history.
2. We develop a compiler-runtime co-design that turns progressively available program information into exact usable runtime serving information.
3. We design a multi-tenant proactive-prefill runtime that opportunistically schedule workloads.


# Design 
Based on our observations, we build the a compile-runtime co-design


# Evaluation

## Microbenchmark: history-free workflow cold start

This microbenchmark measures what compiler-guided proactive prefill recovers in the regime the introduction defines: the model is already serving, the KV cache is empty, and no execution of the workflow has ever been observed.

**Setup.** One NVIDIA RTX 3090 (24 GB) runs vLLM 0.19.1 with automatic prefix caching and priority scheduling, serving Qwen2.5-3B-Instruct in FP16 with 8 generation slots. Seven byLLM programs are registered as tenants, each with its own connection and its own compiled topology: a tool-using research agent (investigate → answer → audit), an operations agent with two ReAct tool loops, a 3-way LLM-routed ticket handler (`visit ... by llm()`), a support-email pipeline whose reply step carries a 700-token style guide, a six-stage data pipeline, a three-branch moderation fan-out, and a 5-way incident dispatcher. Tool bodies sleep 200–350 ms and consecutive calls are separated by 100–200 ms, standing in for tool execution and interpreter work. Each program has five inputs, cycled across trials.

**Conditions.** `off` is the baseline: the same server, prefix caching on, speculation disabled. `spec` is the full system: may-run-next topology, `const`/`ret` provenance, tool-turn warming and the route probe, with speculative prefill admitted during decode under the profiled per-step token allowance (16 uncached tokens per step at one concurrent decode, 48 at two to nine, 32 at ten to sixteen, calibrated to a +10% mean time-between-tokens bound). `spec-idle` uses the same predictions but prefills only while the engine has no request in flight.

**Protocol.** Every trial boots a fresh server process, so the KV cache and all runtime state start empty; each tenant then runs exactly one workflow and the server is stopped. In *single* mode the seven workflows run strictly one after another; in *multi* mode all seven start within a 3 s window. A trial uses the same inputs and the same start schedule in every condition, and condition order is rotated across trials. We ran 20 trials (840 workflow executions). A (trial, tenant) pair enters a comparison only when both executions exited cleanly and served the same callsite sequence; 24 executions failed (the operations agent exceeds its ReAct iteration limit on one of its five inputs, in every condition) and 4–7 pairs per comparison served different sequences. Both are excluded.

**Metrics.** The primary metric is *uncached prompt tokens at serve*, the part of each request's prompt that vLLM had to prefill (prompt tokens minus prefix-cache hits); it is deterministic and independent of queueing. We also report TTFT, measured from engine admission to first token, summed over the calls of one workflow (ΣTTFT), mean time-between-tokens (TBT), and the number of speculative tokens issued. Differences are paired per (trial, tenant) and reported with a bootstrap 95% confidence interval.

### Single-task cold start

Table E1 compares `spec-idle` with `off` in single mode (136 pairs).

| tenant | n | uncached tokens / workflow, off → spec-idle | Δ uncached, 95% CI | ΣTTFT ms, off → spec-idle | Δ ΣTTFT, 95% CI | ratio |
|---|---|---|---|---|---|---|
| dispatch | 20 | 741 → 645 | +141 [+113, +178] | 126 → 118 | +3.6 [−7.1, +13.1] | 1.07× |
| moderation | 20 | 229 → 139 | +201 [+145, +259] | 81 → 48 | +32.0 [+25.6, +38.8] | 1.68× |
| operations | 16 | 747 → 371 | +385 [+375, +396] | 158 → 123 | +34.1 [+28.4, +39.2] | 1.29× |
| pipeline | 20 | 592 → 133 | +457 [+452, +461] | 143 → 106 | +30.5 [+19.4, +41.6] | 1.35× |
| research | 20 | 644 → 472 | +174 [+169, +179] | 167 → 153 | +7.1 [−1.2, +16.0] | 1.09× |
| route | 20 | 926 → 616 | +268 [+226, +302] | 189 → 161 | +14.6 [−2.6, +27.3] | 1.17× |
| support_email | 20 | 796 → 188 | +608 [+608, +608] | 131 → 76 | +55.0 [+50.9, +59.6] | 1.72× |
| **all** | 136 | | **+317 [+288, +347]** | **140 → 112** | **+25.0 [+20.2, +29.5]** | **1.25×** |

*Table E1. Single-task cold start, `spec-idle` vs `off`. Δ is off − spec-idle; positive favours speculation.*

Speculation removes 317 uncached tokens per workflow (44% of the baseline's 720) and cuts ΣTTFT from 140 ms to 112 ms. The token reduction is significant for every tenant; the TTFT reduction is significant for five of seven. The warm fraction of served prompts rises from 52.9% to 74.8%.

The per-callsite view separates the two kinds of call. Downstream calls, whose predecessors ran before them, arrive almost fully warm: `compose_reply` drops from 590 to 46 uncached tokens (TTFT 72 → 24 ms), `analyze_metrics` from 315 to 28 (39 → 14 ms), `escalate_case` from 272 to 32 (44 → 15 ms), `format_output` from 144 to 10 (24 → 15 ms). Entry calls are unchanged: `investigate` (383 uncached), `collect_metrics` (239), the two routing prompts (254 and 201) and the classifiers that open the email, moderation and pipeline programs (65–92) have no preceding byLLM call, and their bytes are the user's input. After speculation, entry calls account for most of the uncached tokens that remain.

`spec` (speculation also during decode) saves the same tokens (+315 per workflow) but yields 1.19× on ΣTTFT instead of 1.25× and raises mean TBT from 9.13 ms to 9.90 ms: in single-task execution the engine is idle during every tool window, so speculating during decode adds interference without adding coverage.

### Multi-tenant cold start

Table E2 repeats the comparison in multi mode, where the seven workflows cold-start concurrently (131 and 129 pairs).

| condition | warm fraction | mean TBT ms | spec prefills / trial | Δ uncached / workflow, 95% CI | ΣTTFT ms, off → cond | Δ ΣTTFT, 95% CI | ratio |
|---|---|---|---|---|---|---|---|
| off | 53.2% | 10.82 | 0 | | 193 | | |
| spec | 63.3% | 11.87 | 85 | +147 [+126, +169] | 193 → 195 | −10.3 [−18.6, −2.4] | 0.99× |
| spec-idle | 57.1% | 11.02 | 10 | +57 [+38, +79] | 190 → 199 | −11.6 [−18.7, −5.2] | 0.96× |

*Table E2. Multi-tenant cold start vs `off`, pooled over tenants.*

Under concurrency the engine is rarely empty, so `spec-idle` issues 10 speculative prefills per trial instead of 66 and recovers 57 tokens per workflow. Budgeted admission (`spec`) keeps speculating inside decode steps and recovers 147 tokens per workflow, 46% of the single-task saving, at +10% TBT. Neither condition reduces ΣTTFT; both are 10 ms worse per workflow, and no tenant improves significantly. Two measured quantities account for this. The interference profile puts the prefill cost of one uncached token on this model and GPU at about 0.04 ms, so 147 tokens spread over a workflow's five to seven calls are worth roughly 1 ms per call; and with seven workflows in flight every request first waits for the running engine step, which lifts every callsite's TTFT to a 30 ms floor (13–19 ms in single mode) and adds about 2.8 ms to a real request that lands behind a speculative step.

### Cost of speculation

In single mode `spec-idle` prefills 6.2k speculative tokens per trial for seven workflows, 1.1 s of engine time spent inside tool windows that would otherwise be idle. Against the 3.0k uncached tokens it removes from paired serves, 0.35 of each speculative token is used. The unused remainder comes from branches that are warmed but not taken (every `resolve_*` handler of the dispatcher, both branches of the router, all three moderation outcomes) and from prefixes re-issued after a `ret` binding lands; the accounting counts the re-issued invariant part as spent although it is already resident, so the true waste is lower than this figure.
