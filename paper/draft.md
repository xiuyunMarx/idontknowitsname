# Proactive Prefill: Compiler-Derived Speculative Prompt Construction for Language-Integrated LLM Agents

*Working draft for internal discussion — 2026-08-18. Status: all measurements are from this project's own runs; references were gathered via same-day search at citation level and have NOT been independently verified (verification pass skipped by author's choice) — check titles/venues/ids before external use. Reference [4] (PromptCache) lacks its official title/author list.*

## Abstract

Agentic LLM applications issue chains of model calls, and every call pays a time-to-first-token (TTFT) cost to prefill a prompt that is, to a first approximation, already known before the call happens. In a language-integrated agent framework such as Jac's byllm, we measure that 96.6% of a representative call's prompt characters are compile-time constant, and much of the remainder is determined by program state observable well before the call is issued. We present Proactive Prefill, a serving-side runtime that exploits this: a compiler pass derives, with zero developer annotations, (i) the byte-exact invariant prefix of every LLM call site, (ii) a may-happen-next topology over call sites, and (iii) a provenance table mapping each call's arguments to their value sources. A guard server uses these artifacts to speculatively construct prompts before calls arrive — warming invariant prefixes and output-grammar automata during predecessors' decode phases, and feeding argument bindings the moment their source values are observed, across any number of intervening calls. All speculation is advisory: served prompts are always rebuilt and verified against real values, so a misprediction costs a cache miss, never a wrong prompt. On real applications running unmodified except for a one-line connector swap, we observe 2.35× end-to-end TTFT reduction on a four-call chain, 5.3× on a router's cold start when graph context is fed ahead of the first request, 8.3× on the first structured-output call via grammar precompilation, and 2.67× on a fan-in consumer call from binding speculation alone. We also report a negative result with design consequences: speculation submitted inside any call's TTFT-critical scheduling window cancels its own benefit, and priority scheduling does not prevent this; displacing all speculative work into decode shadows does.

## 1 Introduction

Agentic applications are becoming the dominant shape of LLM workloads: a routed multi-agent assistant, a retrieval-augmented pipeline, or a tool-using worker issues not one model call but a chain of them, with each call's output steering the next. Serving systems have responded by reusing computation across calls — most visibly automatic prefix caching, which stores the key-value (KV) state of previously seen prompt prefixes so that repeated prefixes skip prefill [3, 6]. A growing body of work extends this reuse across agents and turns [1, 2, 7].

These systems share a structural assumption: the prompt is an opaque string that comes into existence when the application submits a request. Everything the server can do, it does after that moment — cache what it has already seen, schedule what it has just received. The consequence is that the first execution of every call site pays full prefill; a freshly deployed or restarted engine pays it for every call site; and the dynamic portion of each prompt pays it on every single call. In workflow-aware caching work this last cost is explicitly written off: KVFlow, the closest system to ours, states that the dynamic parts of agent prompts "are less valuable for caching" and always evicts them first [2].

The premise of this paper is that for a language-integrated agent framework, the opacity assumption is false. In Jac's byllm [13], an LLM call is a typed function — `def respond(message: str, chat_history: list) -> str by llm()` — whose prompt is constructed by the framework from the signature, semantic annotations, tool schemas, and the declared return type. The prompt is therefore a compile-time object with a small runtime-varying suffix. We measure this directly: on a production-style RAG application's main call site, the invariant portion is a strict string prefix covering 96.6% of prompt characters, and across call sites of 1k-4k tokens the invariant fraction is around 95%. Moreover, the runtime-varying suffix — the argument bindings — is itself largely determined before the call: arguments arrive from earlier calls' results, from walker (agent) state visible in earlier requests, or from constants.

Exploiting this requires solving four problems. First, the invariant/variant split, the inter-call structure, and the argument sources must be recovered without asking developers to annotate anything — otherwise the approach degenerates into the annotation-based structure exposure of prior work [1]. Second, speculation must be incapable of corrupting results: a serving layer that guesses prompt content must never serve a guessed prompt. Third, argument values become observable at scattered, unpredictable moments; the mechanism that turns an observation into warmed cache must follow dataflow, not call adjacency. Fourth — and this we learned from our own measurements — speculative work competes with the very latency it is trying to remove, and naive scheduling gives back everything it gains.

Proactive Prefill addresses these with a compiler/runtime co-design. A static pass over the program's IR produces three artifacts: byte-exact invariant prompt prefixes for every call site (including tool protocols and output schemas), a may-happen-next call-site topology (covering sequential flow, branching, graph-traversal dispatch, and LLM-routed `visit` statements, which are themselves call sites), and a binding-provenance table classifying every argument as constant, agent-state field, or the return value of another call site. A guard server owning a vLLM engine [6] consumes these artifacts. It organizes speculation in three tiers of stability: compile-time invariants, warmed via the topology during predecessors' decode; quasi-static graph context (a router's candidate set and its choice grammar), fed once per graph change and reused across requests; and per-request bindings, fed in arrival order the moment provenance fires, then matched and reordered — never injected — at serve time. Correctness rests on a single advisory principle: speculation only ever populates cache; every served prompt is rebuilt from real values, with fed content used only where it matches verbatim.

We evaluate on two real Jac applications and three targeted workloads, all running through the real interpreter with a one-line model-connector change, on Qwen2.5-3B with deterministic decoding. Speculation reduces end-to-end TTFT 2.35× on a four-call chain (individual calls 3.4-8.3×); feeding a router's graph context ahead of the first request on a fresh engine reduces its cold-start TTFT 5.3×; grammar precompilation inside warms removes the ~300 ms first-call cost of structured output; and on a fan-in workload with ~1.4k tokens of bindings, binding speculation alone accelerates the consumer call 2.67×. Engine-reported cached-token counts confirm that every gain is attributable to its intended mechanism. We additionally show that a stability-ordered prompt layout for LLM-routed dispatch improves routing accuracy on our probe from 4/10 to 8/10 while tripling its reusable prefix.

This paper contributes: (1) the observation, with measurements, that language-integrated LLM calls have compiler-recoverable prompt structure — invariants, topology, and argument provenance — requiring no annotations; (2) a three-tier speculative prompt-construction runtime built on an advisory correctness principle that makes aggressive over-approximation safe; (3) dataflow-fired binding speculation with verified serve-time reordering, which converts the prompt content prior work deems uncacheable into reusable prefix; (4) a scheduling discipline — speculation displaced into decode shadows — motivated by measured interference that priority scheduling alone does not prevent; and (5) an implementation and evaluation on unmodified real applications.

## 2 Background and Motivation

### 2.1 Anatomy of a language-integrated LLM call

In byllm, the framework constructs each call's prompt from declared program structure. The system message carries a fixed persona and, for tool-using calls, a rendered tool protocol with per-tool JSON schemas. The user message opens with the qualified typed signature and the author's semantic annotations (`sem` strings), followed by one binding line per argument, `name = repr(value)`, and closes with an optional instance-state zone. Structured returns additionally induce a JSON-schema decoding constraint whose grammar automaton the engine compiles on first use. Every component except the binding values and instance state is determined at compile time; crucially, the invariant components form a contiguous prefix of the token sequence, with the runtime-varying bindings at the very end. LLM-routed graph dispatch (`visit ... by llm()`) follows the same pattern with different zones: a compile-time goal, a graph-dependent candidate list, and a per-request agent-state zone.

### 2.2 Where first-token latency goes

Prefix caching makes repeated prefixes nearly free, so the TTFT costs that remain in agentic serving are concentrated in three places, which we quantified during system bring-up. First, cold call sites: the first request to each call site — every call site, after a deploy or restart — prefills its full invariant (on a 0.5B model with a ~1.2k-token call site, we measured 1153.7 ms cold against 279.4 ms with the invariant pre-warmed, 4.1×). Second, first structured-output calls: compiling the output grammar cost ~300 ms on first use in our setup, serialized into that call's TTFT. Third, per-request content: binding values never repeat, so under the standard regime they are prefilled on every call — this is the portion KVFlow's authors explicitly deem not worth caching [2], and it grows with exactly the workloads that matter (long histories, retrieved context, long inter-agent messages).

### 2.3 The observability window

What makes these costs attackable is that the missing information usually exists before it is needed. Which call runs next is constrained by program structure. What its arguments will be is often already determined: a router's request carries the agent state whose fields become the routed agent's arguments; a completed call's result is the next call's input, bound to a variable that never mutates. The window between observability and use — a predecessor's entire decode phase, often seconds — is idle prefill capacity on a decode-bound GPU. Proactive Prefill is the machinery that turns that window into warmed cache.

## 3 Design

### 3.1 Three tiers of prompt stability

We organize all speculation around when content becomes known and how long it stays valid:

| Tier | Content | Becomes known | Valid across |
|---|---|---|---|
| 1. Invariant | persona, signature, sems, tool protocol, output schema and its grammar | compile time | all requests, forever |
| 2. Quasi-static | routed dispatch's candidate descriptions and choice grammar | when the graph is built or changes | all requests until graph change |
| 3. Per-request | argument bindings | when provenance fires at runtime | one call (and its retry/tool turns) |

Each tier has its own trigger. Tier 1 is warmed on deployment (optional) and speculatively via the topology: when call X emits its first token, every may-successor's invariant is warmed during X's decode. Tier 2 is fed when graph context is available — supplied explicitly, or learned by the server from the first served routing prompt and exported so a restarted engine can be fed before its first request. Tier 3 is fed by dataflow (§3.3).

### 3.2 The advisory principle

No speculative content is ever served. Warms and feeds only populate the engine's prefix cache; when a real call arrives, its prompt is constructed from the real arguments byllm supplies. Fed binding lines are used only for reordering (§3.3), and only where they match the real rendering byte-for-byte; any divergence falls back to the framework's default rendering from that line onward. A wrong guess therefore costs at most the cache miss that would have occurred anyway. This principle is what licenses aggressive over-approximation everywhere else in the design: the topology may include edges that never fire, provenance may feed values that go stale, and the system remains correct by construction.

### 3.3 Compiler-derived artifacts

**Invariant construction.** The static pass compiles each call site's invariant prompt exactly as the framework's runtime would render it — the same strings, the same zone order, byte for byte. Byte exactness is load-bearing: prefix caches match hashed token blocks, and we found during bring-up that an invisible whitespace divergence at character 19 (the framework's triple-quoted strings do not dedent) silently zeroed every cache hit while leaving behavior otherwise unchanged. Structural sharing is not enough; the artifact must be the identical string.

**Call-site topology.** A may-happen-next graph over call sites is computed from control flow: sequential statements, both branch arms, loop back-edges, continuation into callers, graph-traversal dispatch (a walker ability ending hands control to any ability that walker can still trigger), and LLM-routed `visit` statements, which are themselves call sites with their own invariants. The graph is deliberately over-approximate — a spurious edge wastes an idle-time warm; a missing edge costs a cold call.

**Binding provenance.** For every call-site argument, the pass classifies its source: a constant; a field of agent state (with any slice expression preserved); the return value of another call site (traced through local variables); or unknown, which is never fed. Provenance is inverted into a consumer index — for each producer call site, the (consumer, parameter) pairs its result flows into — enabling dataflow firing: when X completes, its consumers are looked up directly, at any distance downstream. In `a = f(...); b = g(...); h(a, b)`, h's first binding is fed the moment f completes, while g has not yet started. Results accumulate in a store so a consumer fed at different times evaluates against everything observed so far. Return values are safe to feed at any distance because the bound variables are immutable after assignment; agent-state fields are mutable, so field-sourced feeds are restricted to topology-adjacent successors, where adjacency doubles as a freshness heuristic.

**Arrival order and serve-time reordering.** The framework renders bindings in declaration order, but values become available in arrival order, and a KV prefix can only be extended at its end. We therefore feed bindings in arrival order and reconcile at the serving boundary: the served prompt's binding block is scanned for lines that match the fed session verbatim, and matched lines are moved to the front in fed order — content is moved, never injected. Declaration order has no cross-request cache value anyway, since binding values are unique per request; arrival order is strictly better for overlap. Sessions are matched by content, persist across a call's tool-use turns so every turn reorders identically, and self-heal: re-feeding a changed value truncates the stale suffix.

### 3.4 Scheduling discipline: speculation in decode shadows

Our first implementation triggered successor warming when a call started, and binding feeds when a producer completed. Both interfered with real traffic in the same way: the speculative request entered the engine in the same scheduling window as a real call's prefill and delayed its first token — measurably, even with the engine's priority scheduling demoting all speculative work, because the interference includes work (such as grammar compilation) incurred at submission. The fix is temporal, not priority-based: speculative work is submitted only inside decode shadows. Successor warming and field feeds trigger on the current call's first token — its TTFT-critical window has just closed, with nearly all of its decode still ahead. Result-sourced feeds are deferred at producer completion and flushed at the next call's first token. One consequence is accepted openly: in a zero-think-time sequential chain, the last-arriving binding of a consumer cannot profitably be fed (its observation and its use are adjacent), and the system abandons it rather than pay the interference.

### 3.5 Cache-aware prompt layout

Stability ordering also applies to the prompt itself. The framework's routed-dispatch prompt interleaved a per-request agent-state zone before the graph-stable candidate list, capping cross-request prefix reuse at the compile-time goal line. Reordering the zones by stability — goal, current node, candidates, then per-request state — makes everything up to the state zone a reusable prefix. The reorder is gated behind an environment flag and changed routing behavior only for the better in our probe (§5.4).

## 4 Implementation

The system is three components around an unmodified application. A static layer (engine-free Python) compiles the Jac program's IR and produces the per-call-site artifacts. A guard server owns a vLLM v1 asynchronous engine (prefix caching, chunked prefill, priority scheduling enabled) and exposes the serving and feeding interfaces; warms carry the call site's structured-output constraint so grammar compilation happens inside the warm. A thin connector, `InterceptorLLM`, is a drop-in byllm model class that intercepts at the model-call boundary: prompt construction, the tool-execution loop, and typed-output retries all run unchanged in the application process, and only raw completions are served remotely — which also guarantees the served prompt equals the native one by construction. Application integration is one line: `glob llm = InterceptorLLM(...)`. Speculation layers are independently switchable (no-speculate; no-provenance-feed) for ablation.

## 5 Evaluation

**Setup.** Qwen2.5-3B-Instruct on a single RTX 3090 (24 GB); greedy decoding; every condition on a freshly started engine; all workloads executed by the real Jac interpreter through the interceptor. Baseline in all comparisons is the same server with speculation disabled — i.e., vanilla prefix caching remains on, so every reported gain is over an already prefix-cached system. Engine-reported per-request cached-token counts serve as ground truth for attribution. We note two methodological cautions from our own runs: engine numerics vary with batch composition, so greedy trajectories can diverge across conditions (we therefore compare matched calls, and built a router-free driver for variance-sensitive measurements); and within a process, call sites share the persona prefix, so later "cold" calls are partially warmed by earlier ones.

### 5.1 Chain workload: invariant speculation

A four-call pipeline (analyze → draft → polish → count_words) with ~1.3k-token invariants per site; the last call returns a typed integer under a structured-output constraint.

| Call | No speculation | Speculation | Gain | Cached tokens (spec) |
|---|---|---|---|---|
| analyze (entry) | 269.9 ms | 241.6 ms | — | 0 |
| draft | 138.6 ms | 40.3 ms | 3.4× | 1328 |
| polish | 117.2 ms | 34.3 ms | 3.4× | 1104 |
| count_words (typed) | 304.6 ms | 36.5 ms | 8.3× | 64 |
| **End-to-end TTFT** | **830.3 ms** | **352.7 ms** | **2.35×** | |

The entry call has no predecessor and is unaffected — that is deploy-time warming's job, disabled here by design. count_words' larger gain is grammar precompilation: its structured-output automaton compiles inside the speculative warm rather than in its TTFT. With deploy-time warming added, mean per-call TTFT on this chain drops from 207.1 ms to 40.1 ms (5.2×).

### 5.2 Multi-agent RAG application

An unmodified retrieval-augmented assistant: an LLM-routed dispatcher over five specialist agents, tool-driven retrieval, multi-turn tool loops. On a representative question, comparing matched calls across conditions: the routed agent's first turn improves from 82.3 ms (cached 0) to 33.7 ms (cached 528) — 2.4× — while its second turn is unchanged (~156 ms; the tool-result turn is already covered by ordinary prefix caching) and the router entry call is unchanged. End-to-end gain on this workload is bounded by Amdahl's law at 1.08×: the entry router dominates and speculation cannot reach it — which motivates §5.3.

### 5.3 Quasi-static graph context: killing the router's cold start

The router's candidate list and choice grammar depend only on the graph, not the request. The server learns them from the first served routing prompt and exports them; feeding them to a freshly started engine before any request arrives moves the first route from 362.1 ms (cached 80 — goal line only) to 68.9 ms (cached 288 — the full stable prefix, grammar precompiled): 5.3× on precisely the call speculation cannot otherwise reach. Steady-state routing is unchanged, confirming the mechanism targets only the cold start.

### 5.4 Cache-aware routing layout

On a ten-message labeled probe (two per agent category), the stability-ordered routing prompt lifted routing accuracy from 4/10 to 8/10 and cross-request cached tokens from 80 to 288 (steady TTFT 35.1 → 31.8 ms). We attribute the accuracy change to recency — the user message becomes the final zone before generation — and report it as a favorable side effect confirmed on a small probe, not a claimed contribution.

### 5.5 Binding speculation: the content prior work writes off

Three producer calls (~470 tokens of output each) fan into one consumer whose binding zone totals ~1.4k tokens. Ablating only binding feeds (invariant speculation stays on in both conditions):

| Condition | Consumer TTFT | Consumer cached | Producers' TTFT | E2E TTFT |
|---|---|---|---|---|
| Feeds off | 149.1 ms | 96 | 32.6 / 31.0 ms | 335.9 ms |
| Feeds at producer completion | 79.2 ms | 1520 | 66.0 / 66.8 ms | 335.7 ms |
| Feeds in decode shadows | **55.9 ms (2.67×)** | 960 | 32.3 / 30.5 ms | **240.4 ms (1.40×)** |

The middle row is the negative result of §3.4: feeding at completion binds all three arguments but pollutes each following call's prefill by ~34 ms, and end-to-end gains vanish. Decode-shadow deferral keeps producers clean, accepts losing the structurally unwinnable last binding, and converts the mechanism's cache wins into latency wins. Event logs confirm dataflow firing: the first producer's result is fed to the consumer ~13 s before the consumer runs, two calls ahead. This workload is deliberately worst-case — fully sequential, zero think time; think time or concurrency widens every window the mechanism uses.

## 6 Discussion and Limitations

**Evaluation scale.** All numbers are single-request, single-GPU, on a 3B model, against a prefix-caching baseline on the same engine; SGLang [3] and larger models are pending, as are multi-request concurrency experiments — where we expect speculation to gain value, since decode-bound batches have more idle prefill capacity, but this is a hypothesis, not a result. The RAG evaluation is one case from a labeled suite; running the full suite is pending.

**Reach of provenance.** Field-sourced feeds currently read agent state as rendered in a predecessor's request, so a framework-side truncation of long field values limits feedable size; result-sourced feeds currently cover string returns and tool-call payloads. Arguments computed by opaque local code are invisible until the call arrives — the mechanism's observability equals the intercepted traffic, by design.

**Single-flow assumption.** Binding sessions and the result store assume one workflow instance per call site at a time; concurrent instances of the same workflow need a flow identifier, which the wire protocol does not yet carry.

**Retention.** We create cache entries but delegate eviction entirely to the engine's LRU. Workflow-aware retention is exactly the problem KVFlow solves [2]; the two systems are complementary — ours governs creation, theirs residency — and composing them is future work.

## 7 Related Work

**Application-structure-aware serving.** Parrot [1] exposes application-level structure (prompt structure and inter-request dependencies) to the serving system through semantic-variable API annotations. We target the same information but obtain it from the compiler of a language-integrated framework, with no annotations, and extend it to argument-value provenance, which annotation-based interfaces do not capture. SGLang [3] co-designs a programming language with its runtime and reuses prefixes via its radix cache; its prompts are, however, explicit programmer-written strings, whereas byllm's implicit, framework-constructed prompts are what make a compile-time invariant/variant split derivable. DSPy [5] compiles declarative LM pipelines but targets prompt optimization, not serving latency.

**Workflow-aware KV management.** KVFlow [2] anticipates which agent runs next using an application-supplied step graph, evicts accordingly, and prefetches already-computed KV from CPU to GPU. It manages the residency of existing cache entries under the assumption that fixed prompts have been seen (its evaluation pre-warms them) and explicitly excludes dynamic prompt content; we speculatively create entries that have never existed — including that dynamic content — from compiler-derived structure. The systems use disjoint resources (idle prefill compute vs. PCIe bandwidth) and compose. A broader 2026 wave of agentic-serving work addresses scheduling and cache lifetimes — session-centric scheduling [8], KV time-to-live for multi-turn agents [7], disaggregated conversation-level scheduling [11], copy-on-write KV for multi-LoRA agents [9], congestion-based concurrency control [10] — orthogonal to prompt-construction speculation.

**Prefix and modular reuse.** PromptCache [4] reuses attention states of prompt modules across requests; engine-level automatic prefix caching [3, 6] is our substrate and our baseline. These reuse what has been seen; we manufacture what is about to be needed.

**Terminology.** "Speculative" in LLM serving usually refers to speculative decoding (draft-then-verify token prediction) and its serving-system descendants [12]; a separate 2025 proposal uses "speculative prefill" for token-importance-based prompt pruning. Our speculation is of prompt content and cache state, not tokens; we use "proactive prefill" to avoid the collision.

## 8 Conclusion

Language-integrated agent frameworks make prompts compiler-visible objects, and that visibility is worth real latency: invariants can be warmed before first use, graph context before the first request, and argument bindings before the calls that carry them — safely, because speculation is advisory and verification is byte-exact. The design lessons generalize beyond Jac: any framework whose calls are typed and whose orchestration is programmatic exposes the same three tiers of stability; and any speculation mechanism sharing an engine with real traffic must be displaced in time, not merely demoted in priority. The immediate next steps are scale — larger models, stronger baselines, concurrency — and composition with workflow-aware retention.

## References

[1] Parrot: Efficient Serving of LLM-based Applications with Semantic Variable. OSDI 2024. https://github.com/microsoft/ParrotServe

[2] KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. NeurIPS 2025. arXiv:2507.07400

[3] Efficiently Programming Large Language Models using SGLang. arXiv:2312.07104

[4] PromptCache: modular attention-state reuse across requests. MLSys 2024.

[5] DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv:2310.03714

[6] vLLM. https://github.com/vllm-project/vllm

[7] Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live. arXiv:2511.02230

[8] SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling. arXiv:2607.08565

[9] ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache. arXiv:2604.06370

[10] CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control. arXiv:2601.22705

[11] Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving. arXiv:2606.01839

[12] StreamServe: Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving. arXiv:2604.09562

[13] The Jac Programming Language and Jaseci Stack. https://www.jac-lang.org/
