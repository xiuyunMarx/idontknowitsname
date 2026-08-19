# KVFlow vs Proactive Prefill — differentiation notes (deep-read 2026-08-18)

KVFlow: arXiv 2507.07400 (NeurIPS'25). "Efficient Prefix Caching for Accelerating
LLM-Based Multi-Agent Workflows." Read via arXiv HTML v1; all quotes verified
against the source with section references.

## Verified load-bearing facts about KVFlow

1. Its "proactive KV prefetching" ONLY moves already-computed KV tensors CPU→GPU
   ("we treat CPU memory as a secondary cache for storing the fixed prompt KV of
   evicted agents", §3.2). It never speculatively COMPUTES prefill for KV that
   does not exist yet.
2. Its evaluation pre-warms every agent's fixed prompt before measuring ("We
   first warm up the cache by executing each agent's fixed prompt multiple
   times...", §4.1) — first-ever cold start is outside its measured scope.
3. It explicitly writes off dynamic prompt content: "the dynamic parts often
   contain the input questions or instructions from users, which are less
   valuable for caching" (§2); varying suffixes always get highest eviction
   priority (§3.1).
4. The Agent Step Graph is SUPPLIED by the application layer — workflow metadata
   embedded into each HTTP request via "just-in-time substitution of the LLM
   call" (§3.3) — not derived by any compiler or static analysis. The paper
   admits fixed-prompt identification "presents a challenge" and offers user
   markup or cache-hit heuristics.

## Differentiation table

| Axis | KVFlow | Proactive Prefill (ours) |
|---|---|---|
| Problem attacked | Cache MANAGEMENT: LRU evicts an agent's KV right before reuse; PCIe reload stalls | Cache CREATION: the KV (and grammar) doesn't exist yet — first-call / cold-start TTFT |
| Core action | Move existing KV CPU→GPU in background threads | Speculatively COMPUTE new prefill (warm), and speculatively CONSTRUCT prompt content (feed binding values) |
| Cold start (fresh agent/graph/engine) | Explicitly out of scope (evaluation pre-warms all agents) | The headline target (deploy warm, topology warm, visit-ctx feed: 5.3× on first route of a fresh engine) |
| Workflow structure source | Application/orchestrator embeds Agent Step Graph metadata into each request; developer-visible | Compiler derives call-site topology from language semantics (control flow, walker traversal, receiver analysis) — zero annotations, zero app changes beyond one connector line |
| Fixed/variant split | User markup or cache-hit heuristics (admitted open challenge, §3.3) | Byte-exact compile-time construction of the invariant from typed signatures + sems — the split is an artifact of compilation, not an identification problem |
| Dynamic prompt content | "Less valuable for caching"; always evicted first | Tier-3 crown: provenance-driven binding-value speculation (arrival-order sessions + serve-time reorder) turns dynamic content into cacheable prefix |
| Argument values | Never touched | Compile-time provenance table (const/field+slice/ret); runtime dataflow firing at first-token/call-end |
| Output constraints | Not addressed | Choice/output grammar (xgrammar FSM) precompiled during warm (~300ms off first typed call) |
| Prompt layout | No control over prompt bytes | Cache-aware zone reordering (stable-first route layout; arrival-order bindings) — co-design of the prompt itself |
| Correctness model | No speculation of content → no risk | Advisory principle: all speculation is KV-only or verbatim-verified reorder; mispredict = cache miss, never a wrong prompt |
| Granularity | Per-agent fixed prompt, radix-node eviction priorities | Per-callsite three tiers: invariant / quasi-static (graph ctx) / per-request (bindings) |
| Steady-state vs first-time | Optimizes steady state under memory pressure | Optimizes first time; steady state inherits engine APC |
| Base system / eval | SGLang 0.4.4 + HiCache; Llama-3.1-8B (A10G), Qwen2.5-32B (H100); warmed workflows, high concurrency | vLLM v1 AsyncLLM APC; Qwen2.5-0.5B/3B (RTX 3090); cold-start & matched-call ablations (eval scale-up pending) |

## Reviewer attacks and responses

- **"KVFlow already anticipates the next agent."** It anticipates WHICH agent
  runs next to schedule a memory TRANSFER of KV it already has. We anticipate
  which call runs next AND what its prompt will contain, to COMPUTE KV that has
  never existed. Different resource (PCIe bandwidth vs idle prefill compute),
  different failure mode (transfer stall vs cold prefill), disjoint measured
  regimes (their eval pre-warms; ours measures the pre-warm-less world).
- **"Workflow-aware = same idea."** Their workflow knowledge is supplied by the
  application at runtime; ours is derived by the compiler at build time. Their
  own paper names fixed-prompt identification an open challenge — our compiler
  makes that challenge vanish by construction. This is the language-integration
  thesis in one sentence.
- **"Dynamic parts don't matter" (their §2 claim).** Direct quotable foil for
  tier-3: what they write off as uncacheable is exactly what provenance
  speculation converts into cacheable prefix. (Action item: our big-binding
  workload must make this gain exceed noise, or the foil cuts back at us.)

## What KVFlow has that we lack (honest; borrow or cite)

- Workflow-aware EVICTION (steps-to-execution priorities at radix-node level) —
  we have no eviction policy at all; under memory pressure our warms are at the
  engine LRU's mercy. Their policy is complementary and citable as the
  steady-state counterpart.
- CPU secondary cache + overlapped PCIe prefetch + status-aware scheduling —
  engineering we don't have; orthogonal, co-applicable.
- High-concurrency evaluation (64 concurrent workflows) and bigger models —
  their eval strengths are precisely our current eval gaps.

## Positioning paragraph (draft for related work)

KVFlow keeps an agent's already-computed prefix KV alive and close (workflow-
aware eviction, CPU→GPU prefetch), assuming the workflow graph is handed to the
serving system by the application and that fixed prompts have been seen before.
Proactive Prefill is the complement: it creates KV (and decoding grammars)
before first use, for prompts the engine has never served — and it needs no
handed-down graph, because the compiler of a language-integrated agent framework
derives the invariant/variant split, the call-site topology, and the argument
provenance from the program itself. The two compose: KVFlow governs the steady-
state residency of what we prewarm.

## Complementarity / co-application

Non-overlapping resources: our speculation spends idle prefill FLOPs during
decode; theirs spends PCIe bandwidth. A combined system: compiler-driven
creation (ours) + workflow-aware retention (theirs). Worth one sentence in the
paper; possibly a joint experiment later.
