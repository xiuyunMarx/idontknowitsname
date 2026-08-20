# PBKV & CacheScout deep-read differentiation (2026-08-19)

Two 2026 papers surfaced by the novelty sweep; both closer in title than KVFlow, one
closer in substance. PBKV quotes were extracted from the arXiv HTML via a single
fetch pass — spot-check against the PDF before citing. **CacheScout section
re-verified 2026-08-19 (three targeted HTML passes; all quotes below confirmed
verbatim)** — PDF-render spot-check of equations still advisable before camera-ready.

**Threat level: CacheScout HIGH (overlaps tier-1 + idle scheduling), PBKV MODERATE
(different action on KV, but owns "prediction-based" framing for dynamic workflows).**

## PBKV — arXiv:2605.06472 (2026-05-07)

"Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache
Management." SJTU / Macquarie / HKUST / Wuhan U. et al.

- Problem: LRU evicts cache entries that go idle during tool calls/agent
  transitions but will be reused — "cache reuse is determined by the workflow's
  structure rather than temporal locality."
- Knowledge source: **learned predictor** — GraphSAGE over a global call graph +
  attention over the workflow prefix + the LLM's prefill hidden state; "trained on
  offline invocation traces with cross-entropy loss"; predicts the next K agent
  invocations. Stated limitation: "The predictor needs to be trained on a specific
  workload."
- Action: manages **EXISTING** KV only — survival-scored eviction + prefetch of
  already-computed blocks from CPU/disk into GPU via SGLang HiCache ("proactively
  load likely-to-be-reused cache nodes into GPU memory before they are hit").
  No speculative prefill of prompts that have not been issued.
- Content speculation: none — "even under identical user prompts, the stochastic
  decoding of LLMs makes the workflows diverge" (they conclude cross-workflow
  private-cache reuse is infeasible; we make the *deterministic* part of it work
  by construction instead of giving up).
- Scheduling: prefetch restricted to "otherwise-idle GPU space and PCIe
  bandwidth", pure-decode batches only, bandwidth-bounded budget. No preemption.
- Eval: Qwen3-14B/32B on 8×A6000; HoVer+LangChain, SWE-bench+AutoGen,
  FinanceBench+CrewAI; 1.85× latency / 2.03× TTFT / 2.55× hit rate vs LRU,
  1.26× vs KVFlow.

## CacheScout — arXiv:2608.14624 (2026-07-16)

"Learning Agent Execution for KV-Cache Management in Agentic Serving." UCSC / UW /
UChicago (LMCache-adjacent authors: Kuntai Du, Yuhan Liu, Junchen Jiang).

- Problem: recurring "agent anchors" (system prompt, tool definitions, few-shot)
  are 53–62% of prompt tokens and 49–173× more reused than session blocks, yet
  LRU evicts them; the engine "cannot tell whether a block belongs to the Travel
  Agent, the Hotel Agent, or the Restaurant Agent."
- Knowledge source: **online first-order Markov chain** over observed dispatches
  (prompt-prefix fingerprint → transition counters); "No offline training or
  predefined workflow graph is required"; 76–86% top-1 next-agent accuracy
  **after ~50 observed dispatches**.
- Action — THE OVERLAP: survival-guided eviction of anchors, **plus background
  prefetch that CREATES KV**: "issues a lightweight warmup request containing
  only the predicted agent's reusable anchor, including its system prompt, tool
  definitions, and a minimal user prompt", executed "during idle periods between
  requests... off the serving critical path." This is invariant warming + idle
  scheduling — our tier-1 mechanism, published 2026-07.
- Content speculation: **none beyond the fixed anchor** — no argument/binding
  values, no graph context, no grammar precompile; anchors are developer-authored
  fixed prefixes, not derived by analysis.
- Scheduling: idle-window execution + a *predictability* gate (R ≥ R_min disables
  prefetch under random routing) + rate limit. Gates on prediction quality, NOT
  on engine interference; no calibration, no preemption of in-flight warmups.
- Eval: Llama-3.1-8B on 8×RTX PRO 6000, Qwen3-235B-A22B on 4×H200; AutoGen
  six-agent supervisor over GSM8K/MT-Bench/GAIA/SWE-bench; hit rate +10–18pp,
  mean TTFT −18–45%, throughput +19–57%; vs vLLM and Continuum (TTL pinning).
- Stated limitation: "Gains diminish when execution approaches random routing."

Hand-verification pass (2026-08-19) added the following confirmed facts:
- Predictor: first-order Markov chain over agent transitions, agent identity from
  the "prompt-prefix fingerprint"; transition matrix `P_ij = (C_ij+eps)/sum_k(C_ik+eps)`;
  prediction = argmax_j P_ij. **Top-1 only — never multi-step, never multiple
  candidates**: "it predicts the next agent from the same transition matrix and
  warms the corresponding anchor between requests."
- Warmup content confirmed anchor-only: "an inference request containing only the
  predicted agent's anchor with max_tokens=1"; the user zone is "a minimal user
  prompt" — a SYNTHETIC prompt. They assert the blocks are "constructed exactly as
  they would be during a real agent invocation" but the paper never discusses
  divergence between the synthetic warmup prefix and real served prompts (no
  byte-verification anywhere; serving is untouched: "no changes to either
  component", requests go through "the standard serving API").
- Scheduling gaps confirmed: warmup priority in the engine scheduler is not
  disclosed; NO mechanism for a real request arriving mid-warmup (no abort, no
  preemption, no quantified interference — "off the critical path" is asserted,
  not enforced); the rate limit's value/mechanism is never specified; the gate is
  R >= R_min where R = 1 − H(A_{t+1}|A_t)/H(A_{t+1}) (prediction quality, NOT
  engine interference), and the R_min value used is not disclosed.
- Reuse-disparity numbers: anchor blocks reused 49–173× on average vs 12–15× for
  session-history blocks ("agent anchors receive 4–13× more reuse than
  session-specific history"); anchors are 53–62% of all prompt tokens — i.e.
  **their own measurement of their coverage ceiling**.
- No dedicated Limitations/Future Work section; no mention of workflow structure
  from program analysis, argument prefill, grammar/structured output, or decode
  interference. Eval: Llama-3.1-8B on 8×RTX PRO 6000 (96GB), Qwen3-235B-A22B-FP8
  scale-out; six-agent AutoGen supervisor on GSM8K/MT-Bench/GAIA/SWE-bench;
  0.2–50 sessions/s sweep. Compared against vLLM and Continuum; cites KVFlow,
  Parrot, SAGA, TokenCake, Autellix, KVCOMM.
- arXiv v1 submitted 2026-07-16 (the 08-14 sighting was later circulation, not a
  new version — treat priority date as July).

## Differentiation table (ours = Proactive Prefill)

| Axis | PBKV | CacheScout | Ours |
|---|---|---|---|
| Knowledge source | offline-trained GraphSAGE (per workload) | online Markov chain (needs ~50 dispatches) | **compiler** (UniIR static analysis): correct from request #1, zero training/observation |
| What is known | next-K agent IDs | next agent ID | full may-happen topology (BFS-layered reachable set) + **binding provenance** (which value feeds which future param) + visit candidate sets |
| Action on KV | move/evict EXISTING blocks | evict + CREATE anchor KV via warmup | CREATE the **longest knowable prefix**: invariant + graph ctx + **speculated binding VALUES** + xgrammar precompile |
| Prompt-content speculation | none (declared infeasible) | none (fixed anchor only) | **yes** — dataflow ret/field feeds; anchors are 53–62% of tokens (their number) and we additionally cover most of the rest (92–100% byte-verified serve hits) |
| Serve-path integration | none | none ("no changes to either component") | binding-zone reorder + verified arrival-order replay — served prompt byte-identical to native, fed prefix made a strict prefix |
| Idle scheduling | pure-decode-batch budget | idle windows + predictability gate + rate limit | **calibrated** TBT-interference budget (per device+model sweep) + 30ms grace + real-arrival **preemption** of in-flight warms |
| First request / cold start | needs trained predictor | blind for first ~50 dispatches | full mechanism live on request #1 (our showcase is literally the first request) |
| Framing | learned prediction for dynamic workflows | framework-agnostic runtime layer | language-integrated: MTP compiles prompts, so serving *knows* instead of *predicts* |

## What this changes for our paper

1. **Tier-1 anchor warming in idle windows is no longer novel on its own** —
   CacheScout owns that claim as of 2026-07. Do not lead with it; cite it as the
   strongest evidence that the direction matters, then show what a compiler makes
   possible that runtime learning cannot: (a) first-request correctness (their
   50-dispatch warm-up window is exactly our showcase regime), (b) the exact
   reachable SET layered by distance rather than top-1 next-agent, (c) the other
   ~40% of prompt tokens — bindings — which both papers explicitly leave cold.
2. **Lead with tiers 2–3** (quasi-static visit ctx + value speculation via
   provenance) and the **verified-advisory serve path**; that combination remains
   unclaimed in all four related systems (Parrot, KVFlow, PBKV, CacheScout).
3. **Scheduler positioning**: theirs gate on prediction quality; ours gates on
   measured engine interference and preempts. Complementary — say so, and keep the
   calibration + preemption ablations (they are the only defense if a reviewer
   says "CacheScout already does idle warmup").
4. **Use their numbers for our motivation**: anchors = 53–62% of prompt tokens
   (CacheScout) matches our ~95%-invariant claim direction but shows what
   annotation-free static analysis adds on top; PBKV's "cross-workflow reuse is
   generically infeasible" is the assumption our provenance mechanism refutes for
   the deterministic slice of the prompt.
5. **Urgency**: two groups (one LMCache-adjacent) shipped adjacent systems in
   May and July 2026. Value speculation is the remaining moat; arXiv now.
