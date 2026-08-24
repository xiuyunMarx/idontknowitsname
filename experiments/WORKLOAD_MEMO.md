# Cold-start memorandum: system, algorithms, workload, results

The system design and algorithms behind compile-time-guided speculative
prefill, why each tenant program can gain from it, how the multi-tenant
workload is composed, and what the cold-start experiment
(`coldstart_experiment.py`, Qwen2.5-3B, RTX 3090) measured.

## System design

A Jac agent program's `by llm()` calls are intercepted client-side: jaclang's
`InterceptorLLM` replaces the model backend and ships every call — and every
`visit ... by llm()` graph-routing decision — over a length-prefixed JSON TCP
protocol. The wire carries only callsite keys (`Owner.name`, `name`,
`__visit@file:line`) and argument reprs; tools execute on the client, with the
server driving the ReAct loop through `tool_call`/`tool_result` frames and
typed-parse failures returning as `reject` frames.

The server side is one `GuardServer` owning a single vLLM `AsyncLLM` engine
(prefix caching, priority scheduling). Each program (tenant) gets a dedicated
`InterceptorLLMBackend` listener; all backends feed one asyncio queue, and all
tenants share the engine and its KV cache. At program registration a static
pass (`ProgramTopology`) parses the Jac source once. Crucially, the server
constructs every prompt end-to-end from the static pass plus the wire's arg
reprs — the canonical prompt bytes live server-side, which is what makes
"warm path == serve path" achievable byte-for-byte.

Speculation runs as one background task (`monitor_idle`, 10 ms tick) that
spends engine-idle time prefilling the bytes the compiler can prove will (or
likely will) be requested next. Speculative prefills are submitted at low
scheduler priority; real calls always preempt.

## Algorithms

### Static program analysis

From the Jac UniIR the pass extracts, per program:

- **Decls and callsites.** Every byLLM decl (params, types, semstrings, tools,
  call params, return type) and every callsite, keyed per source location.
  `visit` statements become pseudo-decls keyed `__visit@file:line`.
- **May-run-next topology**, per callsite: which byLLM callsites can execute
  next, following statement order, branches, loops (self-edges pruned), and —
  for walkers — which node entry abilities each `visit` can fire.
- **Parameter provenance**, per (callsite, parameter): a small lattice —
  `const` (literal or default, known at compile time), `field`
  (walker/node state, scope-aware), `ret` / `ret_item` (the return value, or
  one element of the returned list, of another callsite), `call` (client-side
  computation), `unknown`. Conflicting writers across sites degrade to
  `unknown`.
- **Prompt layout.** Each callsite's prompt is an invariant prefix (system
  persona, signature + schema lines, tool-protocol block) followed by a
  bindings zone. Bindings are appended in bind order and `bound_args`
  insertion order *is* warm byte order; at serve time bound parameters render
  first, with the request's value winning in place. This makes every warm
  state a byte-exact prefix of some serve-time prompt.

### Idle-window speculation loop

```
every δ = 10 ms:
  if no in-flight calls or not room(1): continue tick
  S ← all in-flight call states, all tenants        # newest-only under ablation
  # 1. certain bytes first: each call's own pending tool turn
  for s in S, newest first:
      if s has a pending tool turn not yet warmed:
          c ← tokens(transcript + assistant_text + result_stub) − tokens(transcript)
          if not room(c): end tick
          prefill(turn_prompt, cost = c, priority = low)
  # 2. successors, round-robin across calls so no tenant starves
  for each s: pending(s) ← successors(s) minus already-warmed
      successors(s) = probe-ranked route plan       if s is a visit call
                      sites of may-run-next(s.site) otherwise
  while any pending, one site per call per round, newest call first:
      bind ready const params into site
      p ← render(site warm prefix)
      c ← max(0, tokens(p) − warmed_tokens[site])   # only the uncached delta
      if not room(c): end tick
      prefill(p, cost = c, priority = low)
      warmed_tokens[site] ← tokens(p)

on producer return, for every (consumer, param) with ret/ret_item provenance:
      bind repr of the (first element of the) value into consumer sites,
      overwriting in place; invalidate their warmed_tokens and dedup marks

room(c) ≡ engine empty ∨ inflight_prefill_tokens + c ≤ B[decode_concurrency]
```

Real calls and speculative prefills charge the same in-flight token pool; a
speculative prefill is admitted only for its estimated uncached delta, so a
nearly-resident extension (a tool turn over a just-computed transcript) is
almost free while a cold invariant is charged in full.

### Routing speculation (visit calls)

The router's own prompt is mostly graph-dependent, so the gain is in ranking
its successors. Given the live routing request: parse the candidates zone from
the newest user message that carries it (a typed-retry appends a feedback turn
after it; the retry state remains speculable); if more than one candidate,
render the same messages with `enable_thinking=False` — which only appends an
empty think block after the generation prompt, so the real request stays an
exact prefix of the probe and the probe's prompt is already cached — and issue
a one-token greedy generation with top-k logprobs at low priority. Each
candidate handle is scored by the highest logprob among first tokens
prefix-compatible with it; candidates are ordered by score, and the plan (each
candidate's first byLLM callsites, in that order) is computed once per call
and cached. The probe predicts *this* model's routing choice, which is exactly
the distribution the warm order should follow.

### Safe-budget profiling (token accounting)

Offline, per decode concurrency d: measure baseline median TBT with d decode
streams; add sustained prefill workers of ~C tokens one at a time; the largest
count k whose median TBT stays within (1+slack)·baseline gives budget
B[d] = k·C. If k = 0, halve the worker length until a single worker fits and
take its token count — token granularity often admits a budget where task
granularity reports zero. The table is persisted per model and loaded at boot.
(The cold-start experiment runs with the budget disabled: speculation only in
true idle.)

### Correctness

1. **Serve bytes always win.** A warm binding renders in place, but the
   request's value overrides by name at serve time. A wrong speculation —
   wrong branch, wrong route, stale value — costs only wasted cache.
2. **Warm bytes are exact prefixes.** For `ret`-bound parameters the server
   mirrors the client's typed conversion chain (wire output → `json.loads` →
   `json_to_instance`) byte-for-byte, so the warmed repr equals the repr the
   client will send. Values not derivable server-side (user objects/enums,
   payloads the client itself rejects) are skipped, never guessed.
3. **Speculation is best-effort and isolated.** A failed tick logs and
   continues; low scheduler priority plus the profiled token budget bound the
   interference with real decodes.

Known limitations: a callsite's binding state is shared across concurrent
calls to the same decl (last writer wins); `ret_item` warms only the first
loop iteration (self-edges are pruned); a parameter with multiple branch
writers currently resolves to one writer.

## Why speculation gains anything

A byLLM program alternates between engine work (prefill + decode) and client
work (tool execution, Jac interpretation, network turnarounds). During client
work the engine is idle. The static compiler knows three things about that
moment that a generic serving stack does not:

1. **What may run next** — the per-callsite may-run-next topology, including
   branch fan-outs and, for `visit ... by llm()`, the candidate set (ranked
   live by a 1-token logprob probe riding the router's own cached prefix).
2. **Which bytes of the next prompt are already decided** — the invariant
   prompt (system persona, signature, schema, tool protocol) plus every
   parameter whose provenance is derivable server-side: `const` at compile
   time, `ret`/`ret_item` the moment the producer call returns.
3. **That warming is free of correctness risk** — the client's request bytes
   always win at serve time; a wrong warm binding only wastes cache.

Prefilling those bytes inside the idle window converts the next call's
serve-time prefill into a KV-cache hit. The gain per call is bounded by (a)
how many of the next prompt's tokens are decided early, and (b) whether an
idle window exists to spend. Both vary per program — hence the workload below.

## How the multi-tenant workload is built

Seven programs, one `InterceptorLLMBackend` each (ports 8964-8970), one shared
vLLM engine. Their invariants are disjoint (different decls, tools, sems); the
only cross-program shared bytes are the one-sentence system personas. Each
trial boots a fresh server (empty KV cache), every tenant runs exactly one
round, staggered 0-3 s (seeded per trial, identical across cells), then the
server dies. `single-*` cells run the same seven tenants strictly one at a
time on an equally fresh server: the same cold bytes with the idle windows
undisturbed. `off` = `--no-prefill` (vLLM prefix caching and priority
scheduling stay on); `spec` = global-idle speculation, no token budget.

Concurrency changes exactly one input to the mechanism: tenant A's client-side
window is no longer engine-idle time, because tenant B may be prefilling or
decoding in it. On a 3B model this hardware sustains zero concurrent
speculative prefill within a 10% TBT slack, so `spec` speculates only in true
idle gaps — the multi/single comparison measures how much of the single-task
gain survives when seven tenants compete for those gaps.

## Per-tenant memoranda

### research (8964) — baseline chain
`investigate` (2 tools, ReAct ≤4) → `draft_answer(question, evidence)` →
`audit_answer(answer)`; 0.2 s between calls, tools sleep 0.25-0.35 s.
Windows: tool waits inside `investigate`, then inter-call gaps. Warmable:
`draft_answer`/`audit_answer` invariants during the tool window; `evidence`
and `answer` are `ret`-bound the moment their producers return, extending the
warm prefix inside the 0.2 s gap; `question` is client-only (`call` kind) and
serves cold. Tool-turn warming covers `investigate`'s own next ReAct turn.

### operations (8965) — two tool loops + a const binding
`collect_metrics` (tools) → `analyze_metrics` (tools) → `write_report(request,
analysis, audience="SRE lead")`. Two ReAct loops double the tool windows. The
`audience` default is a compile-time `const`: `write_report`'s warm prefix
includes a binding before any call has even started.

### route (8966) — visit routing, fan-out 3
`visit [-->] by llm()` over Billing/Tech/Security desks → the chosen desk's
`resolve_*` → `summarize`. The router's own prompt is mostly graph-dependent
(only `Goal:` is compile-time), so the gain is not the router — it is the
window *after* the router answers: the probe ranks the three desks by the
model's own next-token logprobs and the desks' first callsites are warmed in
that order. `visitor.ticket` rides the traversal; `self.policy` belongs to the
not-yet-chosen desk and is withheld (scope-aware readiness). `summarize.answer`
is `ret`-bound from whichever desk ran.

### support_email (8967) — long invariant
`classify_email` → `compose_reply` (3 tools, ~1.5k-char style-guide sem) →
`polish_reply`. The style guide makes `compose_reply`'s invariant the largest
warmable byte mass in the workload, all of it decided at compile time. This is
the compiler's best case: a large prompt whose serve-time bytes are almost
entirely predictable, sitting behind a cheap classify call that provides the
window.

### pipeline (8968) — deep chain
Six small ret-chained calls (`extract` → `validate` → `classify` →
`summarize` → `translate` → `format`), 0.10-0.15 s gaps. Each call's gain is
small (short prompts), but every return immediately makes the next site's
binding derivable, and the transitive topology lets one window warm several
downstream sites. Measures how gains compound over depth.

### moderation (8969) — branch fan-out
`classify_content` → one of `escalate_case` (tools) / `draft_warning` /
`archive_note` → `log_decision`. After `classify_content` returns, all three
branch invariants are warmed with `category` bound — two of the three are
wasted by construction (bounded waste, correctness unaffected). Measures
speculation under control-flow uncertainty. Known imprecision: `decision` has
three branch writers but provenance resolves to `escalate_case` only, so
`log_decision` warms a binding only on that branch.

### dispatch (8970) — visit routing, fan-out 5
Same shape as route with five desks and longer playbook fields; the database
desk carries a tool. With five candidates, warm order matters more: the probe
ranking decides which desks' invariants are resident before the router's
choice lands.

## Measured results (8 trials per cell)

Full tables in `results/coldstart-3b/summary.md`. Protocol: fresh server and
empty KV cache per trial, one round per tenant, paired inputs and stagger
across cells; `off` keeps vLLM prefix caching on.

### Headline: per-tenant ΣTTFT speedup (p50 across trials)

| tenant | single-task | multi-tenant | retained |
|---|---|---|---|
| support_email | **1.72x** | 1.06x | 8% |
| moderation | **1.50x** | 1.02x | 5% |
| operations | 1.26x | 1.07x | 28% |
| pipeline | 1.21x | 1.01x | 3% |
| route | 1.19x | 0.98x | -13% |
| dispatch | 1.03x | 1.01x | 52% |
| research | 1.03x | 0.99x | -22% |
| **all** | **1.17x** | 1.07x | — |

Mean TBT: single 9.15 → 9.14 ms, multi 10.56 → 10.99 ms — idle-gated
speculation costs decode nothing. Cache hit over all serves: single 54.1% →
75.4%; multi 54.2% → 56.7%.

### Single-task: the mechanism at work

Per-callsite first-serve medians (TTFT ms / cached tokens), off → spec:

- `support_email:compose_reply` 71/112 → **21/656** — the 1.5k-char invariant
  fully resident before the call arrives; the workload's best case and the
  source of the 1.72x.
- `operations:analyze_metrics` 41/112 → 15/400; `moderation:escalate_case`
  44/112 → 19/352; `route:resolve_billing` 40/112 → 17/336;
  `dispatch:resolve_database` 39/112 → 19/336 — tool-carrying successors
  warmed during the preceding call's tool windows.
- Every tenant's *first* callsite (`classify_email`, `extract_fields`, …)
  shows no gain: speculation is driven by in-flight calls, and nothing is in
  flight before a program's first call. See "Register-time warming" below.

### Multi-tenant: the idle gate never opens in a cold-start burst

Under seven concurrent cold starts the multi-spec cached-token counts equal
multi-off almost everywhere (48/112 medians unchanged): with a 3B model this
hardware admits zero speculative prefill within the TBT slack, so `spec`
speculates only when the engine is empty — and during the burst it never is.
The 1.07x overall comes from windows after the burst thins out.

Together with the 0.5B budget experiment (`results/20260823-fix12-v2`), the
two runs bracket the concurrency problem precisely: a token budget *does*
open the gate under load (+51% speculative volume) but its collisions with
arriving real prefills cost +15% TBT and 230-414 ms tail TTFTs; the pure idle
gate is harmless but stays shut exactly when the bytes are needed most.

### What this points at next

1. **Register-time warming.** Entry-callsite invariants are compile-time
   bytes; every program is registered at boot, before any call. Warming them
   during boot idle would cover the burst's first calls — the one part of the
   cold start that in-flight-driven speculation structurally cannot reach,
   and in the multi cell it is most of the deficit.
2. **Collision-bounded budget speculation.** Cap each speculative prefill
   chunk (e.g. ≤128 tokens, resuming across ticks) so an in-flight chunk can
   only delay an arriving real prefill by a bounded amount, then re-enable
   the token budget under concurrency.
