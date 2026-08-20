# Multi-Intent Cold-Start TTFT

**Proactive Prefill — language-integrated speculative prefill for Jac byllm calls**
Experiment date 2026-08-20 · Qwen3-4B · RTX 3090 24GB · vLLM v1 (AsyncLLM, APC + chunked prefill + priority scheduling)

---

## Summary

- On a 4-desk multi-intent agent, **per-request ΣTTFT drops from 470.1 ms to 264.3 ms (median over 20 cases) — 1.78×**, with the run-to-run spread also halving (sd 29.7 → 15.1 ms).
- The gain is concentrated exactly where the theory predicts. Calls that are preceded by a tool-execution window improve **3.01×** (6069.8 → 2018.7 ms summed over all matched calls); calls with no preceding window sit at **0.99×** — speculation costs them nothing.
- The largest single win is the typed `finalize` call of every desk: **3.85×–5.18×**, consistently across all four desks, because its entire ~1.4k-token prompt is reconstructed from compile-time argument provenance before the call is issued.
- Causality is not inferred from the latency delta alone: the engine's own `cached_tokens` counter rises from 48 to 592–1408 exactly in step with the prefix the provenance table predicts.
- The measurement is deliberately **conservative**: xgrammar grammar pre-compilation — a real benefit of the system worth ~285 ms on the first typed call — was warmed away in *both* conditions before measuring, so the numbers above are pure KV-prefill effect.

---

## 1. What was measured

The benchmark app is `evaluations/multi_intent/main.jac`: a customer-service router with four specialist desks.

```
normalize ──▶ route (visit [-->] by llm, select=1, 4 candidates)
                 │
                 ├─▶ MathDesk  : analyze ─[RAG]─▶ solve     ─[RAG]─▶ finalize : MathBrief
                 ├─▶ CodeDesk  : analyze ─[RAG]─▶ implement ─[RAG]─▶ finalize : CodeBrief
                 ├─▶ DocDesk   : analyze ─[RAG]─▶ answer    ─[RAG]─▶ finalize : DocBrief
                 └─▶ WriteDesk : analyze ─[RAG]─▶ draft     ─[RAG]─▶ finalize : WriteBrief
```

14 byllm call sites total. Each request executes 5 LLM calls. `[RAG]` is a real FAISS + cross-encoder retrieval over a local corpus (~350–500 ms including a simulated 0.3 s tool latency) — these are the idle windows the scheduler is designed to exploit.

The compiler-derived tables the system runs on, extracted with no engine and no annotations:

| binding | provenance | fed? |
|---|---|---|
| `normalize.request` | `field(self)` | no — entry call, nothing precedes it |
| `*Desk.analyze.<ticket>` | `field(visitor.profile)` | yes, at the route's first token |
| `*Desk.<middle>.analysis` | `ret(analyze)` | yes, dataflow |
| `*Desk.<middle>.<rag>` | unknown (tool output) | no — but it is **last** in declaration order, so the fed prefix stays contiguous |
| `*Desk.finalize.analysis` | `ret(analyze)` | yes |
| `*Desk.finalize.<solution>` | `ret(middle)` | yes |
| `*Desk.finalize.<rag>` | unknown (tool output) | no — again last in declaration order |

**"Cold start" here means:** every one of the 20 requests begins with an empty KV cache and empty speculation state. This is enforced by a new `POST /reset` endpoint on the guard server, which drops the engine's prefix cache and every trace a previous run left behind (warm dedup hashes, recorded bindings, stored results, the speculation queue, the workflow position, and all statistics). One booted process therefore serves 20 genuinely cold runs, instead of requiring a process restart per case.

## 2. Protocol

Both conditions run the identical client program, identical prompts, and greedy decoding.

| | baseline | speculation |
|---|---|---|
| server flag | `--no-speculate` | (default) |
| prefix caching (APC) | on | on |
| topology warming, provenance feeds | off | on |
| `--probe` (choice-distribution speculator) | off | off |
| `--deploy-time-prewarm` | off | off |
| client `JAC_ROUTE_CACHE_LAYOUT` | 1 | 1 |

Per condition: boot a fresh server → run **4 dummy cases, one per desk** → then, for each of the 20 cases, `POST /reset` → one `python3 -m jaclang run main.jac` → capture `/stats`.

**Why the 4 dummy cases matter.** vLLM's xgrammar compiled-grammar cache lives in the EngineCore process and is *not* cleared by `/reset`, so the first request in a server's life that needs a given JSON schema pays its compilation on the critical path. The four desks have four different `finalize` return types, so four dummies (one per desk) precompile every schema in the app. Without this, the first case of each condition carries a ~300 ms outlier. Measured directly: the baseline's dummy run recorded ΣTTFT 807.9 ms; the first *measured* baseline case, with grammars already compiled, recorded 503.7 ms.

This choice is conservative — warming grammars away **removes a genuine advantage of our system** (its warms attach the output schema so xgrammar compiles off the critical path) from the reported numbers. Everything below is pure KV prefill.

## 3. Results

### 3.1 Headline

| metric | baseline | speculation | gain |
|---|---|---|---|
| ΣTTFT per request (median of 20) | 470.1 ms | 264.3 ms | **1.78×** |
| ΣTTFT per request (mean) | 466.6 ms | 265.3 ms | 1.76× |
| ΣTTFT standard deviation | 29.7 ms | 15.1 ms | — |
| all matched calls, summed | 9332.5 ms | 5305.5 ms | 1.76× |

### 3.2 The gain splits cleanly by whether a tool window precedes the call

| call class | baseline | speculation | gain |
|---|---|---|---|
| preceded by a RAG tool window (`middle`, `finalize`) | 6069.8 ms | 2018.7 ms | **3.01×** |
| no preceding window (`normalize`, route, `analyze`) | 3262.8 ms | 3286.8 ms | 0.99× |

Per desk, on the addressable calls only: MathDesk **2.98×**, CodeDesk **3.28×**, DocDesk **2.73×**, WriteDesk **3.03×**.

The 0.99× on the second row is a result in its own right: 15 speculative prefills and 7 provenance feeds execute during every request, and they impose **no measurable tax** on the calls they cannot help. That is the admission gate (calibrated busy budget, 30 ms idle-grace hysteresis, preemption on real arrivals) doing its job.

### 3.3 Per call site (median over matched pairs)

| call site | n | baseline | speculation | gain | cached tokens (base → spec) |
|---|---|---|---|---|---|
| `normalize` (entry) | 20 | 37.3 ms | 37.6 ms | 0.99× | 0 → 0 |
| `Desk.triage.visit@84` (route) | 20 | 65.8 ms | 65.9 ms | 1.00× | 0 → 0 |
| `MathDesk.analyze` | 8 | 65.8 ms | 66.1 ms | 1.00× | 48 → 48 |
| `CodeDesk.analyze` | 4 | 64.0 ms | 64.3 ms | 1.00× | 48 → 48 |
| `DocDesk.analyze` | 3 | 52.3 ms | 52.3 ms | 1.00× | 48 → 48 |
| `WriteDesk.analyze` | 5 | 52.3 ms | 52.5 ms | 1.00× | 48 → 48 |
| `DocDesk.answer` | 3 | 102.3 ms | 62.3 ms | **1.64×** | 48 → 640 |
| `MathDesk.solve` | 8 | 119.8 ms | 67.8 ms | **1.77×** | 48 → 800 |
| `WriteDesk.draft` | 5 | 102.0 ms | 50.7 ms | **2.01×** | 48 → 592 |
| `CodeDesk.implement` | 4 | 117.9 ms | 49.7 ms | **2.37×** | 48 → 752 |
| `WriteDesk.finalize` | 5 | 171.1 ms | 41.9 ms | **4.09×** | 48 → 1136 |
| `CodeDesk.finalize` | 4 | 187.0 ms | 42.6 ms | **4.39×** | 48 → 1232 |
| `DocDesk.finalize` | 3 | 185.0 ms | 41.5 ms | **4.46×** | 48 → 1152 |
| `MathDesk.finalize` | 8 | 216.3 ms | 42.7 ms | **5.07×** | 48 → 1408 |

The baseline's uniform `cached = 48` is the system persona shared between call sites — natural prefix-cache reuse. The baseline is genuinely APC-enabled; the comparison is not against a no-cache strawman.

## 4. Per-case breakdown

`middle` is `solve` (math) / `implement` (code) / `answer` (doc) / `draft` (write). Cells are `baseline → speculation` in milliseconds.

| sid | task | desk | normalize | route | analyze | middle | finalize | ΣTTFT |
|---|---|---|---|---|---|---|---|---|
| s01 | bake-sale revenue over 3 days | math | 37→38 (0.97×) | 65→66 (0.99×) | 65→66 (0.99×) | 119→64 (1.85×) | 216→43 (5.03×) | 504→278 (**1.81×**) |
| s02 | `second_largest(nums)` | code | 37→38 (0.96×) | 66→66 (1.00×) | 64→64 (1.00×) | 117→42 (2.77×) | 186→41 (4.52×) | 470→252 (**1.87×**) |
| s03 | Growth plan data retention | doc | 37→39 (0.96×) | 65→65 (1.00×) | 52→52 (1.01×) | 102→63 (1.62×) | 187→41 (4.52×) | 444→260 (**1.71×**) |
| s04 | extension-request email to sponsor | write | 38→38 (0.99×) | 66→66 (1.00×) | 52→52 (1.01×) | 102→55 (1.86×) | 185→42 (4.40×) | 444→253 (**1.75×**) |
| s05 | train travel time, two legs | math | 38→37 (1.03×) | 65→66 (1.00×) | 66→66 (1.00×) | 119→54 (2.21×) | 217→43 (5.09×) | 505→265 (**1.91×**) |
| s06 | run-length encoding `rle(s)` | code | 39→49 (0.79×) | 66→66 (1.00×) | 64→64 (1.00×) | 118→46 (2.55×) | 188→44 (4.29×) | 474→269 (**1.76×**) |
| s07 | overage cost per extra 1M events | math *(gold doc)* | 35→36 (0.98×) | 66→66 (1.00×) | 66→66 (0.99×) | 121→79 (1.52×) | 216→43 (5.06×) | 503→290 (**1.74×**) |
| s08 | rewrite note in formal register | write | 36→36 (0.99×) | 65→67 (0.98×) | 52→53 (0.98×) | 102→54 (1.88×) | 169→43 (3.94×) | 425→253 (**1.68×**) |
| s09 | Maya's savings after a purchase | math | 37→40 (0.93×) | 66→66 (0.99×) | 66→66 (1.00×) | 120→62 (1.93×) | 217→42 (5.17×) | 505→276 (**1.83×**) |
| s10 | `merge_sorted(a, b)` | code | 37→38 (0.97×) | 66→65 (1.01×) | 64→64 (0.99×) | 118→55 (2.15×) | 185→48 (3.85×) | 470→270 (**1.74×**) |
| s11 | which plan includes SAML SSO | doc | 38→36 (1.04×) | 66→65 (1.01×) | 53→52 (1.00×) | 103→60 (1.72×) | 185→41 (4.49×) | 444→255 (**1.74×**) |
| s12 | 150-word headphone description | write | 39→37 (1.03×) | 68→66 (1.03×) | 52→53 (0.99×) | 102→49 (2.08×) | 171→41 (4.17×) | 431→246 (**1.76×**) |
| s13 | seedlings after 10% frost loss | math | 38→40 (0.95×) | 67→67 (1.00×) | 66→66 (1.00×) | 120→74 (1.62×) | 209→43 (4.87×) | 499→289 (**1.73×**) |
| s14 | `is_balanced(s)` bracket check | code | 38→38 (1.00×) | 66→67 (0.99×) | 64→65 (0.99×) | 118→53 (2.22×) | 189→41 (4.55×) | 474→264 (**1.80×**) |
| s15 | encryption at rest | doc | 38→37 (1.04×) | 65→65 (1.00×) | 52→52 (0.99×) | 102→62 (1.64×) | 172→44 (3.91×) | 429→260 (**1.65×**) |
| s16 | 120-word blog intro on automation | write | 39→37 (1.07×) | 66→66 (1.00×) | 53→53 (1.01×) | 102→41 (2.46×) | 185→41 (4.56×) | 446→237 (**1.88×**) |
| s17 | recipe scaling, 4 → 10 people | math | 37→39 (0.95×) | 66→66 (1.00×) | 66→66 (1.00×) | 121→71 (1.70×) | 216→42 (5.18×) | 506→284 (**1.78×**) |
| s18 | `word_freq(sentence)` | math *(gold code)* | 35→37 (0.96×) | 65→66 (0.99×) | 67→67 (1.00×) | 120→60 (2.00×) | 171→41 (4.23×) | 459→270 (**1.70×**) |
| s19 | free-tier monthly event quota | math *(gold doc)* | 36→37 (0.95×) | 66→66 (1.00×) | 66→66 (0.99×) | 120→72 (1.66×) | 187→44 (4.30×) | 474→285 (**1.66×**) |
| s20 | LinkedIn post for an OSS launch | write | 36→37 (0.96×) | 66→66 (1.00×) | 52→52 (0.99×) | 102→51 (2.01×) | 168→42 (4.02×) | 424→249 (**1.70×**) |
| **median** | | | **0.98×** | **1.00×** | **1.00×** | **1.88×** | **4.46×** | **470→264 (1.78×)** |

Reading the table:

- **`finalize` is the most stable column** — all 20 cases land in 3.85–5.18× with no desk-specific exception. It has the longest reconstructable prefix: invariant + `analysis` + `solution`, 1.1–1.4k tokens, every binding sourced from an upstream call's return value.
- **`middle` spans 1.52–2.77×** — only one of its two bindings is feedable; the RAG passage is a tool output the compiler classifies as `unknown`, and its length varies per case, so the uncovered tail varies with it.
- **The first three columns stay inside a 0.95–1.07× noise band.** The single outlier (s06 `normalize`, 39→49 ms) is one 10 ms jitter event, not a systematic cost.
- **s07, s18 and s19 were routed to MathDesk** while their gold labels are doc/code/doc. Routing was **identical case-by-case in both conditions (20/20 agreement, 17/20 accuracy)**, so every matched pair compares the same work, and the misrouted cases show the same acceleration profile as the correctly routed ones.

## 5. Why the gains land where they do

The interference calibration for Qwen3-4B on this GPU returns a **busy budget of 0**: while a real request is decoding, the measured TBT cost of even one concurrent speculative prefill exceeds the 10% slack threshold (the mixed-batch step time is a sum, not a max). Speculation is therefore admitted **only in idle windows**, and this chain has exactly two — the two RAG retrievals.

The event log of a single request makes the consequence concrete:

```
t=0.072s   normalize's first token → 13 call sites enqueued for speculation
           ... queue stays parked: normalize decodes, then route, then analyze,
               with no inter-call gap longer than the 30 ms idle grace ...
t=9.721s   analyze ends → first RAG window opens
t=9.771s   route's warm executes        (19.8 ms)
t=9.791s   analyze's warm executes      — 70 ms AFTER its own call already finished
t=9.893s   solve's warm executes        (102.6 ms, carrying the fed `analysis` binding)
t=10.025s  finalize's warm executes     (131.4 ms, carrying `analysis`)
t=10.121s  solve is called              → cached = 800
t=20.030s  solve ends → second RAG window
t=20.177s  finalize's warm re-executes  (96.8 ms, now carrying `analysis` + `solution`)
t=20.469s  finalize is called           → cached = 1408
```

`normalize`, the route and `analyze` are structurally unreachable in this configuration: nothing precedes them but another decoding call. `normalize` is additionally an in-degree-0 entry site, which by design belongs to deploy-time prewarm — switched off for this experiment.

Per request the mechanism executes **15 warms, skips 18–19 more by content-hash dedup, records 7 provenance feeds, and verifies 4 fed binding lines** at serve time (the serve-side reorder confirms the fed lines byte-for-byte before reusing them). Median parked time across a request is 19.5 s — speculation waiting for windows it is allowed to use, which is the intended behaviour, not starvation.

## 6. Correction to a previously reported figure

`docs/architecture.md` reports "first route on a fresh engine with graph context fed: 362.1 ms → 68.9 ms, **5.3×**", attributed to the quasi-static visit-context feed. That attribution is wrong, and this run supplies the clean counter-evidence:

| route call | TTFT | cached tokens |
|---|---|---|
| first route in a server's lifetime (grammar cold) | 351.5 ms | 0 |
| any later route, after grammar warm — **both conditions** | 65.7 / 66.1 ms | 0 |

Cached tokens are **zero on both sides**, so the 285 ms difference cannot be KV reuse — it is xgrammar compiling the route's runtime-constructed candidate enum schema. The originally reported 362.1 → 68.9 ms moved `cached` by only 80 → 288 tokens, and 208 tokens of prefill on a 3B model is worth 20–30 ms, not 293 ms.

The mechanism is still ours — our warms attach the output schema precisely so the grammar compiles off the critical path — but it should be reported as **grammar pre-compilation (~285 ms), separate from KV prefill (~20–30 ms)**, not merged into a single 5.3× KV-reuse claim.

## 7. Threats to validity

1. **Trajectory divergence.** Only 1 of 20 cases produced a byte-identical final answer across conditions. vLLM's numerics depend on batch composition, and speculative warms change the batch. Per-call TTFT with matched call sites is therefore the honest unit, and `cached_tokens` — which matches the statically predicted prefix length in every case — is the causal evidence, not the latency delta alone.
2. **Answer quality is not evaluated in this run.** Qwen3-4B is a thinking model and `normalize` has a 96-token budget, so the normalized ticket is a truncated reasoning fragment. The gold answer appears in the final response in 5/20 cases — **identically in both conditions**, which is the property that matters here (speculation is advisory and cannot change the served prompt). Evaluating task quality requires a larger token budget or a non-thinking configuration.
3. **A margin that nearly closed silently.** byllm renders walker fields into the route prompt through `_safe_repr(limit=500)`. The observed `profile` values ranged 399–488 characters — within 12 characters of the cap. Past it, the fed value would no longer match the real one, serve-time verification would reject it, and the `analyze` binding feed would degrade to a cache miss without any error. This is a fragile dependency worth removing.
4. **Single flow, no concurrency.** 20 sequential requests, one at a time. The system assumes a single active flow (`prov_results`, workflow position); concurrent flows would need a flow identifier. Speculation should become *more* valuable under load, but that is untested.
5. **The busy budget is device- and model-specific.** `busy_budget = 0` is what makes the first three calls unreachable. A GPU with headroom, or a smaller model, would allow decode-shadow warming and change the shape of the result.
6. **End-to-end wall time is unchanged** (29.6 s vs 29.7 s median per case). This workload is decode-dominated: ~1400 output tokens per request against ~500 ms of total TTFT. TTFT is the metric this system targets; it is not a throughput result.

## 8. Next steps

1. **Retain the quasi-static tier across requests.** `/reset` currently clears the learned visit context too. `POST /reset?keep_visit_ctx=true` models a warm deployment and should attack the route call, which is the largest remaining un-improved cost (66 ms, `cached = 0`, 20/20 cases).
2. **Re-measure with grammar pre-compilation included**, reported as its own line rather than folded into KV numbers — it is worth ~285 ms on a first typed call and is a real property of the design.
3. **Enable the choice-distribution probe.** With four candidate desks this app is the natural test case, but the probe's ordering hint only pays off when the busy budget is ≥ 1; it needs either a device with headroom or decode-vs-decode calibration.
4. **Remove the `_safe_repr` cliff** (item 3 above) so long walker-field bindings feed instead of silently missing.
5. **Concurrent-load sweep**, where idle windows multiply and speculation should compound.

## Reproduction

```bash
# server (one boot per condition; add --no-speculate for the baseline)
python3 runtime/guard_server.py evaluations/multi_intent/main.jac --no-type-check \
    --model Qwen/Qwen3-4B --max-model-len 8192 --gpu-mem 0.60 --greedy --port 8964

# precompile every output schema: 4 dummy cases, one per desk
# then per case: POST /reset  ->  one client run  ->  capture /stats
export JAC_ROUTE_CACHE_LAYOUT=1
cd evaluations/multi_intent && REQUEST="<case text>" python3 -m jaclang run main.jac
curl -s -X POST localhost:8964/reset      # cold start for the next case
curl -s localhost:8964/stats              # ttft, cached_tokens, warms, feeds, reorders, event log
```

Raw per-case `/stats` dumps for both conditions, the sweep driver and the aggregation script live in
`evaluations/multi_intent/results/2026-08-20/` (`base20/`, `spec20/`, `sweep.py`, `agg.py`).
Regenerate every table in this report with `python3 agg.py base20 spec20`.
