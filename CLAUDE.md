# Proactive Prefill — speculative prefill for Jac byllm calls

Research project (2026-08). Jaseci source at `~/jaseci` (jaclang editable, conda env `jaseci`, py3.13). Not a git repo.

## Thesis

In a byllm call, ~95% of the prompt is compile-time constant ([SYSTEM][TOOLS][TOOL_SCHEMA][RESPONSE_FORMAT] + the USER header/sems); only argument bindings and context fields vary. A compiler-derived "side runtime" prebuilds every call site's invariant prefix, uses a static call-site topology graph to speculatively warm the *next* call while the current one decodes, and a static binding-provenance table to speculatively feed the next call's *argument values* as they become observable. Three tiers of prompt stability: **invariant** (compile-time), **quasi-static** (graph-dependent, stable across requests: visit-by here/candidates + choice grammar), **per-request** (bindings, fed in arrival order). Feeding is always ADVISORY — the served prompt is rebuilt/verified from real values; stale speculation costs a cache miss, never a wrong prompt.

Positioning: vs Parrot (OSDI'24) / PromptCache (MLSys'24); differentiator = compiler derives invariant/variant split + workflow DAG + argument provenance from the language, no annotations ("language-integrated speculative prefill").

## Status (2026-08-18): full pipeline working end-to-end on real `jac run`

Interception (byllm `InterceptorLLM` at `model_call_no_stream`) → GuardServer (AsyncLLM v1, APC + chunked prefill + priority scheduling) → Monitor speculation at **first token** (warm successors' invariants + provenance-feed their bindings) and at **call end** (feed ret-sourced bindings) → serve-time **binding-zone reorder** (approach B) makes fed arrival-order prefixes strict prefixes of byllm's decl-order prompts. ReAct/tools run natively on the Jac side; only raw completions hit the server.

Key measured results (Qwen2.5-3B, RTX 3090, greedy; details in docs/architecture.md): chain E2E TTFT 830→353ms (2.35×), speculation-addressable calls 3.4–8.8×; routing quality with cache-aware layout 8/10 vs 4/10 (layout also +cached 80→288); visit quasi-static feed kills fresh-engine first-route cold start 362→69ms (5.3×); grammar (xgrammar) precompile at warm removes ~300ms first-typed-call latency; provenance auto-feed issues 10 binding feeds in route's decode shadow, cached covers full bindings; tier-3 showcase (big_agent, ~1.4k-token bindings, `--no-prov-feed` ablation) consumer call 2.67× / E2E 1.40× on a worst-case zero-think-time sequential flow. Ret feeds fire by DATAFLOW (inverse provenance index + result store — `a=f(); b=g(); h(a,b)` feeds h.a at f's end), deferred to the next call's first token (decode-shadow flush; call_end submission polluted the next call's prefill ~34ms); field feeds stay topology-adjacent (mutable state → adjacency = freshness heuristic). Trajectory nondeterminism across batch compositions (vLLM numerics) means matched-call comparison, not E2E sums, is the honest unit on single cases.

## Layout

- `static/static_parser.py` — engine-free: UniIR compile, `ByLLMDecl` (kind `func`/`visit`), obj/enum→Python translation, byte-faithful byllm prompt renderers, topology (`build_callsite_graph`: receiver-aware, genai visits, traversal-queue continuation edges), **`build_provenance`** (param → const/field(+slice)/ret/unknown).
- `static/async_byllm.py` — `AsyncByLLM(torch.nn.Module)`: prebuilt invariant + `self.sampler`; `visit_stable_prefix()`; `warm_invariant()` passive; `forward(params)` returns parsed typed value. SYSTEM_PERSONA/TOOL_INSTRUCTION copied byte-identically from byllm (Jac triple-quoted strings do NOT dedent — a "clean" copy kills every cache hit).
- `runtime/side_runtime.py` — `SideRuntime`: callsites, `callsites_topo`, `provenance`, `next_callsites()`; CLI dumps both tables.
- `runtime/incremental_feed.py` — three-tier feed layer: `PrefillSession`/`FeedStore`, `partial_user_prompt`/`final_messages` (arrival-order replay w/ verification), `reorder_binding_zone` (serve-boundary reorder), `extract_visit_ctx`/`extract_walker_fields`/`eval_provenance`.
- `runtime/guard_server.py` — `GuardServer` + `Monitor`; HTTP: `/call /generate /feed /feed_visit /visit_ctx /health /stats`. Flags: `--deploy-time-prewarm` (opt-in), `--no-speculate` (APC baseline), `--greedy` (deterministic A/B), `--model/--max-model-len/--gpu-mem/--port`.
- `~/jaseci/jac/jaclang/byllm/llm.jac` + `llm.impl/interceptorLLM.impl.jac` — `InterceptorLLM(BaseLLM)`; `visit_routing.jac` — `JAC_ROUTE_CACHE_LAYOUT=1` cache-aware zone order (stable zones before Walker).
- `demo/chain_demo.jac`, `demo/chain_big.jac` (~1.3k-token invariants); `demo/diamond.jac` (dataflow-firing fixture), `demo/big_agent.jac` (tier-3 showcase, ~1.4k-token bindings), `demo/queue_test.jac` (continuation-edge fixture); `jac_sample/Jac-Rag-GPT/` + drivers `run_case.jac`, `route_probe.jac` (routing-quality probe), `direct_case.jac` (bypasses router variance).

Key naming: functions `Owner.name`; visit sites `walker.ability.visit@line`.

## Roadmap (remaining)

`select=1` refinement to drop spurious continuation edges; invariant promotion (closed-world candidate sets → true invariant); streaming ret chunks into successor bindings mid-decode; `here`/`self`-scope provenance (needs candidate-instance binding); non-str ret feeds; graph-mutation hooks for visit ctx (today: learned from first served route / fed via HTTP); walker-field repr truncation (`_safe_repr` limit 500) degrades long-value feeds to misses; benchmark suite runs (multi-case, larger models, concurrent load).

## Working conventions (user-set)

- Never line-break long string literals — prompt strings must stay byte-faithful to byllm; single source line, explicit `\n`. Byte-parity must be validated against REAL byllm renderings, not our own mirrors.
- Utilities live in `static_parser.py` / `incremental_feed.py`; `AsyncByLLM`/`SideRuntime`/`GuardServer` stay thin. Import direction `runtime → static`, no cycles; `incremental_feed` is engine-free.
- One shared engine, injected. Warm/feed are caller/monitor-triggered, priority=1, and must never sit in a call's TTFT-critical window (speculate at first token, not call start).
- Topology & provenance bias: may-happen over-approximation (spurious edge/feed ≈ free; missed = cold TTFT). Dynamic aliasing out of scope.
- Deploy-time prewarm is OPT-IN (`--deploy-time-prewarm`); default relies on monitor speculation only.
- The user edits/deletes files mid-session — always Read current state before editing; their on-disk changes are canonical.
- Experiments: `--greedy` for A/B; fresh engine per condition; kill orphaned `VLLM::EngineCore` procs before restarting (they hold GPU); `jac` CLI uses a cached runtime snapshot — use `python3 -m jaclang run` for editable jaclang; run Jac drivers from their own directory (module resolution).

## Detail docs

- @docs/architecture.md — components, APIs, HTTP protocol, experiment recipes + all measured results
- @docs/byllm-internals.md — where byllm builds prompts, zone structure, invariance boundary, interception points
- @docs/uniir-recipes.md — verified UniIR parsing recipes and gotchas (topology + provenance)
- docs/session-2026-08-18.md — session archive: decision rationale, bug/gotcha log, open threads, artifact map (incl. the three files patched in ~/jaseci that must be reapplied if that checkout resets)
- docs/kvflow-diff.md — KVFlow deep-read differentiation table (verified quotes; related-work ammunition)
- paper/draft.md — working paper draft (measurements verified; CITATIONS UNVERIFIED — check before external use); demo slide: https://claude.ai/code/artifact/999ad36a-feff-4a61-9d98-1e227b6d3b24
