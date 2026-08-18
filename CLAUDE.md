# Proactive Prefill — speculative prefill for Jac byllm calls

Research project (2026-08). Jaseci source at `~/jaseci` (jaclang editable, conda env `jaseci`, py3.13). Not a git repo.

## Thesis

In a byllm call, ~95% of the prompt is compile-time constant ([SYSTEM][TOOLS][TOOL_SCHEMA][RESPONSE_FORMAT] + the USER header/sems); only argument bindings and context fields vary. A compiler-derived "side runtime" prebuilds every call site's invariant prefix, warms vLLM's prefix cache at deploy time, and uses a static call-site topology graph to speculatively warm the *next* call while the current one decodes. Current focus: first-call cold-start TTFT (incremental/gradual prefill deferred).

Measured: 95% invariant fraction; 4.0–9.7× TTFT cut on 1k–4k-token prompts (user); verified cold 1153.7ms → warm 279.4ms (4.1×) on Qwen2.5-0.5B, ~1.2k-token call site; invariant is a strict string prefix (96.6% of chars on RagChat.respond). Positioning: vs Parrot (OSDI'24) / PromptCache (MLSys'24); differentiator = compiler derives invariant/variant split + workflow DAG from the language, no annotations ("language-integrated speculative prefill").

## Layout

- `static/static_parser.py` — engine-free utility layer: UniIR compile, `ByLLMDecl` (kind `func`/`visit`), obj/enum→Python translation, byte-faithful byllm prompt renderers, call-site topology (`build_callsite_graph`, receiver-aware, `visit by llm()` included).
- `static/async_byllm.py` — `AsyncByLLM(torch.nn.Module)`: one per call site; shared injected engine (`self.model`); prebuilt invariant + `self.sampler`; `warm_invariant()` passive prefill; `forward(params)` returns parsed typed value (raw RequestOutput in `self.last_output`).
- `runtime/side_runtime.py` — `SideRuntime`: `byllm_callsites` (key→AsyncByLLM), `callsites_topo`, `next_callsites(key)` speculation targets; CLI dumps topology.
- `jac_sample/Jac-Rag-GPT/` — sample app; graph: `interact.route.visit@183 -> [5 agent calls]`.

Key naming: functions `Owner.name`; visit sites `walker.ability.visit@line`. `preprocess.py`, `demo.jac`, `verify_takeover.py`, `topo_demo.jac` were deliberately removed — verification drivers are recreated on demand (recipe in docs/architecture.md).

## Roadmap (next phase)

Runtime monitor + SideRuntime as server; speculative prefill before the byllm call; intercept the real call and route it to the side runtime. Interception point: codegen calls `JacLLM.call_llm(model, mt_run) -> model.invoke(mt_run=mt_run)` (`~/jaseci/jac/jaclang/jac0core/impl/runtime.impl.jac:1720`) — an interceptor is a Model-compatible object implementing `invoke(mt_run)`. Also: prebuild candidate node-type descriptions into the visit invariant; profile-weighted topology edges; full ReAct loop once real callables transfer to the server.

## Working conventions (user-set)

- Never line-break long string literals — prompt strings must stay byte-faithful to byllm; single source line, explicit `\n`.
- Utilities live in `static_parser.py`; `AsyncByLLM`/`SideRuntime` stay thin. Import direction `runtime → async_byllm → static_parser`, no cycles.
- One shared vLLM engine, injected via `bind_engine` — never per call site. `warm_invariant()` is caller-triggered only, never automatic.
- Topology bias: may-happen-next over-approximation is accepted (spurious edge = wasted prefill ≈ free; missed edge = cold TTFT). Dynamic aliasing (`f = self.m; f()`) deliberately out of scope.
- The user edits/deletes files mid-session — always Read current state before editing; their on-disk changes are canonical.

## Detail docs

- @docs/architecture.md — APIs, experiment recipe, environment facts
- @docs/byllm-internals.md — where byllm builds prompts, zone structure, invariance boundary
- @docs/uniir-recipes.md — verified UniIR parsing recipes and gotchas
