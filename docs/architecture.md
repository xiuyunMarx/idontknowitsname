# Architecture, APIs, and experiment recipe

## static/static_parser.py (engine-free: no vllm/torch)

- `ByLLMDecl` dataclass: `name, qualifier, kind ("func"|"visit"), params [{name,type,sem,required}], return_type, sem, owner_sem, tools (OpenAI schemas, finish_tool last), call_params (literal llm() kwargs), extra_system_prompt, intent (visit), reponse_format, return_type_obj (materialized Python type), module, lineno`. `signature()` renders `name(p: T, ...) -> R`.
- `build_uniir(path, type_check)` → `(JacProgram, Module)`; `collect_type_defs(program)` → obj/enum specs; `materialize_type` → obj→dataclass / enum→Enum (sems on `_jac_semstr`/`_jac_semstr_inner`, cyclic refs degrade to Any); `json_schema_of` expands obj fields + enum legends (`1=LOW, 2=MEDIUM...`).
- `build_decl(program, ab, type_defs)` → ByLLMDecl for a genai ability; `build_visit_decl(program, vs)` for `visit ... by llm()` (resolves `intent=GLOB_NAME` through `_glob_literal`).
- Prompt renderers, byte-faithful ports of byllm: `format_tools_for_prompt`, `response_format_of(rt, defs)`, `finish_tool_schema(rt, defs)`, `extract_finish_output(text)`.
- Topology: `find_byllm_abilities`, `find_genai_visits`, `ability_key` (`Owner.name`), `visit_key` (`walker.ability.visit@line`), `parse_callsite_topology(program, ab, funcDecls)`, `parse_visit_topology`, `build_callsite_graph(program, funcDecls)` → `Dict[key, List[key]]`. Receiver-aware `_call_decl_keys`: `self.m()` → owner class + base chain; bare call → module-level decl; unknown receiver → all same-name candidates (over-approx).

## static/async_byllm.py

`AsyncByLLM(decl, engine=None, sampling_params=None)` — torch.nn.Module per call site.
- `_build_invariant()`: func kind → SYSTEM (persona + TOOL_INSTRUCTION + `# Calling tools` block when tools) and USER prefix (qualified signature `--- sem` + indented schema rows); visit kind → routing system prompt (+ select suffix) and `Goal: {intent}`. Prebuilds `self.sampler` (temperature/max_tokens from llm() literals, `stop=["</tool_call>"]` with tools, `structured_outputs=StructuredOutputsParams(json=...)` from response schema).
- `build_full_prompt(params)`: appends bindings `name = repr(value)` in declaration order (+ `self` identity zone when `params["self"]` and owner_sem); visit kind appends `Walker:`/`Current node:`/`Candidates (choose by handle):` zones from params.
- `warm_invariant()`: system+invariant-user request, max_tokens=1 — warms APC. Passive.
- `forward(params)`: chat with `self.sampler`, stash `self.last_output` (RequestOutput; TTFT in .metrics), return `parse_response(text)` — str passthrough, finish_tool extraction for tools, else json.loads → unwrap `schema_object_wrapper` → `TypeAdapter(return_type_obj).validate_python` (real typed instances).
- Import shim: `try: from static_parser import ... except ImportError: from static.static_parser import ...`.

## runtime/side_runtime.py

`SideRuntime(repo_path, type_check=True)`: compiles, `collect_type_defs`, builds `func_decls` + `byllm_callsites` (functions AND genai visits), `callsites_topo = build_callsite_graph(...)`. `next_callsites(key)` → AsyncByLLM list to speculatively warm. `bind_engine(engine)` broadcasts. CLI: `python runtime/side_runtime.py <file.jac> [--no-type-check]` dumps the topology. Inserts repo root into sys.path for `static.*` imports.

## Experiment recipe (verified 2026-08-18)

```python
engine = vllm.LLM(model="Qwen/Qwen2.5-0.5B-Instruct", enable_prefix_caching=True,
                  enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.45)
rt = SideRuntime("app.jac"); rt.bind_engine(engine)
for fn in rt.byllm_callsites.values(): fn.warm_invariant()   # deploy-time warm
# TTFT probe: deepcopy fn.sampler, max_tokens=1, wall-clock engine.chat(fn.build_full_prompt(params))
result = fn(params)   # parsed typed value; fn.last_output.metrics for TTFT
```

Checks a verification driver should assert: (1) `msgs[1]["content"].startswith(fn.invariant_user_prefix)` — strict prefix property; (2) parsed type matches `decl.return_type_obj`; (3) warm vs cold probe TTFT. Caveat: call sites share the SYSTEM persona prefix, so within one process later cold probes are already partially warm — per-site clean cold numbers need a fresh engine each.

## Environment

RTX 3090 24GB. HF cache: Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, opt-125m. vllm 0.19.1 (`StructuredOutputsParams` replaced `GuidedDecodingParams`; offline chat: `engine.chat(messages, sampling_params=..., use_tqdm=False)`). torch 2.10. pydantic available.
