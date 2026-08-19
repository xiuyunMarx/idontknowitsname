# Architecture, APIs, and experiment results

## System shape

```
Jac program (unchanged semantics; glob llm = InterceptorLLM(...))
   │  byllm builds prompts natively (MTRuntime.factory / route_visit);
   │  ReAct loop, tool execution, typed retries all run Jac-side
   ▼
InterceptorLLM (byllm connector, ~/jaseci .../llm.impl/interceptorLLM.impl.jac)
   │  intercepts at model_call_no_stream: POST /generate {key, messages, schema,
   │  temperature, max_tokens, stop}; visit routing has no MTIR → key="" and the
   │  server matches it by prompt shape; fallback to plain Model on error/stream
   ▼
GuardServer (runtime/guard_server.py)  — owns AsyncLLM v1 engine
   ├─ APC (prefix caching) + chunked prefill + priority scheduling
   ├─ Monitor: first-token speculation (warm + provenance feed), call-end ret feed
   ├─ FeedStore: per-request sessions (arrival order) + per-visit-site graph ctx
   └─ serve: visit ctx learning; binding-zone reorder (approach B); stats
```

Three tiers of prompt stability (design core; see incremental_feed.py docstring):

| tier | content | trigger | reuse |
|---|---|---|---|
| invariant | persona, signature+sems, tool protocol, output schema/grammar | compile time | forever, cross-request |
| quasi-static | visit-by `here`/`candidates` zones + choice grammar | graph known/changed | cross-request until graph changes |
| per-request | argument bindings | value observed (provenance firing) | one call (all its ReAct turns) |

Advisory principle: warms/feeds only ever pre-populate KV; the served prompt is
byllm's own rendering (reordered only where fed lines match verbatim). Stale
speculation = cache miss from the divergence point, never a wrong prompt.

## Static layer (static/)

`static_parser.py` (engine-free):
- `ByLLMDecl`: name, qualifier, kind (`func`/`visit`), params [{name,type,sem,required}], return_type(+`return_type_obj` materialized class), sem/owner_sem, tools (OpenAI schemas, finish_tool last), call_params, intent (visit), reponse_format.
- Extraction: `build_uniir`, `find_byllm_abilities`, `find_genai_visits` (VisitStmt whose target is BinaryExpr `by` + llm FuncCall), `build_decl`, `build_visit_decl` (resolves `intent=GLOB` literals), `collect_type_defs`+`materialize_type` (obj→dataclass, enum→Enum, sems on `_jac_semstr*`), `json_schema_of` (obj fields + enum legends).
- Topology `build_callsite_graph` → `Dict[key, List[key]]`: sequential/branch/loop scan over ordered `kid` lists; caller continuations; receiver-aware call resolution (`self.m()` → owner class + base chain; bare → module-level; unknown → all same-name); genai visits as first-class callsites; **traversal-queue continuation** (walker/node ability body end → may-edges to every ability that walker can still trigger, incl. same-type self-edges and walker exit abilities).
- **Provenance `build_provenance`** → `Dict[key, Dict[param, spec]]`, spec ∈ `{kind: const, value}` | `{kind: field, scope: visitor|here|self, attr, slice}` | `{kind: ret, of: key}` | `{kind: unknown}`. Kw + positional args; bare names traced to `x = <byllm call>(...)` in enclosing scope; conflicting sites demote to unknown.

`async_byllm.py`: `AsyncByLLM` per callsite — `invariant_system` / `invariant_user_prefix` prebuilt (byte-faithful to byllm incl. the non-dedented SYSTEM_PERSONA/TOOL_INSTRUCTION and the doubled tool-name quirk); `visit_stable_prefix(here, candidates)`; `build_full_prompt` (visit branch uses cache-aware order Goal→here→candidates→Walker); prebuilt `self.sampler` (temperature/max_tokens literals, `stop=["</tool_call>"]` with tools, `structured_outputs` from static schema); `warm_invariant()`; `forward()` → parsed typed value via pydantic TypeAdapter on the materialized class (`schema_object_wrapper` unwrap, finish_tool extraction).

## Feed layer (runtime/incremental_feed.py, engine-free)

- `PrefillSession.feed(name, repr_text)`: arrival-ordered; same-name re-feed with new text truncates from that item.
- `FeedStore`: sessions[(key,sid)], `sessions_for(key, ttl=600)`, visit_ctx[key] = {here, candidates, schema}.
- `partial_user_prompt(fn, session)` = invariant + fed lines (the warmable prefix).
- `final_messages(fn, params, session)` = server-side prompt build for `/call`: replay fed order while `repr(params[name])` matches, cut at first divergence, rest in decl order, then self zone.
- `reorder_binding_zone(user_text, fn, sessions)` (**approach B**): locate the contiguous binding block (single-line `name = repr`, names ⊆ decl params; non-contiguous → bail), content-match the best session (longest verbatim fed prefix), move matched lines to front in fed order. No sid on the wire — pure content matching. Sessions are kept alive across ReAct turns (same reorder every turn preserves inter-turn APC).
- `extract_visit_ctx(user_text)`: learn here/candidates from a served route prompt (cache-aware layout only). `parse_arch_fields`/`extract_walker_fields`: field reprs from `_describe_arch` renderings (paren-balanced, quote-aware; tolerates sem heads with parens). `eval_provenance(spec, walker_fields, results)` → repr text: const → repr(value); ret → results[of]; field(visitor) → raw repr, slice via ast.literal_eval + guarded subscript eval (truncated `_safe_repr` values fail closed → no feed).

## GuardServer (runtime/guard_server.py)

Engine: `AsyncLLM.from_engine_args(enable_prefix_caching, enable_chunked_prefill, scheduling_policy="priority", enforce_eager, ...)`. All warms/feeds run at priority=1 (yield to real calls). Prompts rendered via tokenizer chat template (content-parts flattened).

Monitor events:
- `on_call_start(key)`: log only.
- `on_first_token(key, rid, walker_fields)`: speculation trigger (NOT call start — submitting warms in the same scheduling window as a call's prefill measurably delayed it; first token ends the TTFT-critical window with ~95% of decode still ahead). Per topology successor: invariant/ctx warm + provenance feed of field/const params (walker_fields parsed from this route request's Walker zone; field feeds stay topology-adjacent on purpose — mutable state, adjacency doubles as freshness heuristic). Also FLUSHES ret feeds deferred at earlier call ends — this call's decode is the shadow they run in.
- `on_call_end(key, ..., rid, result_repr)`: stores the parsed result (str returns, finish_tool payloads) in `prov_results` and marks ret-consumers pending via the inverse provenance index (`ret_consumers`, DATAFLOW firing — consumers any number of calls downstream: `a=f(); b=g(); h(a,b)` marks h at f's end). Feeds are NOT submitted here: a feed submitted at call_end races the immediately following call's prefill (~34ms TTFT pollution measured per call in tight sequential flows).
- All provenance feeds for one consumer share session sid="prov" (one arrival-ordered replayable prefix even when fed from multiple events); re-fed same value = no-op, changed value truncates (self-healing across requests). Single-flow assumption; concurrent flows would need a flow id.

Warm covers the longest known stable prefix: invariant, extended for visit sites by fed/learned ctx, **with the structured-outputs schema attached so xgrammar compiles the grammar off-path** (grammar precompile was worth ~300ms on first typed calls).

HTTP: `POST /call {key, params(reprs), sid?}` (server-built prompt, session replay) · `POST /generate {key?, messages, schema, temperature, max_tokens, stop}` (byllm-built prompt; visit key matched by shape; ctx learning; binding reorder) · `POST /feed {key, sid, param, value}` · `POST /feed_visit {key, here, candidates, schema?}` · `GET /visit_ctx` (export for post-restart prewarm) · `GET /health` (topology) · `GET /stats` (calls w/ ttft+cached_tokens+prompt_head, warms, feeds, reorders, monitor events).

Flags: `--deploy-time-prewarm` (opt-in warm_all at boot), `--no-speculate` (vanilla-APC baseline), `--no-prov-feed` (tier-3 ablation: invariant warms stay on), `--greedy` (temperature=0 on served requests), `--model --max-model-len --gpu-mem --port --no-type-check`. Env (Jac side): `JAC_ROUTE_CACHE_LAYOUT=1` enables the reordered route prompt.

## Measured results (Qwen2.5-3B-Instruct, RTX 3090, greedy, fresh engine per condition)

Chain (demo/chain_big.jac, ~1.3k-token invariants, no deploy warm; ablation = speculation on/off):
- E2E TTFT sum 830.3 → 352.7ms (**2.35×**); per call: draft 138.6→40.3 (3.4×), polish 117.2→34.3 (3.4×), count_words 304.7→36.5 (**8.3×**, grammar precompile), analyze unchanged (entry, in-degree 0 — deploy warm's job, excluded by design). `cached_tokens` confirms invariant hits (1328/1104).
- Deploy warm + speculation (earlier run): mean TTFT 207→40ms (5.2×).

Jac-Rag-GPT case rag_qa-018 (route + ReAct agent):
- Matched-call: agent turn1 82.3→33.7ms (2.4×, cached 0→528); turn2 unchanged (natural APC); route unchanged (entry). E2E 650→602 (1.08× — Amdahl: route dominates).
- Cache-aware route layout (JAC_ROUTE_CACHE_LAYOUT=1): routing accuracy 8/10 vs 4/10 (recency of message helps 3B), steady route cached 80→288, TTFT 35.1→31.8ms.
- Visit quasi-static feed (restart scenario: learn ctx → fresh engine → /feed_visit before any request): first route 362.1ms/cached=80 → **68.9ms/cached=288 (5.3×)**; accuracy unchanged.
- Manual tier-3 feed via /call (long message+history): 103.3→35.0ms (2.9×), cached 528→800.
- Automatic provenance feed (zero manual): route decode shadow → 10 feeds (5 agents × message, chat_history[-10:] slice applied), chosen agent turn1 cached 528→624; chain ret-feeds: draft cached 1328→1408, polish 1104→1216, count_words 64→160 (each delta = its binding zone). At these binding sizes (~100 tok) TTFT delta is under noise — value scales with binding size.

Tier-3 showcase (demo/big_agent.jac: 3 long producers ~470 tok each fan into final_verdict; ablation `--no-prov-feed`, both with invariant speculation on):
- Dataflow firing verified: final_verdict.timeline fed the moment report_timeline ends — two calls / ~13s before final_verdict runs (impossible under topology-adjacent feeding; fixture demo/diamond.jac).
- Naive call_end submission (A2): verdict TTFT 149.1→79.2ms (cached 96→1520, all 3 bindings) BUT producers polluted 32→66ms → E2E wash (335.7 vs 335.9ms). Root cause: feed races the immediately following call's prefill.
- With decode-shadow deferral (A3, current code): producers unpolluted (32.3/30.5ms), verdict 149.1→**55.9ms (2.67×)** with 2/3 bindings fed (cached 960; the LAST-arriving binding is structurally unwinnable in a zero-think-time sequential flow — its feed flushes at the consumer's own first token as a harmless no-op). E2E TTFT 335.9→**240.4ms (1.40×)**. In flows with think time, concurrency, or non-adjacent consumers, the abandoned last hop shrinks and gains grow.

Caveats to carry into papers: baseline must be APC-enabled (it is, in all above); vLLM numerics differ across batch compositions → greedy trajectories can diverge (report matched calls or average many cases); intra-process cross-callsite persona sharing partially warms later cold calls.

## Reproduction recipes

```bash
# server (from repo root; add --no-speculate for baseline, --deploy-time-prewarm to warm at boot)
python3 runtime/guard_server.py demo/chain_big.jac --no-type-check --port 8964 \
    --model Qwen/Qwen2.5-3B-Instruct --greedy
# client (run from the .jac file's own directory; python -m jaclang, NOT the jac CLI snapshot)
cd demo && python3 -m jaclang run chain_big.jac
# RAG app: server on jac_sample/Jac-Rag-GPT/main.jac --max-model-len 8192;
# client: cd jac_sample/Jac-Rag-GPT && CASE_SID=<unique> JAC_ROUTE_CACHE_LAYOUT=1 python3 -m jaclang run run_case.jac
# route quality probe: python3 -m jaclang run route_probe.jac   (10 labeled messages, prints ACCURACY)
# deterministic agent driver (no router variance): CASE_MSG="..." python3 -m jaclang run direct_case.jac
curl -s localhost:8964/stats   # ttft / cached_tokens / feeds / reorders / event timeline
```

Static-only inspection (no GPU): `python3 runtime/side_runtime.py <app.jac> --no-type-check` dumps topology + provenance.

Hygiene: kill orphaned `VLLM::EngineCore` before restarting (`nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill`); the `jac` CLI runs a cached runtime snapshot in `~/.cache/jac/rt` that does NOT see edits to `~/jaseci`.

## Environment

RTX 3090 24GB; conda env `jaseci` (py3.13); vllm 0.19.1 (v1 AsyncLLM; `StructuredOutputsParams` replaced GuidedDecodingParams; offline `LLM.chat`); torch 2.10; HF cache has Qwen2.5-0.5B/3B-Instruct, opt-125m; fastapi/uvicorn/requests available. byllm config: apps may set `[plugins.byllm.model] default_model` in jac.toml — a `by llm` whose `llm` glob isn't imported silently uses that (watch for accidental OpenAI calls).
