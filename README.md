# Proactive Prefill

**Language-integrated speculative prefill for [Jac](https://www.jac-lang.org/) byllm calls.**

In a byllm LLM call, ~95% of the prompt is compile-time constant — the system persona, the function's typed signature and `sem` strings, the tool protocol, the output schema. Only argument bindings vary at runtime, and they sit at the very end of the token sequence. This project exploits that: a compiler-derived *side runtime* prebuilds every call site's invariant prefix, uses a static call-site topology graph to warm the **next** call's prefix while the current one decodes, and uses a static argument-provenance table to speculatively feed the next call's **binding values** the moment they become observable. By the time an agent is invoked, its entire prompt is already in the KV cache.

Unlike Parrot (OSDI'24) or PromptCache (MLSys'24), no programmer annotations are needed: the invariant/variant split, the workflow DAG, and the argument provenance are all derived from the language itself (Jac's compiler IR + walker semantics).

## Results (Qwen2.5-3B, RTX 3090, deterministic decoding, APC-enabled baselines)

| Scenario | Baseline | With speculation | Gain |
|---|---|---|---|
| 4-call chain, E2E TTFT (no deploy warm) | 830.3 ms | 352.7 ms | **2.35×** |
| Speculation-reachable chain calls | 138–305 ms | 34–40 ms | **3.4–8.3×** |
| First typed call (structured output) | 304.7 ms | 36.5 ms | **8.3×** (grammar precompiled at warm) |
| RAG agent first turn (after router) | 82.3 ms | 33.7 ms | **2.4×** |
| First route on a fresh engine (graph ctx fed) | 362.1 ms | 68.9 ms | **5.3×** |
| Route prompt, cache-aware layout | 4/10 routed correctly | 8/10 | quality ↑, cached 80→288 tok |

All numbers from real `jac run` executions through the interceptor, not synthetic probes. Correctness is never at risk: speculation is *advisory* — served prompts are always rebuilt/verified from real values, so a wrong guess costs a cache miss, never a wrong prompt.

## How it works

```
      compile time                          runtime
┌───────────────────────┐   ┌──────────────────────────────────────────┐
│ static/static_parser  │   │ Jac program (glob llm = InterceptorLLM)  │
│  · byllm call sites   │   │   byllm builds prompts + runs ReAct      │
│  · invariant prompts  │   │        │ raw completions only            │
│  · topology graph     │   │        ▼                                 │
│  · provenance table   ├──▶│ GuardServer (vLLM AsyncLLM, APC)         │
│  · Jac obj/enum →     │   │  · warm invariants (priority-isolated)   │
│    Python classes     │   │  · Monitor: at call X's FIRST TOKEN,     │
└───────────────────────┘   │    warm successors + feed their bindings │
                            │    evaluable from X's request; at X's    │
                            │    END, feed ret(X)-sourced bindings     │
                            │  · serve: reorder binding zone to the    │
                            │    fed arrival order (verified, safe)    │
                            └──────────────────────────────────────────┘
```

Three tiers of prompt stability, each with its own trigger:

1. **Invariant** (compile-time): persona + signature + sems + tool protocol + output grammar. Warmed at deploy (opt-in) or speculatively via the topology graph.
2. **Quasi-static** (graph-dependent): a `visit ... by llm()` router's candidate list and choice grammar — stable across requests until the graph changes. Learned from served traffic or fed when the graph is built; survives engine restarts via export/re-feed.
3. **Per-request** (bindings): fed one by one in *arrival order* as values become observable — from a predecessor's request state (`visitor.message` while the router decodes) or its result (`ret(analyze)` → `draft`'s argument). A serve-time reorder makes arrival-order prefixes match byllm's declaration-order rendering.

## Repository layout

```
static/
  static_parser.py     engine-free compile-time layer: call-site extraction,
                       invariant prompt construction, topology, provenance
  async_byllm.py       AsyncByLLM: one object per call site (invariant, sampler,
                       typed output parsing via materialized Jac obj/enum classes)
runtime/
  side_runtime.py      SideRuntime: callsites + topology + provenance; CLI dump
  incremental_feed.py  engine-free feed layer: sessions, arrival-order prompts,
                       binding-zone reorder, provenance evaluation
  guard_server.py      GuardServer + Monitor + HTTP API (the side-runtime server)
demo/                  chain demos (~1.3k-token invariants)
jac_sample/Jac-Rag-GPT/  RAG multi-agent sample app + experiment drivers
docs/                  architecture / byllm internals / IR parsing recipes
```

Two small patches live in the jaseci checkout (`~/jaseci/jac/jaclang/byllm/`): `InterceptorLLM` (a `BaseLLM` connector intercepting at `model_call_no_stream`, so byllm's own machinery stays untouched) and an env-gated cache-aware zone order for `visit_routing.jac`.

## Quick start

```bash
# 1. start the guard server (loads the app's IR, boots vLLM, serves the interceptor)
python3 runtime/guard_server.py demo/chain_big.jac --no-type-check \
    --model Qwen/Qwen2.5-3B-Instruct --greedy --port 8964

# 2. run the Jac program (its glob llm is an InterceptorLLM pointing at :8964)
cd demo && python3 -m jaclang run chain_big.jac

# 3. inspect what happened
curl -s localhost:8964/stats | python3 -m json.tool   # TTFT, cached_tokens, feeds, reorders
```

Useful flags: `--no-speculate` (vanilla prefix-caching baseline for A/B), `--deploy-time-prewarm` (warm every invariant at boot), `--greedy` (deterministic experiments). For the RAG app set `JAC_ROUTE_CACHE_LAYOUT=1` on the client to enable the cache-aware route layout.

Static analysis only (no GPU):

```bash
python3 runtime/side_runtime.py jac_sample/Jac-Rag-GPT/main.jac --no-type-check
# prints the call-site topology and the binding provenance table
```

## Status & roadmap

Working end-to-end: interception, invariant warming, topology speculation, visit-context quasi-static feed (incl. grammar precompile), automatic provenance-driven binding feed, serve-time reorder. Open: streaming a predecessor's tokens into a successor's binding mid-decode; `here`/`self`-scope provenance; invariant promotion for provably-closed candidate sets; multi-case benchmark sweeps under concurrent load.

Research project, 2026. See `docs/` for full APIs, measured data, and reproduction recipes.
