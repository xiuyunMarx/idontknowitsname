# Proactive Prefill for Jac byLLM

This prototype reduces time to first token (TTFT) by predicting future Jac
`by llm()` calls and preparing their vLLM KV-cache prefixes during tool execution
and inter-call idle windows. Multiple Jac programs use separate TCP backends but
share one GPU inference engine.

## How it works

```text
Jac tenant A ──► backend :8964 ─┐
                                ├──► GuardServer ──► shared vLLM engine
Jac tenant B ──► backend :8965 ─┘         │
                                          └── topology-guided idle prefill
```

The static pass extracts byLLM declarations, invocation topology, and argument
provenance. While a current call is decoding, executing a local tool, or waiting
between calls, the guard renders the longest known prefix of likely successors:

- invariant system prompt and function schema;
- compile-time constant arguments;
- return values already produced by upstream calls.

The real request uses the same prefix construction, allowing vLLM automatic
prefix caching (APC) to reuse the prepared KV blocks.

## Requirements

- Linux with an NVIDIA GPU supported by vLLM
- Python environment containing `vllm`, `torch`, and Jac 0.31+
- The Jac checkout installed in editable mode when using the custom
  `InterceptorLLM`
- A locally available or downloadable Hugging Face model

This development environment needs the Conda C++ runtime ahead of the system
runtime:

```bash
export LD_LIBRARY_PATH=/home/xiaoyu/miniconda3/envs/jaseci/lib
```

Verify the active Jac installation:

```bash
which jac
python -c 'import jaclang; print(jaclang.__file__)'
```

## Run the multi-tenant example

Start the proactive server:

```bash
python start_server.py \
  --model Qwen/Qwen3-8B \
  --program research:jac_programs/research_agent.jac:8964 \
  --program operations:jac_programs/operations_agent.jac:8965
```

Run both tenants concurrently from another terminal:

```bash
QUESTION="How are customer exports encrypted?" \
  jac run jac_programs/research_agent.jac &

REQUEST="Calculate the error rate and report p95 latency." \
  jac run jac_programs/operations_agent.jac &

wait
```

The example tools sleep for 200–350 ms and the programs leave 200 ms between
successive byLLM calls. These windows make invariant and parameter-extended
prefill behavior observable.

## Baseline

The matched baseline keeps vLLM APC enabled but disables proactive idle serving:

```bash
python start_server.py \
  --no-prefill \
  --model Qwen/Qwen2.5-7B-Instruct \
  --program research:jac_programs/research_agent.jac:8964 \
  --program operations:jac_programs/operations_agent.jac:8965
```

`--profile` runs the startup interference sweep and fills the safe speculative
prefill concurrency table:

```bash
python start_server.py ... --profile
```

## Measured result

One matched concurrent two-tenant trial was run on an RTX 3090 with
Qwen2.5-3B-Instruct. Each condition used a fresh engine. The metric below covers
the four predicted successor calls; entry calls cannot be predicted from an
ongoing predecessor.

| Successor call | Baseline TTFT | Proactive TTFT | Reduction | Cached tokens |
|---|---:|---:|---:|---:|
| `draft_answer` | 34.61 ms | 30.26 ms | 12.6% | 32 → 96 |
| `audit_answer` | 30.39 ms | 12.84 ms | 57.7% | 48 → 96 |
| `analyze_metrics` | 37.58 ms | 29.44 ms | 21.7% | 112 → 352 |
| `write_report` | 19.02 ms | 16.08 ms | 15.5% | 48 → 112 |
| **Total** | **121.60 ms** | **88.62 ms** | **27.1%** | **240 → 656** |

This trial gives a 1.37× successor-TTFT speedup. End-to-end wall time, including
fixed tool sleeps and inter-call intervals, improved by 3.0% for research and
3.8% for operations. This is an initial functional result, not a statistically
powered evaluation; repeat trials and confidence intervals are still needed.

Runtime logs expose the underlying measurements:

```text
[serve] ... duration_ms=<first-token latency> cached_tokens=<hit> prompt_tokens=<total>
[prefill] ... duration_ms=<warm request duration>
```

## Validate

```bash
python -m py_compile runtime/engine.py runtime/server.py start_server.py
jac check jac_programs/research_agent.jac jac_programs/operations_agent.jac
python -m utils.jac_static_parser jac_programs/research_agent.jac
python -m utils.jac_static_parser jac_programs/operations_agent.jac
```

## Repository layout

- `runtime/engine.py` — vLLM serving, prefill, TTFT metrics, and interference
  profiling.
- `runtime/server.py` — multi-tenant dispatch, ReAct state, and idle scheduling.
- `utils/interceptor_receiver.py` — framed TCP transport.
- `utils/jac_static_parser.py` — topology, provenance, and byte-stable prompts.
- `utils/utils.py` — prompt and Jac UniIR helpers.
- `jac_programs/` — validated research and operations tenant examples.
- `start_server.py` — command-line entry point.
- `INTERFACES.md` — detailed protocol and internal interface contract.

## Known limitations

- The idle scheduler currently chooses the most recently admitted active call.
- Flow identity is tied to one backend connection rather than an explicit
  request/workflow ID.
- Stateful callsite bindings assume one active call per Jac client.
- Dynamic registration cannot compile an unknown Jac source because the wire
  registration frame does not contain a source path.
