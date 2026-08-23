# Proactive Prefill Interfaces

## Architecture

```text
Jac program / InterceptorLLM
        │ framed JSON over a persistent TCP connection
        ▼
InterceptorLLMBackend ──► GuardServer event queue
                              ├── real generation ──► ModelEngine / vLLM
                              ├── per-call ReAct task and inbox
                              └── idle monitor ─────► speculative prefill
```

Each configured Jac program has one `InterceptorLLMBackend` and TCP port. All
backends feed one `GuardServer` queue and share one vLLM engine. A backend accepts
one client connection at a time; different backends can serve concurrently.

## Wire protocol

Every frame is a 4-byte big-endian payload length followed by a UTF-8 JSON body.

| Direction | Type | Required fields | Meaning |
|---|---|---|---|
| client → server | `register` | `program_name`, `model_name` | Register a client on its preconfigured backend; `pid` is optional. |
| client → server | `call` | `id`, `key`, `program_name`, `args` | Start a byLLM call. Optional fields: `site`, `pid`, `self`, `call_params`. |
| server → client | `tool_call` | `call`, `name`, `arguments`, `text` | Ask the Jac client to execute one local tool. |
| client → server | `tool_result` | `call`, `content` | Return the local tool result or error text. |
| server → client | `final` | `call`, `output`, `text` | Complete a call. The client parses `output` into its declared Jac type. |
| client → server | `reject` | `call`, `feedback` | Reject a final value that failed typed parsing and request regeneration. |
| client → server | `generate` | `id`, `key`, `messages` | Single-turn generation used by visit routing. |
| server → client | `result` | `id`, `text` | Complete a `generate` request. |
| server → client | `error` | `error` | Report a protocol, generation, or state error; `id` is included when known. |

`key` is `Owner.name` for a method or `name` for a module-level function.
`site` is `file.jac:line`. `args` contains parameter names mapped to Jac/Python
`repr` strings; the server inserts these strings verbatim into prompts.

Optional numeric values may arrive as JSON `null`. The server applies defaults:

- `temperature`: `0.7`
- `max_tokens`: `512`
- `max_react_iterations`: `8`
- `max_output_retries`: `2`

Literal values extracted from `by llm(...)` override the generation defaults.

## Registration and server lifetime

Programs must be configured before serving because a `register` frame does not
contain a source path or listening port:

```python
server.add_program("research", "jac_programs/research_agent.jac", 8964)
await server.serve()
```

`GuardServer.serve()` starts every TCP listener, the incoming-event monitor, and,
unless disabled, the engine-idle monitor. Registration verifies the program name,
associates the backend connection with the configured topology, and initializes
its completed-call set.

## Request dispatch and ReAct

`monitor_task()` never waits for a complete model workflow. It continuously
drains the shared queue:

- `call` starts an independent `_process_request()` task.
- `tool_result` and `reject` are routed by `(backend identity, call id)` into that
  task's private inbox.
- `generate` starts an independent single-turn task.

A ReAct task can therefore suspend on `await state.inbox.get()` while the monitor
admits messages and calls from other tenants.

```text
call → generate → tool_call → await tool_result → generate → ... → final
```

The text tool protocol requests:

```text
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>
```

For model compatibility, the server also accepts a bare JSON tool object and
consumes only its first JSON value. If a tool-enabled model returns non-empty
plain text instead of calling `finish_tool`, the server treats that text as
`finish_tool.final_output`; the Jac client still performs typed validation. An
empty response is an error.

After `final`, a `reject` regenerates within `max_output_retries`. The next `call`
on the same backend is the implicit acknowledgement of the preceding final.

## Static topology and prompt interface

```python
program = ProgramTopology(program_name, src_path)
program.parse_dependency()
```

Important queries:

- `site_of(key, site)` resolves the exact invocation site, falling back to the
  declaration's first site when source location is absent or unmatched.
- `next_calls(key)` returns an over-approximated set of reachable successor keys.
- `ready_params(consumer, done)` reports parameter readiness. Constants and
  fields are statically ready; `ret` and `ret_item` become ready when their
  producer key is in `done`.
- `consumers_of(producer)` returns `(consumer, parameter)` dataflow edges.

`ByLLMCallsite` owns byte-stable prompt construction:

- `render_invariant_prompt()` renders the signature and semantic schema.
- `incremental_prompt(values)` binds known argument `repr` strings.
- `get_ready_prompt()` returns the invariant system/user prefix plus every bound
  parameter already available.
- `assemble_prompt(args, self_view)` builds the real prompt as an extension of
  that warm prefix.
- `clear_bindings()` resets served speculative state.

Prefix identity is essential: vLLM APC reuses KV blocks only when the served
token prefix matches the speculative token prefix.

## Idle-prefill interface

`ModelEngine.is_engine_idle()` admits speculative work when either:

- the engine has no unfinished request (temporal idle), or
- the current decode concurrency has an unused profiled prefill slot (spatial
  idle).

`monitor_idle()` currently selects the most recently admitted call. For each
eligible slot it:

1. Looks up successors with `next_calls(current_site.key)`.
2. Calls `ready_params(successor, state.done)`.
3. Binds ready constants and retains return parameters propagated by completed
   producers.
4. Renders `successor_site.get_ready_prompt()`.
5. Submits a priority-1, one-token generation through `ModelEngine.prefill()`.

vLLM manages KV-cache lookup and eviction through APC. `prefilled_sites` only
prevents duplicate scheduling of the same application-level callsite state
within one active call; it is not a KV-cache index.

When a call produces a final value, its key enters the backend's `done` set and
the value is bound into every return consumer. A rejected final removes the key
from `done`.

Use `--no-prefill` to disable `monitor_idle()` while leaving APC and all on-demand
serving enabled. This is the baseline mode.

## Model engine

`ModelEngine` enables vLLM prefix caching, chunked prefill, and priority
scheduling. Its runtime operations are:

- `render(messages)` — apply the model's chat template.
- `generate(prompt, request_id, sampling_params)` — serve a real request.
- `prefill(prompt, request_id)` — submit a low-priority, one-token cache warm.
- `is_engine_idle()` — test temporal/spatial prefill capacity.
- `_profile(...)` — populate `max_parallelizable_prefill[decode_count]` and stop
  when no concurrent prefill is safe.

The first output of every real generation logs first-token latency, cached prompt
tokens, and total prompt tokens. Prefill submissions log their duration.

## Current limitations

- Ongoing-call selection is most-recent-first; no fairness or probability policy
  is implemented yet.
- Completed-call state is scoped to a backend connection, not an explicit
  workflow/flow identifier.
- `ByLLMCallsite.bound_args` is stateful and assumes the backend's documented
  single-call client behavior.
- Traversal-queue continuation for walker/node abilities is not modeled in the
  rewritten topology pass.
- Registration activates preconfigured programs; it cannot dynamically compile
  an unknown source program.
