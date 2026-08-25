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
| client → server | `generate` | `id`, `key`, `messages` | Single-turn generation used by visit routing. Optional fields: `site`, `pid`, `schema`, sampling overrides. |
| server → client | `result` | `id`, `text` | Complete a `generate` request. |
| server → client | `error` | `error` | Report a protocol, generation, or state error; `id` is included when known. |

`key` is `Owner.name` for a method, `name` for a module-level function, and
`__visit@<file>.jac:<line>` for a `visit <edges> by llm()` routing call. A routing
call has no callable behind it, so both ends derive its key from source location:
the static pass from the `VisitStmt`, the client from the `.jac` stack frame
executing the visit. `site` is `file.jac:line`. `args` contains parameter names
mapped to Jac/Python `repr` strings; the server inserts these strings verbatim
into prompts.

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

`GuardServer.serve()` first warms the process (chat template compiled, every
callsite's invariant prefix tokenized, a few random-token requests through the
engine so CUDA initialisation and AsyncLLM's one-off RPCs are paid), then starts
every TCP listener, the incoming-event monitor, and, unless disabled, the
engine-idle monitor. `listening on` is printed only after the warmup, so a
harness that waits for it starts tenants against an already-serving engine.
A real admission kills the speculative request in flight (`kill_speculation`):
a queued one is dropped, one already in a running step finishes that step.
With `--cache-isolation instance` every client connection (one workflow
instance) gets its own vLLM `cache_salt`; its real requests, speculative
prefills and route probes carry it, so no KV block is ever shared across
instances and an instance's cache is dead once it disconnects. The default
(`none`) leaves the prefix cache shared, so instances of the same program reuse
each other's invariant prefixes. Registration verifies the program name,
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
  Inside one ability body control flow is followed: the first byLLM call of
  what comes next, through branches (each branch's first call; a call-free
  branch continues past the `if`), loops (back edge to the loop body's first
  call, then the loop's exit) and `return`. Across abilities the graph is followed: the
  **last** byLLM call of a body (one after which no unconditional byLLM call
  remains) is succeeded by the **first** byLLM call of the abilities that fire
  when the walker arrives at the nodes the body visited — plain `visit` and
  `visit ... by llm()` alike, node-side (`can x with W entry` in the node) and
  walker-side (`can x with N entry` in the walker). The walker's queue order is
  kept: a body's first unconditional visit is dequeued first; an ability reached
  by visit V is followed by V's other candidate types (a `select>1` router) and
  by the targets of the visits after V in the same body; the walker's exit
  abilities close every traversal.
- `ready_params(consumer, done, via=None)` reports parameter readiness. Constants
  and fields are statically ready; `ret` and `ret_item` become ready when their
  producer key is in `done`, `ret_any` when any of its producers is. `via` names
  the call control passes through to reach `consumer`; when it is a routing call,
  node-scoped sources are withheld (see below).
- `consumers_of(producer)` returns `(consumer, parameter)` dataflow edges.

### Visit routing

A `visit <edges> by llm(...)` gets a synthesized decl with `kind == "visit"`,
keyed by location. Its compile-time invariant is byllm's routing system text
(fixed by `select=`) plus `Goal: <intent>`, which `route_visit` emits ahead of
every runtime zone under both layouts. The three runtime zones — `walker`,
`here`, `candidates` — are bound through `incremental_prompt()` like parameters,
in the order `route_visit` will emit them.

- `decl.candidates` over-approximates the node archetypes the router may pick:
  the edge expression's node filter, else every node type whose entry abilities
  can fire for this walker.
- `next_calls(visit_key)` returns the **first byLLM call of each candidate
  node's firing arrival abilities** (node side and walker side), reached
  through the enclosing body's queue order as described above.
- Provenance carries a `binding`: `walker` for walker state (`self.x` in a walker
  ability, `visitor.x` in a node ability) and `node` for node state (`here.x` in
  a walker ability, `self.x` in a node ability). Walker state rides through a
  traversal step unchanged; node state belongs to whichever node the router picks
  and does not exist until it answers, so `ready_params(..., via=<visit key>)`
  withholds it.
- A field written by a byLLM call (`visitor.answer = classify(...)`) is indexed
  as a dataflow producer, which is what makes a result reach a consumer in
  another ability — a local never crosses an ability boundary.

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
2. Calls `ready_params(successor, state.done, via=current_site.key)`.
3. Binds ready constants and retains return parameters propagated by completed
   producers.
4. Renders `successor_site.get_ready_prompt()`.
5. Submits a priority-1, one-token generation through `ModelEngine.prefill()`.

vLLM manages KV-cache lookup and eviction through APC. `prefilled_sites` only
prevents duplicate scheduling of the same application-level callsite state
within one active call; it is not a KV-cache index.

When a call produces a final value, its key enters the backend's `done` set and
the value is bound into every return consumer. A rejected final removes the key
from `done`. A routing `generate` is tracked the same way: it is registered as
the current call before generation, its key enters `done` when the router
answers, and the state survives until the client's next frame — which is exactly
the idle window in which the candidate nodes' first calls get warmed.

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
