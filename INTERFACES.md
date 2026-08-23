# InterceptorLLM Interfaces

## Wire protocol

4-byte big-endian length + UTF-8 JSON body, over one persistent TCP connection.
One dedicated `InterceptorLLMBackend` per Jac program; no concurrent calls on one
`InterceptorLLM` instance (hard error). The server drives the ReAct loop.

| Dir | type | fields | meaning |
|-----|------|--------|---------|
| c→s | `register` | `program_name, pid, model_name` | once, right after connect |
| c→s | `call` | `id, key, site: "file.jac:line"\|null, program_name, pid, args:{name: repr}, self: repr\|null, call_params` | start one byllm call (plain calls too: zero tool rounds); `site` is the invocation location, resolved server-side via `site_of` |
| s→c | `tool_call` | `call, name, arguments, text` | execute this tool locally; strictly one at a time |
| c→s | `tool_result` | `call, content` | tool output (or error text) |
| s→c | `final` | `call, output, text` | end of call; client parses `output` into the declared type |
| c→s | `reject` | `call, feedback` | `final` failed typed parsing; server regenerates (budget = `max_output_retries`) |
| c→s | `generate` | `id, key, messages, schema, temperature, max_tokens, stop` | visit routing only: single turn, full messages |
| s→c | `result` | `id, text` | reply to `generate` |
| s→c | `error` | `error` (+ `id`) | any failure |

`key` = `Owner.name` or `name` (empty for visit routing). `text` on server frames is
the raw generated text, recorded client-side for conversation write-back.
`call_params` carries `max_react_iterations` (server enforces; server must produce
a `final` on overflow) and `max_output_retries`.

## Jac client — `InterceptorLLM` (`jaseci/jac/jaclang/byllm/llm.jac`)

```jac
glob llm = InterceptorLLM(model_name=..., comm_ip="localhost", comm_port=8964, program_name=...);  # default: basename(sys.argv[0])
```

Connects + registers in `postinit`. Overrides `_invoke_react_loop` with the
`call`/`tool_call`/`final` state machine; only tool execution and typed-result
construction stay client-side. Falls back to a plain `Model` when streaming or
when the server is unreachable; reconnects on the next call after a drop.

## Server — `InterceptorLLMBackend` (`utils/interceptor_receiver.py`)

```python
queue: asyncio.Queue = asyncio.Queue()
backends = [InterceptorLLMBackend(name, queue, comm_ip="localhost", comm_port=port), ...]
await asyncio.gather(*(b.listen() for b in backends), guard.run(queue))
```

Asyncio, pure transport: the backend parses every frame into a typed
`byLLMRequest` (a bad frame is answered with an error frame and never reaches
the queue), validates `register`, then puts `(backend, request)` on the shared
queue and goes back to reading. The guard server consumes the queue, keeps
per-call conversation state (keyed by backend + `request.id`/`request.call` —
`tool_result`/`reject` arrive as their own queue events), and replies with
`await backend.send(frame)`; a `generate` reply must echo the request's `id`.

`byLLMRequest`: `type`, `pid` (-1 when the frame carries none), and per-type
optionals — `id`/`key`/`args`/`nest_scope_desc` (wire `self`)/`call_params`
(call), `call`/`content` (tool_result), `call`/`feedback` (reject),
`program_name`/`model_name` (register), `messages`/`schema`/`temperature`/
`max_tokens`/`stop` (generate). `byLLMRequest.from_frame(msg)` parses and
validates; `req.to_frame()` reconstructs the wire dict. While a call waits for its tool_result the loop is free (the
proactive-prefill window); blocking generation must go through `run_in_executor`
or an async engine. A second concurrent connection is rejected; a program-name
mismatch on `register` shuts the backend down.

## Static pass — `utils/jac_static_parser.py`

```python
p = ProgramTopoly(program_name, src_path); p.parse_dependency()
```

- `p.decls: {key: ByLLMDecl}` — every `def ... by llm()`: params (name/type/sem),
  return type, sems, OpenAI tool schemas (finish_tool last), literal call_params.
- `p.callsites: [ByLLMCallsite]`, `p.sites_of(key)`
- `p.next_calls(key)` — may-run-next keys (over-approximated topology)
- `p.ready_params(consumer, done)` — per-param readiness given completed keys
  (`const`/`field` ready now; `ret`/`ret_item` ready iff producer in `done`)
- `p.consumers_of(producer)` — `[(consumer key, param)]` dataflow

`ByLLMCallsite` (stateful warm/serve rendering; `args`/`self_view` are wire
reprs, inserted verbatim; resolve a request's site with
`p.site_of(req.key, req.site)` — unmatched/absent `site` falls back to the
decl's first site, whose serve prompt is identical):

- `invariant_system` — persona + system_prompt + tool instruction + text tool protocol
- `render_invariant_prompt()` — signature header + sem'd param schema rows
- `incremental_prompt(known_args)` — binds ready param reprs into `bound_args`
  (bind order = warm prefix order; already-bound names skipped)
- `get_ready_prompt()` — `[system, user-prefix]` of what is known now; the
  server prefills the KV cache with exactly this
- `assemble_prompt(args, self_view=None)` — `[system, user]` for serving, as a
  byte-extension of the warmed prefix: bound params keep their position and
  bytes (a conflicting request value wins in place), remaining params appended
  in declaration order
- `clear_bindings()` — reset warm state after the call is served

Where the callsite's other two duties live:

**Duty 2 — who consumes this site's return (dataflow)**: data on the callsite,
computation in the parser. `self.consumers` (`jac_static_parser.py:54`) is a
callsite attribute, filled by `parse_dependency` at :148 from
`_invert_provenance` (:406, the inversion of `_build_provenance`'s (:383)
param→source table). Equivalent query entry: `JacStaticParser.consumers_of(producer)` (:436).

**Duty 3 — which callsites' params become ready after this call (scheduling)**:
not on `ByLLMCallsite` yet; split across two parser queries:
- `next_calls(key)` (:418) — which byllm calls may run next in the topology;
- `ready_params(consumer, done)` (:423) — per-param readiness of a consumer given
  the completed set (`const`/`field` derivable now; `ret`/`ret_item` ready iff the
  producer is in `done`).

`utils/utils.py`: `schema_entry`, `render_bindings` (never re-reprs),
`render_tool`/`format_tools_for_prompt` (byte-identical to byllm's text tool
protocol), `finish_tool_schema` (param `final_output`, matching byllm), plus the
UniIR extraction helpers (`params_of`, `extract_llm_call`, `literal_of`, ...).


