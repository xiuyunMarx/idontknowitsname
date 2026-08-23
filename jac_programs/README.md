# Multi-tenant byLLM example

Start the proactive-prefill server:

```bash
python start_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --program research:jac_programs/research_agent.jac:8964 \
  --program operations:jac_programs/operations_agent.jac:8965 \
  --program route:jac_programs/route_wire.jac:8966 \
  --profile
```

Run the tenants in separate terminals:

```bash
QUESTION="How are customer exports encrypted?" jac run jac_programs/research_agent.jac
REQUEST="Calculate the error rate and report p95 latency." jac run jac_programs/operations_agent.jac
TICKET="Our export job fails with a 500 after the last release." jac run jac_programs/route_wire.jac
```

The tools intentionally sleep for 200–350 ms, and each workflow leaves another
200 ms between byLLM calls. Tool windows expose invariant prefill opportunities;
inter-call windows let the server extend those prefixes with completed returns.

`route_wire.jac` is the visit-routing tenant. A `support` walker reaches a
`Triage` node and picks its next hop with `visit [-->] by llm(...)`; each desk
node answers with its own byLLM call, and the walker's exit ability summarizes
whichever one ran.

The routing call is keyed by source location (`__visit@route_wire.jac:92`), so
the server can place it in the topology. Its compile-time invariant is the
routing system prompt plus `Goal: <intent>` — the walker, the current node and
the candidate list only exist once the graph does. Its successors are the first
byLLM call of *every* candidate desk, since the choice is not known until the
router answers, which is what the speculative prefill covers:

```
__visit@route_wire.jac:92 -> resolve_billing, resolve_tech, resolve_security, summarize
```

Parameter readiness across that edge is scope-aware. `visitor.ticket` is walker
state and rides through the traversal; `self.policy` belongs to whichever desk
the router picks and is withheld until it has answered.
