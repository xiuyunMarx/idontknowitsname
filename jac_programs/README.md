# Multi-tenant byLLM example

Start the proactive-prefill server:

```bash
python start_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --program research:jac_programs/research_agent.jac:8964 \
  --program operations:jac_programs/operations_agent.jac:8965 \
  --profile
```

Run the tenants in separate terminals:

```bash
QUESTION="How are customer exports encrypted?" jac run jac_programs/research_agent.jac
REQUEST="Calculate the error rate and report p95 latency." jac run jac_programs/operations_agent.jac
```

The tools intentionally sleep for 200–350 ms, and each workflow leaves another
200 ms between byLLM calls. Tool windows expose invariant prefill opportunities;
inter-call windows let the server extend those prefixes with completed returns.
