# Evaluation-tenant smoke: `smoke-check`

trials=[0]; cells=['spec']

## single

### per-cell totals

| cell | trials | serves | warm fraction | mean TBT ms | spec prefills/trial | aborted/trial | spec tokens/trial |
|---|---|---|---|---|---|---|---|
| spec | 1 | 38 | 66.2% | 19.68 | 179 | 9 | 11353 |

### per tenant (workflows with rc=0; uncached = prompt − cached, summed over the workflow)

| tenant | cell | n | calls/workflow | uncached tok/workflow | ΣTTFT ms | wall s |
|---|---|---|---|---|---|---|
| hover | spec | 1 | 11.0 | 2920 | 895 | 21.5 |
| deep_search | spec | 0 | nan | nan | nan | nan |
| intercode_sql | spec | 1 | 4.0 | 1531 | 388 | 10.5 |

### single: spec vs off, paired per (trial, tenant)

| tenant | pairs (same sequence / all) | Δuncached tok/workflow (base−cell) | ΔΣTTFT ms (base−cell) | ΣTTFT ratio |
|---|---|---|---|---|

### single: per callsite (mean uncached tokens / mean TTFT ms)

| callsite | n | spec |
|---|---|---|
| deep_search:DataAnalysisAgent.data_research | 1 | 77 / 36 |
| deep_search:FactCheckAgent.fact_check | 1 | 76 / 35 |
| deep_search:PaperSearchAgent.paper_research | 1 | 77 / 39 |
| deep_search:ResearchSupervisor.decompose_task | 18 | 61 / 47 |
| deep_search:__visit | 2 | 302 / 116 |
| hover:HoverAgent.decompose_claim | 1 | 166 / 59 |
| hover:SearchData.plan_query | 2 | 254 / 82 |
| hover:SearchData.reason_hop | 2 | 140 / 48 |
| hover:SearchData.retrieve_evidence | 2 | 269 / 89 |
| hover:SummarizeData.assess_gap | 2 | 192 / 62 |
| hover:Verdict.synthesize_answer | 1 | 453 / 135 |
| hover:Verdict.verify_claim | 1 | 592 / 140 |
| intercode_sql:diagnose | 1 | 51 / 28 |
| intercode_sql:finalize | 1 | 44 / 22 |
| intercode_sql:generate_sql | 1 | 717 / 173 |
| intercode_sql:revise_sql | 1 | 719 / 166 |
