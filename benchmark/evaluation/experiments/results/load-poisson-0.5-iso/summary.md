# Multi-tenant blend: `load-poisson-0.5-iso`

blend: benchmark/evaluation/blends/poisson-0.5.json — 12 workflows/trial, mode=multi (poisson 0.5/s), ratio {'hover': 0.3333333333333333, 'intercode_sql': 0.3333333333333333, 'deep_search': 0.3333333333333333}, seed 20260824; workers=16, model=Qwen/Qwen2.5-7B-Instruct, cache isolation=instance

## per cell

| cell | trials | workflows (rc=0) | mean / peak concurrency | makespan s | warm fraction | call TTFT p50 / p95 / p99 ms | mean TBT ms | spec prefills/trial | aborted/trial |
|---|---|---|---|---|---|---|---|---|---|
| off | 3 | 36/36 | 3.6 / 8 | 54 | 37.4% | 130 / 232 / 355 | 23.73 | 0 | 0 |
| spec | 3 | 36/36 | 3.6 / 8 | 54 | 45.5% | 107 / 231 / 340 | 24.62 | 394 | 70 |

## spec vs off: paired per workflow (same trial, same order)

| tenant | pairs (same seq / all) | uncached tok/workflow base→cell | Δuncached | ΣTTFT ms base→cell | ΔΣTTFT [95% CI] | ratio | per-call p95 base→cell | wall s base→cell |
|---|---|---|---|---|---|---|---|---|
| hover | 12 / 12 | 2577→2178 | +399 | 1406→1152 | +253.4 [+119.0, +408.0] | 1.22x | 221→201 | 17.3→17.1 |
| intercode_sql | 12 / 12 | 1873→1764 | +109 | 630→618 | +12.8 [-27.9, +45.2] | 1.02x | 240→264 | 5.8→5.8 |
| deep_search | 9 / 12 | 3921→3349 | +572 | 1845→1786 | +59.0 [-51.4, +168.1] | 1.03x | 234→211 | 26.2→26.4 |
| **all** | 33 / 36 | 2790→2430 | **+360** | 1294→1185 | **+108.4 [+40.5, +181.8]** | **1.09x** | 232→226 | 16.4→16.4 |

## per callsite (mean uncached tokens / mean TTFT ms)

| callsite | n | off | spec |
|---|---|---|---|
| deep_search:DataAnalysisAgent.data_research | 26 | 292 / 131 | 192 / 104 |
| deep_search:DeepResearch.build_report | 48 | 149 / 97 | 129 / 93 |
| deep_search:DeepResearch.summarize | 24 | 282 / 122 | 254 / 126 |
| deep_search:FactCheckAgent.fact_check | 26 | 256 / 112 | 186 / 94 |
| deep_search:PaperSearchAgent.paper_research | 48 | 275 / 119 | 230 / 127 |
| deep_search:ResearchSupervisor.decompose_task | 48 | 259 / 121 | 257 / 122 |
| deep_search:WebSearchAgent.web_research | 44 | 291 / 139 | 220 / 122 |
| deep_search:__visit | 96 | 286 / 133 | 262 / 134 |
| hover:HoverAgent.decompose_claim | 24 | 143 / 99 | 143 / 89 |
| hover:SearchData.plan_query | 48 | 157 / 106 | 120 / 84 |
| hover:SearchData.reason_hop | 48 | 177 / 135 | 150 / 87 |
| hover:SearchData.retrieve_evidence | 48 | 279 / 121 | 238 / 124 |
| hover:SummarizeData.assess_gap | 48 | 185 / 132 | 141 / 90 |
| hover:Verdict.synthesize_answer | 24 | 365 / 166 | 294 / 126 |
| hover:Verdict.verify_claim | 24 | 473 / 153 | 442 / 169 |
| intercode_sql:diagnose | 24 | 179 / 105 | 157 / 93 |
| intercode_sql:finalize | 24 | 137 / 94 | 124 / 78 |
| intercode_sql:generate_sql | 24 | 902 / 233 | 902 / 247 |
| intercode_sql:revise_sql | 24 | 655 / 199 | 580 / 200 |
