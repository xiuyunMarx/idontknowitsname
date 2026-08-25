# Multi-tenant blend: `load-uniform-w3-iso`

blend: benchmark/evaluation/blends/uniform-w3.json — 12 workflows/trial, mode=multi (uniform over 3.0s), ratio {'hover': 0.3333333333333333, 'intercode_sql': 0.3333333333333333, 'deep_search': 0.3333333333333333}, seed 20260824; workers=16, model=Qwen/Qwen2.5-7B-Instruct, cache isolation=instance

## per cell

| cell | trials | workflows (rc=0) | mean / peak concurrency | makespan s | warm fraction | call TTFT p50 / p95 / p99 ms | mean TBT ms | spec prefills/trial | aborted/trial |
|---|---|---|---|---|---|---|---|---|---|
| off | 3 | 36/36 | 6.7 / 12 | 33 | 37.8% | 145 / 329 / 407 | 31.26 | 0 | 0 |
| spec | 3 | 36/36 | 7.0 / 12 | 32 | 38.7% | 142 / 311 / 401 | 32.16 | 129 | 45 |

## spec vs off: paired per workflow (same trial, same order)

| tenant | pairs (same seq / all) | uncached tok/workflow base→cell | Δuncached | ΣTTFT ms base→cell | ΔΣTTFT [95% CI] | ratio | per-call p95 base→cell | wall s base→cell |
|---|---|---|---|---|---|---|---|---|
| hover | 12 / 12 | 2608→2503 | +105 | 1608→1553 | +55.0 [-84.0, +201.7] | 1.04x | 291→266 | 20.3→20.4 |
| intercode_sql | 12 / 12 | 1876→1871 | +5 | 918→849 | +69.5 [-47.5, +177.6] | 1.08x | 303→302 | 7.7→7.8 |
| deep_search | 8 / 12 | 3829→3808 | +22 | 2330→2236 | +93.7 [-98.1, +291.3] | 1.04x | 327→298 | 28.7→28.2 |
| **all** | 32 / 36 | 2771→2727 | **+44** | 1619→1546 | **+72.8 [-15.4, +162.5]** | **1.05x** | 307→289 | 18.9→18.8 |

## per callsite (mean uncached tokens / mean TTFT ms)

| callsite | n | off | spec |
|---|---|---|---|
| deep_search:DataAnalysisAgent.data_research | 27 | 265 / 157 | 282 / 166 |
| deep_search:DeepResearch.build_report | 48 | 151 / 90 | 120 / 78 |
| deep_search:DeepResearch.summarize | 24 | 277 / 171 | 282 / 154 |
| deep_search:FactCheckAgent.fact_check | 25 | 234 / 148 | 263 / 140 |
| deep_search:PaperSearchAgent.paper_research | 48 | 269 / 167 | 276 / 165 |
| deep_search:ResearchSupervisor.decompose_task | 48 | 260 / 170 | 259 / 150 |
| deep_search:WebSearchAgent.web_research | 44 | 280 / 164 | 290 / 175 |
| deep_search:__visit | 96 | 284 / 168 | 274 / 161 |
| hover:HoverAgent.decompose_claim | 24 | 143 / 127 | 143 / 118 |
| hover:SearchData.plan_query | 48 | 156 / 148 | 151 / 131 |
| hover:SearchData.reason_hop | 48 | 185 / 145 | 182 / 155 |
| hover:SearchData.retrieve_evidence | 48 | 282 / 156 | 276 / 144 |
| hover:SummarizeData.assess_gap | 48 | 182 / 124 | 176 / 131 |
| hover:Verdict.synthesize_answer | 24 | 380 / 161 | 314 / 139 |
| hover:Verdict.verify_claim | 24 | 476 / 174 | 477 / 174 |
| intercode_sql:diagnose | 24 | 179 / 210 | 179 / 159 |
| intercode_sql:finalize | 24 | 138 / 195 | 138 / 172 |
| intercode_sql:generate_sql | 24 | 902 / 251 | 902 / 254 |
| intercode_sql:revise_sql | 24 | 657 / 263 | 652 / 265 |
