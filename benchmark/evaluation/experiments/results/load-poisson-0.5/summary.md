# Multi-tenant blend: `load-poisson-0.5`

blend: benchmark/evaluation/blends/poisson-0.5.json — 12 workflows/trial, mode=multi (poisson 0.5/s), ratio {'hover': 0.3333333333333333, 'intercode_sql': 0.3333333333333333, 'deep_search': 0.3333333333333333}, seed 20260824; workers=16, model=Qwen/Qwen2.5-7B-Instruct

## per cell

| cell | trials | workflows (rc=0) | mean / peak concurrency | makespan s | warm fraction | call TTFT p50 / p95 / p99 ms | mean TBT ms | spec prefills/trial | aborted/trial |
|---|---|---|---|---|---|---|---|---|---|
| off | 3 | 36/36 | 3.5 / 8 | 52 | 58.2% | 84 / 167 / 254 | 21.86 | 0 | 0 |
| spec | 3 | 36/36 | 3.6 / 8 | 51 | 62.6% | 79 / 159 / 276 | 22.02 | 422 | 82 |

## spec vs off: paired per workflow (same trial, same order)

| tenant | pairs (same seq / all) | uncached tok/workflow base→cell | Δuncached | ΣTTFT ms base→cell | ΔΣTTFT [95% CI] | ratio | per-call p95 base→cell | wall s base→cell |
|---|---|---|---|---|---|---|---|---|
| hover | 12 / 12 | 1640→1488 | +153 | 966→905 | +61.5 [-0.2, +120.0] | 1.07x | 164→134 | 16.1→16.3 |
| intercode_sql | 12 / 12 | 1312→1254 | +58 | 477→482 | -5.9 [-42.3, +28.3] | 0.99x | 192→199 | 5.5→5.5 |
| deep_search | 5 / 12 | 2656→2257 | +400 | 1391→1342 | +48.4 [-21.7, +116.0] | 1.04x | 154→160 | 23.7→23.8 |
| **all** | 29 / 36 | 1869→1666 | **+203** | 944→910 | **+34.7 [-0.5, +70.6]** | **1.04x** | 170→164 | 15.1→15.2 |

## per callsite (mean uncached tokens / mean TTFT ms)

| callsite | n | off | spec |
|---|---|---|---|
| deep_search:DataAnalysisAgent.data_research | 27 | 190 / 98 | 98 / 69 |
| deep_search:DeepResearch.build_report | 49 | 122 / 77 | 123 / 79 |
| deep_search:DeepResearch.summarize | 24 | 167 / 85 | 124 / 84 |
| deep_search:FactCheckAgent.fact_check | 27 | 158 / 76 | 101 / 73 |
| deep_search:PaperSearchAgent.paper_research | 48 | 147 / 75 | 113 / 83 |
| deep_search:ResearchSupervisor.decompose_task | 48 | 168 / 95 | 163 / 94 |
| deep_search:WebSearchAgent.web_research | 42 | 150 / 84 | 109 / 73 |
| deep_search:__visit | 96 | 238 / 115 | 230 / 114 |
| hover:HoverAgent.decompose_claim | 24 | 59 / 58 | 59 / 58 |
| hover:SearchData.plan_query | 48 | 127 / 79 | 108 / 74 |
| hover:SearchData.reason_hop | 48 | 130 / 78 | 120 / 74 |
| hover:SearchData.retrieve_evidence | 48 | 105 / 74 | 93 / 73 |
| hover:SummarizeData.assess_gap | 48 | 152 / 92 | 121 / 83 |
| hover:Verdict.synthesize_answer | 24 | 289 / 130 | 283 / 124 |
| hover:Verdict.verify_claim | 24 | 263 / 135 | 260 / 115 |
| intercode_sql:diagnose | 24 | 81 / 57 | 59 / 54 |
| intercode_sql:finalize | 24 | 88 / 63 | 75 / 66 |
| intercode_sql:generate_sql | 24 | 580 / 185 | 580 / 181 |
| intercode_sql:revise_sql | 24 | 562 / 171 | 540 / 181 |
