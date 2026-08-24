
## multi-tenant cold start: per-trial ΣTTFT per tenant (ms)

| tenant | off p25/p50/p75 | spec p25/p50/p75 | speedup (p50) |
|---|---|---|---|
| dispatch | 166/188/210 | 164/186/222 | 1.01x |
| moderation | 103/134/180 | 114/131/166 | 1.02x |
| operations | 222/238/279 | 195/221/275 | 1.07x |
| pipeline | 210/226/249 | 206/225/335 | 1.01x |
| research | 179/192/220 | 178/194/230 | 0.99x |
| route | 169/236/271 | 165/242/272 | 0.98x |
| support_email | 178/193/205 | 175/182/187 | 1.06x |
| **all** | p50 203 | p50 189 | 1.07x |

## single-tenant cold start: per-trial ΣTTFT per tenant (ms)

| tenant | off p25/p50/p75 | spec p25/p50/p75 | speedup (p50) |
|---|---|---|---|
| dispatch | 118/119/160 | 115/116/138 | 1.03x |
| moderation | 61/89/104 | 54/59/72 | 1.50x |
| operations | 135/146/199 | 105/115/160 | 1.26x |
| pipeline | 119/120/124 | 96/100/114 | 1.21x |
| research | 159/161/170 | 149/156/164 | 1.03x |
| route | 184/188/203 | 156/158/194 | 1.19x |
| support_email | 130/131/133 | 76/76/81 | 1.72x |
| **all** | p50 131 | p50 112 | 1.17x |

## Per-callsite first-serve TTFT median (ms) / median cached tokens

| tenant:site | multi-off | multi-spec | single-off | single-spec |
|---|---|---|---|---|
| dispatch:__visit | 53 / 16 | 54 / 16 | 35 / 32 | 35 / 32 |
| dispatch:debrief | 29 / 48 | 31 / 48 | 18 / 48 | 13 / 112 |
| dispatch:resolve_auth | 32 / 48 | 31 / 48 | 18 / 48 | 17 / 96 |
| dispatch:resolve_database | 52 / 112 | 53 / 112 | 39 / 112 | 19 / 336 |
| dispatch:resolve_frontend | 20 / 48 | 20 / 48 | 18 / 48 | 16 / 96 |
| dispatch:resolve_network | 33 / 48 | 33 / 48 | 19 / 48 | 16 / 96 |
| moderation:archive_note | 35 / 48 | 34 / 48 | 20 / 48 | 18 / 96 |
| moderation:classify_content | 36 / 48 | 37 / 48 | 20 / 48 | 20 / 48 |
| moderation:draft_warning | 37 / 48 | 31 / 48 | 23 / 48 | 19 / 112 |
| moderation:escalate_case | 51 / 112 | 47 / 112 | 44 / 112 | 19 / 352 |
| moderation:log_decision | 37 / 48 | 35 / 48 | 19 / 48 | 15 / 112 |
| operations:analyze_metrics | 52 / 112 | 49 / 112 | 41 / 112 | 15 / 400 |
| operations:collect_metrics | 50 / 112 | 48 / 112 | 35 / 112 | 35 / 112 |
| operations:write_report | 25 / 48 | 30 / 48 | 19 / 48 | 16 / 112 |
| pipeline:classify_record | 32 / 48 | 34 / 48 | 18 / 48 | 14 / 128 |
| pipeline:extract_fields | 56 / 48 | 44 / 48 | 20 / 48 | 20 / 48 |
| pipeline:format_output | 39 / 48 | 39 / 48 | 25 / 48 | 20 / 184 |
| pipeline:summarize_record | 33 / 48 | 32 / 48 | 19 / 48 | 14 / 144 |
| pipeline:translate_summary | 30 / 48 | 33 / 48 | 18 / 48 | 13 / 112 |
| pipeline:validate_fields | 33 / 48 | 34 / 48 | 19 / 48 | 13 / 128 |
| research:audit_answer | 31 / 48 | 34 / 48 | 19 / 48 | 13 / 104 |
| research:draft_answer | 35 / 48 | 33 / 48 | 19 / 32 | 15 / 128 |
| research:investigate | 58 / 112 | 56 / 112 | 84 / 0 | 87 / 0 |
| route:__visit | 43 / 16 | 48 / 16 | 34 / 0 | 34 / 0 |
| route:resolve_billing | 44 / 112 | 51 / 112 | 40 / 112 | 17 / 336 |
| route:resolve_tech | 44 / 48 | 30 / 48 | 30 / 48 | 15 / 96 |
| route:summarize | 30 / 48 | 30 / 48 | 18 / 48 | 13 / 128 |
| support_email:classify_email | 34 / 48 | 34 / 48 | 20 / 48 | 20 / 48 |
| support_email:compose_reply | 80 / 112 | 76 / 112 | 71 / 112 | 21 / 656 |
| support_email:polish_reply | 37 / 48 | 35 / 48 | 18 / 48 | 13 / 112 |

## Speedup: single-task vs multi-tenant (p50 ΣTTFT, off/spec)

| tenant | single speedup | multi speedup | retained under concurrency |
|---|---|---|---|
| dispatch | 1.03x | 1.01x | 52% |
| moderation | 1.50x | 1.02x | 5% |
| operations | 1.26x | 1.07x | 28% |
| pipeline | 1.21x | 1.01x | 3% |
| research | 1.03x | 0.99x | -22% |
| route | 1.19x | 0.98x | -13% |
| support_email | 1.72x | 1.06x | 8% |

## Cache hit and TBT

- multi-off: serves=273 hit=45104/83188 (54.2%) mean_TBT=10.56ms
- multi-spec: serves=268 hit=45456/80207 (56.7%) mean_TBT=10.99ms
- single-off: serves=273 hit=45088/83276 (54.1%) mean_TBT=9.15ms
- single-spec: serves=273 hit=62768/83194 (75.4%) mean_TBT=9.14ms
