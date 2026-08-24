## Overall

| condition | serves | TTFT p50 (ms) | TTFT p95 (ms) | cache hit | mean TBT (ms) | spec prefills | spec tokens | E2E p50 (s) |
|---|---|---|---|---|---|---|---|---|
| off | 274 | 9.3 | 15.5 | 82.8% | 2.24 | 0 | 0 | 2.8 |
| newest-idle | 270 | 9.4 | 15.2 | 84.8% | 2.24 | 215 | 30843 | 2.8 |
| global-idle | 270 | 9.2 | 16.2 | 88.1% | 2.21 | 339 | 42956 | 2.8 |
| global-budget | 274 | 9.5 | 18.8 | 88.7% | 2.58 | 513 | 53101 | 2.8 |

## TTFT p50 (ms) per tenant

| tenant | off | newest-idle | global-idle | global-budget |
|---|---|---|---|---|
| dispatch | 10.3 | 9.5 | 9.3 | 10.5 |
| moderation | 8.7 | 8.7 | 9.7 | 9.4 |
| operations | 9.8 | 9.7 | 10.1 | 10.0 |
| pipeline | 8.2 | 8.7 | 8.4 | 8.7 |
| research | 9.4 | 10.5 | 8.7 | 8.9 |
| route | 10.2 | 10.0 | 9.9 | 9.9 |
| support_email | 9.7 | 10.1 | 9.8 | 9.9 |

## Cache-hit ratio per tenant

| tenant | off | newest-idle | global-idle | global-budget |
|---|---|---|---|---|
| dispatch | 72.9% | 76.8% | 78.5% | 76.6% |
| moderation | 84.3% | 90.3% | 90.6% | 91.0% |
| operations | 86.2% | 91.0% | 91.4% | 91.4% |
| pipeline | 72.8% | 80.0% | 88.0% | 91.3% |
| research | 86.9% | 87.8% | 89.8% | 90.1% |
| route | 87.9% | 86.1% | 87.9% | 89.1% |
| support_email | 83.9% | 84.6% | 93.8% | 94.2% |

## TTFT p50 (ms) per callsite (top by volume)

| tenant:site | off | newest-idle | global-idle | global-budget |
|---|---|---|---|---|
| dispatch:__visit@dispatch_router.jac | 11.1 | 11.8 | 11.2 | 12.7 |
| dispatch:debrief | 8.6 | 8.6 | 8.4 | 8.4 |
| dispatch:resolve_auth | 11.4 | nan | nan | nan |
| dispatch:resolve_network | 8.4 | 9.5 | 7.9 | 12.1 |
| moderation:classify_content | 8.1 | 7.9 | 8.3 | 11.6 |
| moderation:escalate_case | 11.2 | 9.4 | 10.7 | 10.2 |
| moderation:log_decision | 8.3 | 8.6 | 9.7 | 8.6 |
| operations:analyze_metrics | 9.8 | 10.2 | 9.9 | 9.9 |
| operations:collect_metrics | 11.2 | 9.7 | 10.1 | 10.4 |
| operations:write_report | 10.3 | 8.0 | 10.6 | 8.4 |
| pipeline:classify_record | 7.7 | 8.5 | 7.7 | 7.8 |
| pipeline:extract_fields | 9.0 | 8.8 | 9.1 | 9.2 |
| pipeline:format_output | 9.0 | 9.7 | 8.4 | 8.8 |
| pipeline:summarize_record | 8.2 | 8.9 | 8.5 | 8.9 |
| pipeline:translate_summary | 8.5 | 8.6 | 9.0 | 8.7 |
| pipeline:validate_fields | 8.0 | 9.2 | 8.4 | 8.6 |
| research:audit_answer | 10.3 | 8.9 | 7.7 | 8.2 |
| research:draft_answer | 8.9 | 10.5 | 8.1 | 9.0 |
| research:investigate | 10.7 | 13.0 | 10.2 | 9.9 |
| route:__visit@route_wire.jac | 10.6 | 11.6 | 10.9 | 10.4 |
| route:resolve_billing | 10.2 | 10.0 | 9.4 | 9.6 |
| route:summarize | 8.3 | 8.6 | 9.9 | 10.7 |
| support_email:classify_email | 8.5 | 9.2 | 9.0 | 8.4 |
| support_email:compose_reply | 12.3 | 10.7 | 11.9 | 11.8 |
| support_email:polish_reply | 9.4 | 8.1 | 9.6 | 9.0 |

## TBT no-harm check (vs off, +10% tolerated)

- off: 2.24 ms (+0.0%) OK
- newest-idle: 2.24 ms (-0.2%) OK
- global-idle: 2.21 ms (-1.5%) OK
- global-budget: 2.58 ms (+15.3%) VIOLATION
