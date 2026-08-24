# Cold-start results: `A-main`

config: {"trials": 20, "cells": ["off", "spec", "spec-idle"], "modes": ["single", "multi"], "tool_scale": 1.0, "model": "Qwen/Qwen2.5-3B-Instruct"}

client rows: 840, failed (rc≠0): 24


## single-task cold start


### single: per-cell totals

| cell | trials | serves | warm fraction (cached/prompt) | mean TBT ms | spec prefills/trial | spec tokens/trial | spec ms/trial |
|---|---|---|---|---|---|---|---|
| off | 20 | 660 | 52.9% | 9.13 | 0 | 0 | 0 |
| spec | 20 | 657 | 74.7% | 9.90 | 121 | 6352 | 2853 |
| spec-idle | 20 | 660 | 74.8% | 9.14 | 66 | 6166 | 1100 |

### single: spec vs off

trials=20; pairs used=132; dropped: {'different callsite sequence': 4, 'client failed': 4}

| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |
|---|---|---|---|---|---|---|
| dispatch | 20 | 741→645 | +140.7 [+112.2, +176.7] * | 126→129 | +3.0 [-5.1, +10.9] | 0.98x |
| moderation | 20 | 229→139 | +198.9 [+144.0, +255.5] * | 81→58 | +27.9 [+19.1, +36.5] * | 1.38x |
| operations | 12 | 766→382 | +387.0 [+371.0, +403.0] * | 165→137 | +30.8 [+16.8, +41.6] * | 1.20x |
| pipeline | 20 | 592→133 | +460.0 [+457.2, +462.4] * | 143→113 | +24.2 [+10.2, +37.4] * | 1.26x |
| research | 20 | 644→473 | +173.5 [+168.6, +178.4] * | 167→153 | +8.4 [+0.1, +16.1] * | 1.10x |
| route | 20 | 926→616 | +268.4 [+226.5, +302.4] * | 189→171 | +9.0 [-6.5, +19.6] | 1.11x |
| support_email | 20 | 796→188 | +608.0 [+608.0, +608.0] * | 131→75 | +56.2 [+51.1, +60.7] * | 1.75x |
| **all** | 132 | | +315.4 [+286.7, +345.3] * | 141→118 | +22.3 [+17.4, +27.0] * | 1.19x |

`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.

Speculation efficiency (single, spec): 127049 speculative tokens prefilled, 41634 uncached tokens removed from paired serves → 0.33 useful per spent (pairs only; unpaired trials' spend is counted, their savings are not).

### single: spec-idle vs off

trials=20; pairs used=136; dropped: {'client failed': 4}

| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |
|---|---|---|---|---|---|---|
| dispatch | 20 | 741→645 | +141.2 [+112.6, +177.8] * | 126→118 | +3.6 [-7.1, +13.1] | 1.07x |
| moderation | 20 | 229→139 | +200.6 [+144.6, +258.8] * | 81→48 | +32.0 [+25.6, +38.8] * | 1.68x |
| operations | 16 | 747→371 | +385.0 [+375.2, +395.8] * | 158→123 | +34.1 [+28.4, +39.2] * | 1.29x |
| pipeline | 20 | 592→133 | +457.4 [+451.9, +461.4] * | 143→106 | +30.5 [+19.4, +41.6] * | 1.35x |
| research | 20 | 644→472 | +174.2 [+169.4, +179.2] * | 167→153 | +7.1 [-1.2, +16.0] | 1.09x |
| route | 20 | 926→616 | +267.6 [+225.9, +301.9] * | 189→161 | +14.6 [-2.6, +27.3] | 1.17x |
| support_email | 20 | 796→188 | +608.0 [+608.0, +608.0] * | 131→76 | +55.0 [+50.9, +59.6] * | 1.72x |
| **all** | 136 | | +317.2 [+288.0, +346.8] * | 140→112 | +25.0 [+20.2, +29.5] * | 1.25x |

`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.

Speculation efficiency (single, spec-idle): 123324 speculative tokens prefilled, 43140 uncached tokens removed from paired serves → 0.35 useful per spent (pairs only; unpaired trials' spend is counted, their savings are not).

### single: first serve per callsite — median uncached tokens / TTFT ms

| tenant:site | prompt | off | spec | spec-idle |
|---|---|---|---|---|
| dispatch:__visit | 286 | 254 / 34 | 254 / 34 | 254 / 34 |
| dispatch:debrief | 116 | 68 / 19 | 6 / 17 | 6 / 13 |
| dispatch:resolve_auth | 141 | 93 / 18 | 45 / 16 | 45 / 15 |
| dispatch:resolve_database | 398 | 286 / 38 | 62 / 17 | 62 / 17 |
| dispatch:resolve_frontend | 144 | 96 / 18 | 48 / 15 | 48 / 21 |
| dispatch:resolve_network | 147 | 99 / 19 | 51 / 16 | 51 / 15 |
| moderation:archive_note | 129 | 81 / 40 | 33 / 16 | 33 / 16 |
| moderation:classify_content | 113 | 65 / 19 | 65 / 19 | 65 / 19 |
| moderation:draft_warning | 140 | 92 / 29 | 28 / 13 | 28 / 14 |
| moderation:escalate_case | 384 | 272 / 44 | 32 / 15 | 32 / 15 |
| moderation:log_decision | 127 | 79 / 18 | 35 / 15 | 35 / 15 |
| operations:analyze_metrics | 427 | 315 / 39 | 30 / 15 | 28 / 14 |
| operations:collect_metrics | 351 | 239 / 34 | 239 / 34 | 239 / 34 |
| operations:write_report | 152 | 112 / 19 | 33 / 15 | 33 / 16 |
| pipeline:classify_record | 135 | 87 / 25 | 7 / 13 | 7 / 13 |
| pipeline:extract_fields | 140 | 92 / 19 | 92 / 20 | 92 / 20 |
| pipeline:format_output | 193 | 144 / 24 | 10 / 18 | 10 / 15 |
| pipeline:summarize_record | 153 | 105 / 26 | 9 / 37 | 9 / 32 |
| pipeline:translate_summary | 123 | 75 / 18 | 11 / 13 | 11 / 13 |
| pipeline:validate_fields | 136 | 88 / 18 | 8 / 14 | 8 / 13 |
| research:audit_answer | 117 | 69 / 18 | 14 / 13 | 14 / 13 |
| research:draft_answer | 151 | 119 / 19 | 23 / 15 | 22 / 14 |
| research:investigate | 383 | 383 / 83 | 383 / 86 | 383 / 85 |
| route:__visit | 201 | 201 / 34 | 201 / 34 | 201 / 34 |
| route:resolve_billing | 380 | 268 / 37 | 44 / 17 | 44 / 16 |
| route:resolve_tech | 132 | 84 / 33 | 36 / 15 | 36 / 15 |
| route:summarize | 136 | 88 / 18 | 9 / 13 | 9 / 13 |
| support_email:classify_email | 118 | 70 / 20 | 70 / 20 | 70 / 20 |
| support_email:compose_reply | 702 | 590 / 72 | 46 / 20 | 46 / 24 |
| support_email:polish_reply | 125 | 77 / 18 | 13 / 13 | 13 / 13 |

## multi-task cold start


### multi: per-cell totals

| cell | trials | serves | warm fraction (cached/prompt) | mean TBT ms | spec prefills/trial | spec tokens/trial | spec ms/trial |
|---|---|---|---|---|---|---|---|
| off | 20 | 664 | 53.2% | 10.82 | 0 | 0 | 0 |
| spec | 20 | 659 | 63.3% | 11.87 | 85 | 3422 | 2572 |
| spec-idle | 20 | 663 | 57.1% | 11.02 | 10 | 984 | 229 |

### multi: spec vs off

trials=20; pairs used=131; dropped: {'client failed': 4, 'different callsite sequence': 5}

| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |
|---|---|---|---|---|---|---|
| dispatch | 20 | 767→700 | +64.0 [+50.3, +77.3] * | 183→192 | -33.1 [-63.8, -6.2] * | 0.95x |
| moderation | 20 | 262→204 | +102.9 [+57.4, +159.7] * | 112→126 | -13.3 [-30.1, +1.8] | 0.88x |
| operations | 13 | 846→493 | +347.7 [+280.6, +419.5] * | 220→200 | +17.8 [-1.7, +34.4] | 1.10x |
| pipeline | 20 | 601→376 | +236.9 [+211.9, +263.5] * | 216→224 | -4.8 [-23.5, +11.7] | 0.97x |
| research | 20 | 521→396 | +129.6 [+117.7, +141.7] * | 201→185 | +5.0 [-12.6, +21.3] | 1.09x |
| route | 18 | 910→747 | +132.1 [+92.2, +177.2] * | 247→256 | -18.8 [-41.7, -0.1] * | 0.97x |
| support_email | 20 | 796→746 | +85.6 [+48.0, +152.0] * | 177→182 | -15.7 [-36.1, +2.2] | 0.97x |
| **all** | 131 | | +147.2 [+125.9, +169.4] * | 193→195 | -10.3 [-18.6, -2.4] * | 0.99x |

`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.

Speculation efficiency (multi, spec): 68441 speculative tokens prefilled, 19277 uncached tokens removed from paired serves → 0.28 useful per spent (pairs only; unpaired trials' spend is counted, their savings are not).

### multi: spec-idle vs off

trials=20; pairs used=129; dropped: {'different callsite sequence': 7, 'client failed': 4}

| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |
|---|---|---|---|---|---|---|
| dispatch | 20 | 767→763 | +8.1 [-13.2, +27.9] | 183→204 | -36.6 [-60.6, -15.5] * | 0.90x |
| moderation | 20 | 262→240 | +67.8 [+18.1, +130.9] * | 112→126 | -6.4 [-15.4, +2.8] | 0.88x |
| operations | 13 | 846→670 | +192.0 [+98.9, +283.4] * | 220→226 | +0.5 [-16.7, +14.4] | 0.97x |
| pipeline | 20 | 601→593 | +60.5 [+29.4, +94.0] * | 216→222 | -4.2 [-21.0, +12.2] | 0.97x |
| research | 20 | 521→514 | +33.6 [+15.1, +55.4] * | 201→195 | -6.3 [-25.0, +8.9] | 1.03x |
| route | 16 | 914→892 | +44.1 [+10.9, +90.5] * | 250→264 | -23.7 [-52.2, -2.0] * | 0.95x |
| support_email | 20 | 796→796 | +37.6 [+0.0, +109.6] | 177→185 | -2.8 [-10.8, +6.4] | 0.95x |
| **all** | 129 | | +57.0 [+38.0, +78.5] * | 190→199 | -11.6 [-18.7, -5.2] * | 0.96x |

`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.

Speculation efficiency (multi, spec-idle): 19680 speculative tokens prefilled, 7354 uncached tokens removed from paired serves → 0.37 useful per spent (pairs only; unpaired trials' spend is counted, their savings are not).

### multi: first serve per callsite — median uncached tokens / TTFT ms

| tenant:site | prompt | off | spec | spec-idle |
|---|---|---|---|---|
| dispatch:__visit | 286 | 281 / 55 | 282 / 57 | 272 / 63 |
| dispatch:debrief | 116 | 68 / 33 | 26 / 30 | 65 / 31 |
| dispatch:resolve_auth | 141 | 93 / 33 | 45 / 31 | 93 / 33 |
| dispatch:resolve_database | 398 | 286 / 50 | 286 / 51 | 286 / 65 |
| dispatch:resolve_frontend | 144 | 96 / 32 | 96 / 33 | 96 / 38 |
| dispatch:resolve_network | 147 | 99 / 32 | 51 / 32 | 99 / 31 |
| moderation:archive_note | 129 | 81 / 29 | 81 / 33 | 81 / 32 |
| moderation:classify_content | 113 | 69 / 34 | 68 / 34 | 68 / 35 |
| moderation:draft_warning | 140 | 92 / 31 | 91 / 33 | 91 / 36 |
| moderation:escalate_case | 384 | 278 / 52 | 216 / 51 | 266 / 47 |
| moderation:log_decision | 127 | 79 / 35 | 47 / 31 | 74 / 34 |
| operations:analyze_metrics | 427 | 315 / 53 | 75 / 32 | 288 / 52 |
| operations:collect_metrics | 351 | 239 / 49 | 239 / 50 | 239 / 48 |
| operations:write_report | 158 | 106 / 33 | 33 / 27 | 33 / 32 |
| pipeline:classify_record | 135 | 87 / 33 | 54 / 33 | 87 / 33 |
| pipeline:extract_fields | 140 | 97 / 36 | 93 / 38 | 93 / 35 |
| pipeline:format_output | 193 | 145 / 39 | 65 / 30 | 145 / 40 |
| pipeline:summarize_record | 153 | 105 / 35 | 58 / 32 | 105 / 34 |
| pipeline:translate_summary | 123 | 75 / 33 | 59 / 32 | 75 / 33 |
| pipeline:validate_fields | 136 | 88 / 33 | 55 / 35 | 87 / 34 |
| research:audit_answer | 110 | 62 / 33 | 15 / 26 | 62 / 32 |
| research:draft_answer | 150 | 100 / 32 | 50 / 31 | 98 / 34 |
| research:investigate | 383 | 271 / 57 | 271 / 58 | 271 / 56 |
| route:__visit | 201 | 174 / 43 | 172 / 46 | 187 / 43 |
| route:resolve_billing | 380 | 268 / 53 | 235 / 51 | 267 / 51 |
| route:resolve_tech | 138 | 90 / 31 | 90 / 37 | 85 / 42 |
| route:summarize | 136 | 88 / 32 | 12 / 30 | 86 / 30 |
| support_email:classify_email | 118 | 70 / 33 | 70 / 34 | 70 / 32 |
| support_email:compose_reply | 702 | 590 / 77 | 588 / 80 | 590 / 80 |
| support_email:polish_reply | 125 | 77 / 32 | 29 / 29 | 77 / 32 |
