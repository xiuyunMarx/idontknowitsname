# Cold-start results: `B-multi-fixed`

config: {"trials": 20, "cells": ["off", "spec", "spec-idle"], "modes": ["multi"], "tool_scale": 1.0, "model": "Qwen/Qwen2.5-3B-Instruct"}

client rows: 168, failed (rc≠0): 3


## multi-task cold start


### multi: per-cell totals

| cell | trials | serves | warm fraction (cached/prompt) | mean TBT ms | spec prefills/trial | spec tokens/trial | spec ms/trial |
|---|---|---|---|---|---|---|---|
| off | 8 | 267 | 52.5% | 11.08 | 0 | 0 | 0 |
| spec | 8 | 262 | 61.5% | 11.54 | 91 | 3504 | 2331 |
| spec-idle | 9 | 304 | 57.1% | 10.71 | 10 | 928 | 148 |

### multi: spec vs off

trials=8; pairs used=51; dropped: {'different callsite sequence': 4, 'client failed': 1}

| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |
|---|---|---|---|---|---|---|
| dispatch | 8 | 770→754 | +52.0 [+26.4, +77.6] * | 172→189 | -9.6 [-33.3, +7.0] | 0.91x |
| moderation | 8 | 366→246 | +82.0 [+24.0, +170.0] * | 118→132 | -11.6 [-36.8, +9.3] | 0.90x |
| operations | 5 | 766→446 | +345.0 [+329.6, +362.8] * | 211→204 | +17.7 [-13.6, +45.8] | 1.03x |
| pipeline | 8 | 600→459 | +171.0 [+125.8, +216.4] * | 205→195 | +8.6 [-3.7, +21.9] | 1.05x |
| research | 8 | 532→390 | +133.9 [+113.8, +156.0] * | 183→191 | -7.8 [-25.0, +13.2] | 0.96x |
| route | 6 | 758→622 | +176.0 [+72.0, +274.7] * | 200→215 | +1.7 [-36.1, +32.6] | 0.93x |
| support_email | 8 | 796→751 | +62.0 [+30.0, +108.0] * | 174→181 | -13.3 [-29.2, +2.7] | 0.96x |
| **all** | 51 | | +133.1 [+103.1, +165.0] * | 178→187 | -3.4 [-12.0, +5.2] | 0.95x |

`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.

Speculation efficiency (multi, spec): 28034 speculative tokens prefilled, 6788 uncached tokens removed from paired serves → 0.24 useful per spent (pairs only; unpaired trials' spend is counted, their savings are not).

### multi: spec-idle vs off

trials=9; pairs used=51; dropped: {'different callsite sequence': 4, 'client failed': 1, 'missing': 7}

| tenant | n | uncached tok/workflow base→cell | Δuncached (base−cell) 95% CI | ΣTTFT ms base→cell | ΔΣTTFT 95% CI | ΣTTFT ratio |
|---|---|---|---|---|---|---|
| dispatch | 8 | 770→770 | +7.8 [+0.0, +23.2] | 172→176 | -8.8 [-23.6, +5.7] | 0.98x |
| moderation | 8 | 366→246 | +52.1 [+0.1, +144.1] * | 118→141 | -17.7 [-39.7, +3.7] | 0.84x |
| operations | 5 | 766→734 | +104.6 [+12.8, +236.2] * | 211→207 | +10.8 [-13.3, +34.9] | 1.02x |
| pipeline | 8 | 600→584 | +67.6 [+19.0, +134.5] * | 205→205 | -0.5 [-19.2, +14.8] | 1.00x |
| research | 8 | 532→520 | +23.8 [+1.8, +52.0] * | 183→189 | +0.1 [-14.8, +15.7] | 0.97x |
| route | 6 | 886→757 | +72.7 [+10.7, +174.0] * | 248→229 | -0.2 [-27.7, +20.9] | 1.09x |
| support_email | 8 | 796→794 | +4.0 [+0.0, +10.0] | 174→175 | -4.1 [-15.1, +8.9] | 0.99x |
| **all** | 51 | | +43.2 [+20.8, +69.0] * | 181→193 | -3.8 [-11.3, +3.2] | 0.94x |

`*` = 95% CI excludes 0. Δ is base − cell, so positive favours the cell.

Speculation efficiency (multi, spec-idle): 8348 speculative tokens prefilled, 2201 uncached tokens removed from paired serves → 0.26 useful per spent (pairs only; unpaired trials' spend is counted, their savings are not).

### multi: first serve per callsite — median uncached tokens / TTFT ms

| tenant:site | prompt | off | spec | spec-idle |
|---|---|---|---|---|
| dispatch:__visit | 286 | 272 / 49 | 272 / 50 | 262 / 45 |
| dispatch:debrief | 116 | 68 / 33 | 44 / 31 | 68 / 34 |
| dispatch:resolve_auth | 141 | 93 / 30 | 45 / 30 | 93 / 37 |
| dispatch:resolve_database | 398 | 286 / 50 | 286 / 53 | 286 / 62 |
| dispatch:resolve_frontend | 144 | 96 / 35 | 96 / 18 | 96 / 40 |
| dispatch:resolve_network | 147 | 99 / 45 | 83 / 38 | 99 / 33 |
| moderation:archive_note | 129 | 81 / 51 | 81 / 36 | 81 / 30 |
| moderation:classify_content | 113 | 68 / 32 | 68 / 34 | 68 / 33 |
| moderation:draft_warning | 142 | 94 / 35 | 94 / 38 | 94 / 55 |
| moderation:escalate_case | 384 | 278 / 51 | 272 / 53 | 272 / 52 |
| moderation:log_decision | 127 | 95 / 34 | 74 / 33 | 79 / 39 |
| operations:analyze_metrics | 427 | 315 / 53 | 75 / 34 | 280 / 52 |
| operations:collect_metrics | 351 | 239 / 48 | 239 / 42 | 239 / 50 |
| operations:write_report | 145 | 97 / 31 | 33 / 16 | 97 / 27 |
| pipeline:classify_record | 135 | 88 / 31 | 56 / 35 | 86 / 32 |
| pipeline:extract_fields | 140 | 94 / 31 | 94 / 35 | 97 / 35 |
| pipeline:format_output | 194 | 144 / 40 | 71 / 30 | 140 / 40 |
| pipeline:summarize_record | 153 | 106 / 34 | 104 / 32 | 105 / 33 |
| pipeline:translate_summary | 122 | 74 / 31 | 56 / 31 | 73 / 30 |
| pipeline:validate_fields | 136 | 88 / 31 | 56 / 36 | 87 / 31 |
| research:audit_answer | 117 | 66 / 32 | 12 / 30 | 69 / 32 |
| research:draft_answer | 150 | 102 / 30 | 50 / 30 | 98 / 34 |
| research:investigate | 383 | 271 / 57 | 271 / 55 | 271 / 54 |
| route:__visit | 201 | 187 / 44 | 187 / 47 | 200 / 45 |
| route:resolve_billing | 380 | 268 / 52 | 44 / 29 | 267 / 51 |
| route:resolve_tech | 136 | 90 / 27 | 85 / 39 | 63 / 48 |
| route:summarize | 133 | 83 / 31 | 56 / 25 | 56 / 29 |
| support_email:classify_email | 118 | 70 / 33 | 70 / 35 | 70 / 34 |
| support_email:compose_reply | 702 | 590 / 77 | 589 / 80 | 588 / 77 |
| support_email:polish_reply | 125 | 77 / 31 | 29 / 31 | 77 / 33 |
