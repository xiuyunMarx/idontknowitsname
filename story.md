


# Design 
Based on our observations, we build the a compile-runtime co-design system that substantially exploit compile-time colected information and guide the model which mitigate the cold start TTFT.

# Evaluation

All experiments run on one NVIDIA RTX 3090 (24 GB) with vLLM 0.19.1 serving Qwen2.5-7B-Instruct in bf16, automatic prefix caching and priority scheduling on. The interference profile taken on this configuration admits 48 uncached speculative tokens per engine step at every decode concurrency from 1 to 16 (the +10% mean-TBT bound; 64 tokens raise TBT by 21%).

# Table of Content
1. Single Application
2. Slack from full to None -> How system regress
3. Abalation. The effect of each part: {Invariant prompt + tool prefilling + argument prefilling}


Three byLLM programs are the workloads, each with a 30-case dataset (`benchmark/evaluation/datasets`):

- **HoVer**: many-hop claim verification with two hops; 11 byLLM calls per workflow (`decompose_claim`, then per hop `plan_query → retrieve_evidence` (tool loop) `→ reason_hop → assess_gap`, then `verify_claim` (tool loop) `→ synthesize_answer`). Every hand-off after the first crosses a plain `visit`.
- **deep_search**: an LLM-routed supervisor (`visit ... by llm(select=(1,3))`) dispatches one to three research sub-agents per round for two rounds, each running a tool loop over an offline source pack; 15 calls per workflow when three agents run per round.
- **intercode_sql**: text-to-SQL with an execution-feedback repair loop over SQLite (Spider dev databases); one repair turn is forced so every workflow serves `generate_sql → diagnose → revise_sql → finalize`.

Tool bodies sleep 200–350 ms and consecutive calls are separated by 150–200 ms. `off` is the baseline: the same server with speculation disabled. `spec` is the full system. Every trial boots a fresh server. The primary metric is uncached prompt tokens at serve (prompt tokens minus prefix-cache hits), reported next to TTFT summed over a workflow's calls (ΣTTFT). Differences are paired per workflow and reported with a bootstrap 95% confidence interval.

# Single Application Cold Start

One program at a time on an otherwise idle server, 3 trials with cases 0–2 of each dataset, 8 generation slots.

| tenant | pairs (same callsite sequence) | uncached tokens / workflow, off → spec | Δuncached | ΣTTFT ms, off → spec | ΔΣTTFT, 95% CI | ratio |
|---|---|---|---|---|---|---|
| HoVer | 3 (3) | 2456 → 1428 | +1028 | 760 → 552 | +207 | 1.38× |
| deep_search | 3 (2) | 3929 → 2367 | +1562 | 1302 → 985 | +318 | 1.32× |
| intercode_sql | 3 (3) | 2176 → 1744 | +432 | 510 → 431 | +80 | 1.19× |
| **all** | 9 | 2854 → 1846 | **+1007** | 857 → 656 | **+201.5 [+134.5, +266.8]** | **1.31×** |

*Table E3. Single application cold start, `spec` vs `off`. Δ is off − spec; positive favours speculation.*

The warm fraction of served prompts rises from 38.9% to 60.5%; mean TBT moves from 19.52 ms to 19.74 ms. Downstream calls arrive warm: HoVer's `verify_claim` drops from 463 to 144 uncached tokens (121 → 42 ms), `assess_gap` from 196 to 85, `plan_query` from 167 to 86; deep_search's four sub-agent calls from about 275 to 70 (80 → 40 ms) and `summarize` from 282 to 131; intercode's `diagnose` from 265 to 46 (77 → 24 ms). Entry calls are unchanged (`generate_sql` 922 uncached tokens at 208 ms, `decompose_claim` 142, the router prompt 268–295) and account for most of what remains. Two predicted calls stay cold: deep_search's second-round `decompose_task` (224 uncached tokens) binds four findings only when the last sub-agent returns, and HoVer's `synthesize_answer` (278) follows `verify_claim` with no window in between.

# Multi-tenant Cold Start

We set the multi-tenant experiment in FaaS serving, where workflow instances are isolated: privacy and protocol regulations rule out prefix-cache sharing between instances, so every instance starts cold and its KV cache dies with it. The server implements this with a per-instance vLLM `cache_salt` (`--cache-isolation instance`): an instance's real requests, speculative prefills and route probes carry its salt, and no KV block is ever shared across instances. Both `off` and `spec` run under the same isolation. The shared-cache regime, in which instances of the same program reuse each other's prefixes, is reported second.

**Blend.** A blend (`benchmark/evaluation/synthesis_data.py`) is a schedule: 12 workflows per trial, four per program, cases drawn from a seeded per-tenant permutation so the 36 workflows of a 3-trial blend use 12 distinct cases per program; the tenants are interleaved at random. Three arrival traces share the same case order and differ only in start times: `poisson` (a Poisson process of 0.5 workflows/s, all 12 arriving within about 30 s), `uniform` (evenly over 3 s) and `burst` (all within 1 s). Each trial boots a fresh server with 16 generation slots; `off` and `spec` execute the same trace and pair per workflow. Mean concurrency is the average number of workflows in flight over the trial.

| trace | mean / peak concurrency | warm fraction, off → spec | Δuncached / workflow | ΣTTFT ms, off → spec | ΔΣTTFT, 95% CI | ratio | call TTFT p95 ms, off → spec | TBT ms, off → spec | spec prefills (killed) / trial |
|---|---|---|---|---|---|---|---|---|---|
| poisson 0.5/s | 3.6 / 8 | 37.4% → 45.5% | **+360** | 1294 → 1185 | **+108.4 [+40.5, +181.8]** | **1.09×** | 232 → 226 | 23.7 → 24.6 | 394 (70) |
| uniform 3 s | 6.7 / 12 | 37.8% → 38.7% | +44 | 1619 → 1546 | +72.8 [−15.4, +162.5] | 1.05× | 307 → 289 | 31.3 → 32.2 | 129 (45) |
| burst 1 s | 6.7 / 12 | 37.6% → 39.0% | +54 | 1836 → 1707 | +128.3 [+34.8, +228.5] | 1.08× | 457 → 355 | 32.8 → 33.6 | 114 (36) |

*Table E4. Multi-tenant cold start with per-instance cache isolation, `spec` vs `off`, 36 workflow pairs per trace pooled over tenants; 33 (poisson), 32 (uniform) and 36 (burst) of them served identical callsite sequences.*

Under the Poisson trace speculation removes 360 uncached tokens per workflow and 108 ms of ΣTTFT; per-call TTFT p50 falls from 130 to 107 ms. Per tenant: HoVer 2577 → 2178 tokens, 1.22× [+119, +408]; deep_search 3921 → 3349, 1.03× [−51, +168]; intercode_sql 1873 → 1764, 1.02× [−28, +45]. Under the two saturating traces the engine is rarely without a real prefill in flight: speculation issues 114–129 prefills per trial instead of 394, 30–40% of them killed by a real admission, and the warm fraction moves by about one point. ΣTTFT still improves in the burst trace, in the tail rather than in prefill work (per-call p95 457 → 355 ms, p99 607 → 447 ms); the uniform trace is unchanged within its interval. TBT rises by 0.8–0.9 ms per step in every trace.

| trace | mean / peak concurrency | warm fraction, off → spec | Δuncached / workflow | ΣTTFT ms, off → spec | ΔΣTTFT, 95% CI | ratio | call TTFT p95 ms, off → spec | TBT ms, off → spec | spec prefills (killed) / trial |
|---|---|---|---|---|---|---|---|---|---|
| poisson 0.5/s | 3.5 / 8 | 58.2% → 62.6% | +203 | 944 → 910 | +34.7 [−0.5, +70.6] | 1.04× | 170 → 164 | 21.9 → 22.0 | 422 (82) |
| uniform 3 s | 6.2 / 12 | 58.2% → 58.5% | +31 | 1208 → 1217 | −9.1 [−64.3, +42.2] | 0.99× | 242 → 243 | 26.8 → 27.1 | 121 (43) |
| burst 1 s | 5.7 / 12 | 60.4% → 58.6% | +18 | 1261 → 1295 | −34.3 [−101.6, +35.3] | 0.97× | 259 → 259 | 27.3 → 28.2 | 114 (39) |

*Table E5. The same three traces with a shared prefix cache. In the burst trace one `off` deep_search client failed typed-output validation, leaving 35 pairs.*

With a shared cache the baseline already serves 58–60% of prompt tokens from the cache (38.9% for a single instance): four instances of each program start within seconds of each other and the later ones hit the invariant prefixes the earlier ones computed. Speculation adds at most four points of warm fraction, 203 tokens per workflow in the Poisson trace (+35 ms, interval touching zero) and nothing measurable in the saturating traces.

<!-- # Evaluation

## Microbenchmark: history-free workflow cold start

This microbenchmark measures what compiler-guided proactive prefill recovers in the regime the introduction defines: the model is already serving, the KV cache is empty, and no execution of the workflow has ever been observed.

**Setup.** One NVIDIA RTX 3090 (24 GB) runs vLLM 0.19.1 with automatic prefix caching and priority scheduling, serving Qwen2.5-3B-Instruct in FP16 with 8 generation slots. Seven byLLM programs are registered as tenants, each with its own connection and its own compiled topology: a tool-using research agent (investigate → answer → audit), an operations agent with two ReAct tool loops, a 3-way LLM-routed ticket handler (`visit ... by llm()`), a support-email pipeline whose reply step carries a 700-token style guide, a six-stage data pipeline, a three-branch moderation fan-out, and a 5-way incident dispatcher. Tool bodies sleep 200–350 ms and consecutive calls are separated by 100–200 ms, standing in for tool execution and interpreter work. Each program has five inputs, cycled across trials.

**Conditions.** `off` is the baseline: the same server, prefix caching on, speculation disabled. `spec` is the full system: may-run-next topology, `const`/`ret` provenance, tool-turn warming and the route probe, with speculative prefill admitted during decode under the profiled per-step token allowance (16 uncached tokens per step at one concurrent decode, 48 at two to nine, 32 at ten to sixteen, calibrated to a +10% mean time-between-tokens bound). `spec-idle` uses the same predictions but prefills only while the engine has no request in flight.

**Protocol.** Every trial boots a fresh server process, so the KV cache and all runtime state start empty; each tenant then runs exactly one workflow and the server is stopped. In *single* mode the seven workflows run strictly one after another; in *multi* mode all seven start within a 3 s window. A trial uses the same inputs and the same start schedule in every condition, and condition order is rotated across trials. We ran 20 trials (840 workflow executions). A (trial, tenant) pair enters a comparison only when both executions exited cleanly and served the same callsite sequence; 24 executions failed (the operations agent exceeds its ReAct iteration limit on one of its five inputs, in every condition) and 4–7 pairs per comparison served different sequences. Both are excluded.

**Metrics.** The primary metric is *uncached prompt tokens at serve*, the part of each request's prompt that vLLM had to prefill (prompt tokens minus prefix-cache hits); it is deterministic and independent of queueing. We also report TTFT, measured from engine admission to first token, summed over the calls of one workflow (ΣTTFT), mean time-between-tokens (TBT), and the number of speculative tokens issued. Differences are paired per (trial, tenant) and reported with a bootstrap 95% confidence interval.

### Single-task cold start

Table E1 compares `spec-idle` with `off` in single mode (136 pairs).

| tenant | n | uncached tokens / workflow, off → spec-idle | Δ uncached, 95% CI | ΣTTFT ms, off → spec-idle | Δ ΣTTFT, 95% CI | ratio |
|---|---|---|---|---|---|---|
| dispatch | 20 | 741 → 645 | +141 [+113, +178] | 126 → 118 | +3.6 [−7.1, +13.1] | 1.07× |
| moderation | 20 | 229 → 139 | +201 [+145, +259] | 81 → 48 | +32.0 [+25.6, +38.8] | 1.68× |
| operations | 16 | 747 → 371 | +385 [+375, +396] | 158 → 123 | +34.1 [+28.4, +39.2] | 1.29× |
| pipeline | 20 | 592 → 133 | +457 [+452, +461] | 143 → 106 | +30.5 [+19.4, +41.6] | 1.35× |
| research | 20 | 644 → 472 | +174 [+169, +179] | 167 → 153 | +7.1 [−1.2, +16.0] | 1.09× |
| route | 20 | 926 → 616 | +268 [+226, +302] | 189 → 161 | +14.6 [−2.6, +27.3] | 1.17× |
| support_email | 20 | 796 → 188 | +608 [+608, +608] | 131 → 76 | +55.0 [+50.9, +59.6] | 1.72× |
| **all** | 136 | | **+317 [+288, +347]** | **140 → 112** | **+25.0 [+20.2, +29.5]** | **1.25×** |

*Table E1. Single-task cold start, `spec-idle` vs `off`. Δ is off − spec-idle; positive favours speculation.*

Speculation removes 317 uncached tokens per workflow (44% of the baseline's 720) and cuts ΣTTFT from 140 ms to 112 ms. The token reduction is significant for every tenant; the TTFT reduction is significant for five of seven. The warm fraction of served prompts rises from 52.9% to 74.8%.

The per-callsite view separates the two kinds of call. Downstream calls, whose predecessors ran before them, arrive almost fully warm: `compose_reply` drops from 590 to 46 uncached tokens (TTFT 72 → 24 ms), `analyze_metrics` from 315 to 28 (39 → 14 ms), `escalate_case` from 272 to 32 (44 → 15 ms), `format_output` from 144 to 10 (24 → 15 ms). Entry calls are unchanged: `investigate` (383 uncached), `collect_metrics` (239), the two routing prompts (254 and 201) and the classifiers that open the email, moderation and pipeline programs (65–92) have no preceding byLLM call, and their bytes are the user's input. After speculation, entry calls account for most of the uncached tokens that remain.

`spec` (speculation also during decode) saves the same tokens (+315 per workflow) but yields 1.19× on ΣTTFT instead of 1.25× and raises mean TBT from 9.13 ms to 9.90 ms: in single-task execution the engine is idle during every tool window, so speculating during decode adds interference without adding coverage.

### Multi-tenant cold start

Table E2 repeats the comparison in multi mode, where the seven workflows cold-start concurrently (131 and 129 pairs).

| condition | warm fraction | mean TBT ms | spec prefills / trial | Δ uncached / workflow, 95% CI | ΣTTFT ms, off → cond | Δ ΣTTFT, 95% CI | ratio |
|---|---|---|---|---|---|---|---|
| off | 53.2% | 10.82 | 0 | | 193 | | |
| spec | 63.3% | 11.87 | 85 | +147 [+126, +169] | 193 → 195 | −10.3 [−18.6, −2.4] | 0.99× |
| spec-idle | 57.1% | 11.02 | 10 | +57 [+38, +79] | 190 → 199 | −11.6 [−18.7, −5.2] | 0.96× |

*Table E2. Multi-tenant cold start vs `off`, pooled over tenants.*

Under concurrency the engine is rarely empty, so `spec-idle` issues 10 speculative prefills per trial instead of 66 and recovers 57 tokens per workflow. Budgeted admission (`spec`) keeps speculating inside decode steps and recovers 147 tokens per workflow, 46% of the single-task saving, at +10% TBT. Neither condition reduces ΣTTFT; both are 10 ms worse per workflow, and no tenant improves significantly. Two measured quantities account for this. The interference profile puts the prefill cost of one uncached token on this model and GPU at about 0.04 ms, so 147 tokens spread over a workflow's five to seven calls are worth roughly 1 ms per call; and with seven workflows in flight every request first waits for the running engine step, which lifts every callsite's TTFT to a 30 ms floor (13–19 ms in single mode) and adds about 2.8 ms to a real request that lands behind a speculative step.

### Cost of speculation

In single mode `spec-idle` prefills 6.2k speculative tokens per trial for seven workflows, 1.1 s of engine time spent inside tool windows that would otherwise be idle. Against the 3.0k uncached tokens it removes from paired serves, 0.35 of each speculative token is used. The unused remainder comes from branches that are warmed but not taken (every `resolve_*` handler of the dispatcher, both branches of the router, all three moderation outcomes) and from prefixes re-issued after a `ret` binding lands; the accounting counts the re-issued invariant part as spent although it is already resident, so the true waste is lower than this figure. -->
