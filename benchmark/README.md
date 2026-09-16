# 多程序 agent 基准

四个用 Jac 写的结构化 agent，在一个 server 上按并发度扫描或混合运行，比较三种服务方式：
原生引擎（`lruraw`）、只做 prompt 重排（`lru`，论文里的 reorder-only）、完整机制（`ours`：重排 + 基于程序结构的 KV 规划）。

## 目录

```
benchmark/
  applications/           四个 agent 及其数据
    fact_check.jac        HoVer 多跳事实核查：ClaimAnalyzer -> Scout x N -> EvidenceAnalyzer，按轮循环
    coding_agent.jac      HumanEval 5 函数模块：Planner -> Coder -> Tester -> (Analyzer -> Coder)* -> Committer
    doc_analysis.jac      FinanceBench 文档分析：Extractor -> Analyst -> Auditor -> Writer，无环
    BFCL_agent.jac        BFCL v4 web search 风格 ReAct：单站点 Agent -> Executor 自环，工具为 Wikipedia 搜索/抓取
    bench_data/           HoVer/ HumanEval/ finicial_bench/ BFCL/，各自的数据文件和工具结果缓存
    prepare_financebench.py   下载 FinanceBench 的 PDF 并抽出逐页文本到 finicial_bench/pages/
    warm_web_cache.py         把缓存里失败的 Wikipedia 请求按 1 次/秒重取
  mixed_workload.py       驱动：多程序多车道跑 session，解析 server 日志，输出汇总
  sweep_common.py         并发扫描的公共实现；sweep_{coding,fact_check,finance,BFCL}.py 是各程序入口
  find_regime.bash        四个程序 lru/ours 的粗扫描链
  isolate_planner.bash    同一负载上四种 server 配置，隔离 planner 各部件的成本
  verify_clip.bash        验证降低 sglang 准入预留后的 regime 变化
  kill_bench.bash         停掉所有基准进程
  mixed_results/          全部输出；_regime_4096/ 是 max_tokens=4096 时期的旧结果
../start_server.bash      起 server 并等待就绪：./start_server.bash [ours|lru|lruraw|kvonly] [server 参数]
```

## 数据

| 程序 | 输入 | 数量 | 工具 |
|---|---|---|---|
| fact_check | `HoVer/hover_claims_120.tsv`（label, hops, claim） | 120 | Wikipedia 搜索，缓存在 `HoVer/wiki_cache/` |
| coding_agent | `HumanEval/tasks.txt`，5 题一组，5 个偏移 | 160 组，164 题全覆盖 | pytest |
| doc_analysis | `finicial_bench/financebench_open_source.jsonl` + `pages/` | 150 题，84 份 filing | 无 |
| BFCL_agent | `BFCL/BFCL_v4_web_search.json` + `_answers.json`（JSON Lines） | 100 题 | Wikipedia 搜索/抓取，缓存在 `BFCL/web_cache/` |

doc_analysis 的文档 = 证据页 ±4 页（`DA_PAGE_WINDOW`），上限 40000 字符（`DA_DOC_CHARS`），中位约 6.7k token；没有 pages/ 时退回数据集自带的整页证据文本。
工具结果按 md5 缓存，失败结果不写缓存。Wikipedia 在几十路并发下会返回 429，先用低并发跑一遍填缓存，再用 `warm_web_cache.py --program BFCL_agent|fact_check` 补失败条目。

## 运行

```bash
./start_server.bash ours                      # HOST 默认 16 GB；扫描脚本传 HOST=8
python -m benchmark.mixed_workload --tag ours --server-log benchmark/mixed_results/server_ours.log \
    --lanes fact_check=4,coding_agent=8,doc_analysis=4,BFCL_agent=4 --sessions 5 --warmup 1

python -m benchmark.sweep_coding ours --c 16,32,48,64 --sessions 2   # 单程序并发扫描，一个 arm 一个 server
./benchmark/find_regime.bash [all|bfcl|coding|finance|fact]         # 四程序 lru+ours 粗扫描
```

`--sessions N` 是每条车道连续跑的 session 数（0 = 跟随模式，直到固定车道结束）；每个 session 是一个 `jac run`，输入通过 FC_CLAIM / CA_TASKS / DA_TASK / BF_TASK 传入，session 按进程 pid 与 server 日志关联。
环境默认：FC_MIN_ROUNDS=2，三个工具延迟下限 2 s。

输出在 `mixed_results/<tag>/`：`logs/` 每 session 的 stdout，`sessions.jsonl`，`summary.json`；每程序一行追加到 `mixed_results/summary.csv`；扫描另写 `<prefix>_sweep_<arm>.csv`。
指标：JCT 和 TTFT 的 p50/p95/mean、device/host/miss 比例（token 加权）、预测命中率（直接后继 top-1）、准确率（Verdict 行对 expected）、固定窗口 600/900/1200 s 内完成数和满载窗口吞吐。多个 arm 比较吞吐必须用同一个固定窗口。

server 参数：`--lru` 关 planner，`--no-relayout` 关重排，`--no-promote` 只分带不预取，`--eviction lru|priority`，`--engine-log info` 打开 sglang 的批统计（`#running-req`、`#queue-req`、gen throughput），`--host` 主机 KV 层 GB。

## Regime

sglang 的 prefill 准入给每个 running 请求预留 `min(max_new_tokens − 已生成, SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION)`。agent 用 `max_tokens=4096` 时，31.8k token 的 device 池只能同时跑约 7 个请求，TTFT 变成纯排队，缓存命中进不了 TTFT。2026-09-15 起四个 agent 统一 `max_tokens=1024`（旧结果归档在 `_regime_4096/`）；`start_server.bash` 的 `CLIP` 变量可以只改预留估计。
比较 arm 时噪声带 3 到 15%（greedy 在不同批次组合下分叉，重试轮数不同）；同输入同轮数配对是最可靠的对比方式。

## 当前结果（max_tokens=1024，HOST=8，单次运行，均值）

| 程序 | c | JCT reorder-only → ours | TTFT reorder-only → ours | miss reorder-only / ours |
|---|---|---|---|---|
| BFCL | 20 / 40 / 60 | 169→171 / 388→382 / 655→583 s（+2 / −2 / −11%） | 10.2→9.9 / 27.7→27.7 / 48.5→44.4 s | 44/42 · 50/50 · 50/50 |
| coding | 16 / 32 / 48 / 64 | 128→130 / 196→192 / 366→349 / 506→490 s（+2 / −2 / −5 / −3%） | 0.3→0.3 / 2.7→2.5 / 10.8→9.7 / 18.5→17.8 s | 44/44 · 45/43 · 63/56 · 69/66 |
| finance | 8 / 16 / 32 | 96→95 / 229→203 / 464→428 s（−1 / −11 / −8%） | 9.3→9.1 / 29.6→26.2 / 69.4→64.2 s | 53/47 · 68/70 · 73/72 |
| fact | 8 / 14 / 20 | 136→128 / 206→208 / 277→245 s（−5 / +1 / −12%） | 1.5→1.3 / 5.8→5.5 / 11.4→9.6 s | 61/55 · 70/71 · 78/75 |

预测命中：fact 和 BFCL 1.0，finance 0.96 到 0.98，coding 0.86 到 0.88（Analyzer 分支由 pytest 结果决定）。
隔离实验（coding c=32）：priority 策略本身、只分带、完整 planner 相对 lru 的配对中位数分别为 +0.0%、+1.3%、−0.1%，机制没有可测成本。

## 已知问题

- BFCL 上约 43% 的引擎调用是 byllm 的 typed retry（Action JSON 解析失败后重发），准确率 5 到 7%；是模型和 Wikipedia 后端的问题，不影响服务对比但放大了调用数。
- fact_check 的 wiki_cache 是新建的，未覆盖的查询会实时请求 Wikipedia。
- 所有数字都是单次运行。
