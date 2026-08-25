# 评测数据集

三个用例包，每个 30 条用例，分别对应 `../programs/` 下的一个租户程序。每个包都由
`build/` 里的脚本从公开数据源生成，可以随时重建，而不是手工写死的。租户通过一个索引
环境变量选择用例，这样混合负载实验就能像 `microbench/experiments` 轮换输入那样轮换用例。

| 数据包 | 租户 | 用例数 | 选择方式 | 体积 |
|---|---|---:|---|---:|
| `hover_claims_30.json` + `hover_wiki.json` | `HoVer.jac` | 30 条断言 | `HOVER_CLAIM_INDEX` | 12 KB + 250 KB |
| `intercode_sql_30.json` | `intercode_sql.jac` | 30 个问题 | `SQL_TASK_INDEX` | 328 KB |
| `deep_search_30.json` | `deep_search.jac` | 30 个任务 | `DEEP_TASK_INDEX` | 141 KB |

`hover_mini_*.json`、`intercode_sql_tasks.json` 和 `deep_search_sources.json` 是开发程序
时手写的小样例，保留作冒烟测试；用 `$HOVER_CORPUS` / `$INTERCODE_TASKS` / `$DEEP_SOURCES`
指过去即可。

用例选择带固定随机种子（每个构建脚本里的 `SEED = 20260824`），因此在相同输入下重建会得到
同样的 30 条用例。哪些用例一起跑、按什么比例跑，属于后续混合负载实验的问题；每个包都为此
带上了逐用例的元数据（跳数、标签、难度、主题簇、rubric）。

---

## 1. HoVer —— 多跳断言验证

**数据来源。** 断言取自 HoVer dev 发布版（`hover-nlp/hover` 仓库的
`data/hover/hover_dev_release_v1.1.json`，共 4000 条带标注的断言）。文章取自
`BeIR/hotpotqa`，即 2017 年 10 月维基百科快照的 5,233,329 段导语段落——HoVer 标注时用的
就是这一版语料，所以随包发布的文本与金标签是同一时期的。

**构建。**

```bash
python build/build_hover.py            # 会向 --cache（默认 /tmp/hover_build）下载约 976 MB
```

先按（跳数 × 标签）分桶轮流抽取 300 条候选断言，再对语料做两遍扫描（先只读标题，再只取
需要的标题的正文），最后保留证据链完整、且每条金标注句子都还在截断文本之内的 30 条断言。
扫描结果会缓存在 `--cache` 里，之后再调整选择规则或链接推导都不必重新扫描。

**内容。** 30 条断言——2 跳、3 跳、4 跳各 10 条；SUPPORTED 与 NOT_SUPPORTED 各 15 条——
以及 675 篇文章：90 篇金标注文章，加上按标题词重合度为每条断言挑出的 20 篇干扰文章。文章
正文在 1500 字符附近按句子边界截断（平均 291 字符）。每条断言带有 `uid`、`label`、
`num_hops`、`supporting_titles` 以及对应的 HotpotQA id。

**链接。** BEIR 语料不含链接结构，因此 `list_links` 由"提及"推导：若某篇文章的标题，或去掉
消歧括号后的标题（`Life Goes On (Fergie song)` → `Life Goes On`），以词边界形式出现在另一
篇文章的正文中，就记为一条链接。675 篇中有 167 篇带链接。

**构建后的校验**（全部通过租户自己的工具执行）：

- 30 条断言中有 27 条，只用断言原文检索就能命中至少一篇金标注文章；
- 90 篇金标注文章中有 89 篇，用自身标题检索即可找到；
- 90 篇金标注文章中有 46 篇，被同一条断言的另一篇金标注文章提及。

也就是说，检索本身是真实的搜索问题——675 个候选、每条断言约 22 篇词面相近的干扰项——但每条
断言的证据链都是可达的。

## 2. InterCode-SQL —— 执行反馈修复循环

**数据来源。** `princeton-nlp/intercode` 仓库的 `data/sql/spider/ic_spider_dev.json`
（InterCode-SQL 使用的 1034 个 Spider 1.0 dev 问题）与 `ic_spider_dbs.sql`（其 19 个 dev
数据库的 MySQL dump）。

**构建。**

```bash
python build/build_intercode_sql.py    # 下载约 1.5 MB，秒级完成
```

InterCode 把这些数据库跑在 MySQL 上，而租户用的是内存中的 SQLite，因此 dump 需要转换一次：
丢弃 `/*! */` 指令、`LOCK TABLES` 以及用户和建库语句；反引号改为双引号（Spider 里有
`18_49_Rating_Share`、`Official_ratings_(millions)` 这样的列名）；去掉 `AUTO_INCREMENT`、
`ENGINE=`、字符集与排序规则子句；去掉 MySQL 的索引前缀长度
（`PRIMARY KEY ("Year_awarded"(255))`）；删除 `CREATE TABLE` 内部的 `KEY` 索引行；把 `\'`
换成 `''`。外键约束予以保留。

**筛选。** 1034 条金标准查询全部在转换后的数据库上执行：888 条通过；91 条因为在这份数据上
返回空结果被剔除；55 条因为金标准答案是单个 `0` 或 `1` 被剔除——这种结果执行匹配无法与瞎猜
区分开。20 个数据库全部能正常建库，这本身就是转换忠实性的证据。

**内容。** 30 个问题——easy 8、medium 8、hard 7、extra 7，同一数据库最多 3 个——覆盖它们所需
的 16 个数据库（136 条建库语句）。每个任务带有 `id`（在 InterCode dev 文件中的序号）、`db`、
`question`、`gold` 和 `hardness`。

**构建后的校验**：30 条金标准查询经租户自己的 `grade()` 全部返回 `MATCH`，错误查询返回
`MISMATCH`，语法错误的查询返回 `ERROR`。

## 3. Deep Research —— 主管路由与扇出

目前没有公开的 deep research 基准能为四个子智能体同时提供离线资料，因此这份数据包是围绕
LLM 服务与机器学习系统方向的 12 个主题簇（前缀缓存、预填充/解码分离、批处理调度、投机解码、
量化、长上下文、MoE、RAG、智能体、编译器—运行时协同设计、多租户、评测）用真实文献拼装的。

**数据来源。** 论文来自 arXiv API，每个主题簇一条相关性查询，保留 id、标题、作者、年份和摘要。
背景条目来自维基百科 REST summary 接口。实测数列来自本仓库自己的双租户实验（见顶层
`README.md`）。合成数列由构建种子生成，每簇一条，文本里明确标注为
*Synthetic benchmark series (seeded, not measured)*，这样数据分析子智能体能看出手里拿的是
什么。事实核查条目逐句摘自上述摘要与导语，末尾附来源 id。

**构建。**

```bash
python build/build_deep_search.py      # 约 150 次 API 调用，带限速与响应缓存
```

**内容。** 12 个主题簇上的 30 个任务，来自六个模板（`compare`、`evidence`、`provenance`、
`tradeoff`、`survey`、`applicability`），每个模板 5 个；70 篇论文、29 个背景条目、17 条数列
（5 条实测、12 条合成）、99 条事实核查条目。

**Rubric。** 每个任务都列出一份正确报告应当引用的证据：`{papers, pages, metrics, sources}`。
这份清单不是随意挑的，而是租户自己的 `paper_search` / `web_search` 在该任务的 `probe` 查询
（主题簇名加上它对比的两种机制）下返回、且属于该主题簇的条目。因此报告可以直接用 rubric id
做字符串匹配来打分，不需要 LLM 评委，而且 rubric 里不会出现检索不到的东西。

**构建后的校验**：30 份 rubric 全部能在包内解析，且其中每篇论文、每个背景条目都能被租户在该
任务 probe 下检索到。30 个任务都至少要求一篇论文（22 个要求两篇），19 个要求背景条目，15 个
要求数列，20 个要求事实核查条目。30 个任务中有 15 个仅凭任务原文就能检索到 rubric 论文，其余
需要规划器写出的子问题——那正是这个租户要做的工作。

---

## 多租户冷启动的混合采样

`../synthesis_data.py` 从这三个包中采样出一份 blend——它是一张调度表，而不是数据副本，因为
每个租户会自己加载数据包、并按索引选取用例：

```bash
python benchmark/evaluation/synthesis_data.py --blend-ratio 0.5,0.3,0.2 \
    --size 12 --trials 5 --mode multi --window 3
```

`--blend-ratio` 按 `--tenants` 的顺序解读，用最大余数法把一次 trial 的 `--size` 个工作流分配
给各租户。用例取自每个租户带固定种子的排列，并在多个 trial 之间连续消费，因此只有在整个数据包
用完之后才会重复（5 个 trial × 每次 6 个 hover 工作流，恰好把 30 条用例各跑一遍）。输出的
JSON 包含：

- `meta`——种子、比例、每个 trial 的分配数量，以及索引所指向的每个数据包的文件名与 SHA-1，
  让一份 blend 与它所基于的数据版本绑定；
- `server.programs`——传给 `start_server.py` 的 `--program name:path:port` 参数，且只列出
  这份 blend 真正用到的租户；
- `trials[].entries[]`——每个工作流一条：`tenant`、`jac_file`、`port`、`case_index`、
  `case_id`、`strata`、运行所需的 `env`，以及 `start_offset_s`（`multi` 模式下按窗口均匀展开，
  `single` 模式下为 `null`，此时按 `order` 顺序执行）；
- `summary`——各租户的工作流数量、实际用到的不同用例数，以及最终的分层分布。

相同的种子与相同的数据包会产出逐字节一致的结果。

## 出处与许可

- HoVer —— Jiang 等，*HoVer: A Dataset for Many-Hop Fact Extraction And Claim
  Verification*，Findings of EMNLP 2020。断言派生自 HotpotQA（CC BY-SA 4.0）。
- 维基百科正文（经由 `BeIR/hotpotqa`，2017 年 10 月快照）—— CC BY-SA。
- Spider —— Yu 等，*Spider: A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Domain Semantic Parsing and Text-to-SQL Task*，EMNLP 2018 —— CC BY-SA 4.0。
  InterCode —— Yang 等，*InterCode: Standardizing and Benchmarking Interactive Coding
  with Execution Feedback*，NeurIPS 2023（代码 MIT）；其 SQL 环境的数据派生自 Spider。
- arXiv 摘要与元数据通过 arXiv API 获取并连同 arXiv id 一起保存，版权仍属原作者。
- Deep Research 包中的合成数列与任务文本由本仓库的 `build/build_deep_search.py` 生成。

## 全量重建

```bash
cd benchmark/evaluation/datasets
python build/build_intercode_sql.py
python build/build_deep_search.py
python build/build_hover.py            # 下载量最大，放最后跑
rm -rf /tmp/hover_build                # 数据包生成后即可删掉那份 976 MB 的语料 parquet
```

然后确认租户程序仍能加载并通过编译检查：

```bash
cd ../../..
jac check benchmark/evaluation/programs/*.jac
python -m utils.jac_static_parser benchmark/evaluation/programs/HoVer.jac
```
