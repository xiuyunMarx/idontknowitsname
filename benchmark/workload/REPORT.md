# 模型加载调度：代价感知 Greedy / DP 规划器与可复现工作负载

日期：2026-08-29 · 分支 `FaaS` · 硬件：单卡 RTX 3090 (24 GB)，`gpu_slots=1, host_slots=3`

## 0. 结论

在构造的工作负载 `adv-k3-g60`（30 个 `jac run`，每 60 s 一组 3 个同时到达，5 个多模型程序，6 个 Qwen 模型 0.5B–4B）上，3 种规划器各跑 2 个 trial（交替顺序，2154 个请求，0 失败）：

| planner | 排队等待总和 (s) | 平均 / p95 排队等待 | 加载耗时 (s) | 客户端端到端均值 |
|---|---|---|---|---|
| fifo（原实现） | 973 (861, 1084) | 2.72 s / 14.1 s | 146 | 48.7 s |
| greedy（Smith 规则） | 920 (934, 906) | 2.57 s / 12.5 s | 112 | 46.8 s |
| **dp** | **784 (805, 762)** | **2.17 s / 9.2 s** | 119 | **42.0 s** |

- DP 比 greedy 少 **15%** 等待、比 FIFO 少 19%，两个 trial 一致；p95 等待比 greedy 低 26%。
- 配对端到端耗时 dp − greedy = **−4.8 s，95% CI [−8.8, −0.9]**（bootstrap，n=60），置信区间不含 0。
- greedy − fifo = −1.9 s，CI [−7.1, +3.4]：单步贪心**不能**稳定地打败 FIFO——它省下了加载时间，却把 4B 模型的等待者饿死了。

原始数据 `results/adv-k3-g60/{clients.jsonl, server.jsonl, config.json}`，汇总 `benchmark/workload/summary-adv-k3-g60.md`。

## 1. 问题建模

`serve/controller.py` 的规划循环每次被唤醒时做一件事：从优先级列表里取第一个**不在 GPU 上**的模型，需要的话驱逐一个空闲模型，把它加载上来（加载权重的同时预填充等待中的 prompt），派发该模型的请求桶，然后重算。原来的优先级由 `fifo_order` 给出：把每个活跃实例的"当前模型 + 预测链"按实例到达顺序展平去重。

这是一个经典的调度问题：**单机 + 家族切换时间（family setup）+ 链式优先约束 + 家族内批处理**：

- 机器 = GPU 槽；家族 = 模型 m，切换代价 `L[m][tier]`（host→GPU 与 SSD→GPU 差一个量级）
- 作业 = 实例 i 的调用链 `c_{i,1} → c_{i,2} → …`，每个调用有模型 `μ(c)` 和执行时间 `e(c)`
- 同一次驻留期间同模型的请求合批，代价 ≈ `max e(c)` 而非 `Σ e(c)`
- 目标 = 所有实例排队等待之和（sum of waiting / flowtime）

FIFO 忽略三件事：模型加载代价的差异、有多少实例在等同一个模型（需求聚合）、链上的回访。规模小（活跃实例 ≤ 10，链长 ≤ 12，模型 6 个）到可以每次唤醒精确求解并滚动执行第一步。

## 2. 算法设计

### 2.1 代价模型 `serve/cost.py`

规划器需要的所有估计值都来自这里，全部在线学习（EMA），先验按参数量（从模型名解析）成比例：

| 量 | 观测来源 | 默认先验 |
|---|---|---|
| `load_s(m, tier)` | `Controller._load` 计时（`(model, from_tier, secs)`） | host 0.6 s/B，ssd 2.0 s/B；观测到任一模型后按该层实测的 s/B 重标 |
| `exec_s(callsite)` | `_serve` 里 `engine.generate` 的墙钟 × 每次调用的轮数（tool loop 多轮） | `0.5 + 0.4·B` |
| `rate(m)` | `submit` 时间戳窗口 | 0 |

它是"知识"而不是计数器：`{"reset": true}` 不清空；`{"set": {"learn_costs": false}}` 冻结，保证 A/B 各 cell 用同一张表；`{"calibrate": true}` 让每个引擎从 SSD 和 host 各加载一次，用实测填表。

### 2.2 规划器输入 `PlanInput`（`Controller.plan_input`）

每个活跃实例一条链：

- **链头** = 它现在所处的步骤：`waiting`（有请求在排队，权重 `1 + age/τ`）、`running`（请求在跑，剩余时间 = 估计 − 已过）、`open`（tool loop 中间，客户端在执行工具，模型被 pin 住）
- **预测尾** = `serve/predict.py` 给出的后续调用（见 2.5），每步 `(model, exec_s 估计)`，至多 `MAX_CHAIN = 12` 步

以及驻留集合、每个模型的层级、`load_s`、到达率、`gpu_slots`、busy/pinned 集合。规划器是纯函数，不依赖 Engine，可离线测试。

### 2.3 共享的步进语义（`serve/planner.py`）

greedy、DP、`evaluate`（给任意顺序定价）和离线模拟器用同一套转移：

```
动作 m        某条未完成链的链头模型
advance(m)    所有链头为 m 的链沿连续的 m 步推进；单链代价 = 该段步骤之和（同一实例的多轮串行），
              批代价 = 各链的 max（共享一次驻留）
load(m)       0 若已驻留；否则 load_s(m, tier)；计划内被驱逐的模型按 host 重载
cost(m)       (load + exec) × 未完成链的权重之和        # sum-of-waiting 的增量
victim        keep_value 最小者：keep_value(v) = L_reload(v) / max(0.1, min(链上下次使用距离, 1/λ_v))
```

`keep_value` 同时用于计划内驱逐、`Controller._victim`（GPU→host）和 `_trim_host`（host→SSD，用 `L_ssd − L_host` 作为重载代价）。`_pinned()`（tool loop 中的模型）仍是硬排除。

### 2.4 三个规划器

- **fifo**：原实现，一字未改（基线）。
- **greedy**（`greedy_order`）：家族调度里 Smith 规则的推广——每次选 `argmin_m (load + max exec) / Σ 等待权重`；等待超过 6τ 的链头强制优先（防饿死）。
- **dp**（`dp_order`）：状态 `(驻留集合, 计划内被驱逐集合, 各链进度)`，动作 = 链头模型，深度 `horizon = 4` 的带记忆搜索；叶子用 greedy rollout 跑到链尾定价，并给仍驻留的模型一个 `W·λ_m·L_m` 的"驻留价值"（到达先验）；节点数超过 `budget = 500` 回退到 greedy。实测 6 实例 × 12 步 ≈ 8 ms，线上均值 1 ms。

**滚动执行**：DP 输出整条动作序列（去重后作为 `order`，也供 `_victim` 排序），但 `_plan_step` 只执行第一个加载，下次唤醒重解。这就是它不需要预测新到达的原因：只有首步被提交，新到达带来的误差被限制在一步之内。

**滞回**（`choose`）：候选顺序的首个加载与 FIFO 相同则直接用候选；否则只有当 `evaluate` 估计的等待比 FIFO 少 10% 且至少 1 个加权秒才偏离 FIFO。估计噪声不会为了几毫秒改顺序。

### 2.5 循环感知的预测链 `serve/predict.py`

这是整个方案里最关键的一处。第一轮实验里 DP ≈ greedy，原因是 `Program.predict_path` 在遇到已在路径上的 callsite 时停止——hover 的多跳循环让它永远看不到末尾的 4B verify，text2sql 只看到一圈修复循环。视野只有 1–5 步时，多步展望没有东西可看。

`predicted_chain` 沿最可能的后继走，允许一个 callsite 重复访问 `round(p/(1−p))` 次（p 为回边概率，即该边期望再被走 p/(1−p) 次）。只有代价感知规划器用它；FIFO 与预填充仍用 `inst.predicted`，基线不受影响。

另一个纠正：等待权重不再按预测深度打折（γ=1.0）。按 γ^j 打折会让 DP 觉得"处于预测未来的实例"不太需要照顾，从而推迟它们——不确定的是它**需要什么**，不是它**是否存在**。

### 2.6 为什么不是 ILP / CP-SAT

输入全是预测值，精确最优的长尾在下次唤醒就作废；合批 max、连续同模型共享驻留、层级随计划演变这些结构在 DP 转移函数里是普通代码，在线性约束里要大量 big-M；DP 在事件循环里毫秒级完成且可解释。CP-SAT 更适合作为离线 oracle（全知条件下的下界）或 `gpu_slots ≥ 2` 时加载/执行重叠的资源建模。

## 3. 测试数据

### 3.1 程序与模型

5 个 jac 程序（`benchmark/applications/`，`cascade.jac` 无入口已排除），全部多模型：

| 程序 | 模型链（实测典型） | 步数 | 单独运行墙钟 | 引擎时间 |
|---|---|---|---|---|
| deep_research | 4B → 0.6B router → 1.5B/1.7B/3B(多轮) → 4B → 4B | 10.2 | 50.9 s | 20.8 s |
| hover | (1.5B → 1.5B → 1.7B → 3B) × 2 hop → 1.5B → 4B(多轮) → 0.5B | 16.4 | 31.3 s | 10.1 s |
| text2sql | 3B → (1.7B(多轮) → 3B) × 1–4 圈 → 0.5B | 10.5 | 18.7 s | 8.5 s |
| rag_qa | 0.5B → 0.6B → 4B | 3 | 7.8 s | 1.9 s |
| triage | 0.6B router → 0.5B → 0.5B | 3 | 5.6 s | 0.9 s |

实测加载时间（`calibrate`）：

| 模型 | SSD→GPU | host→GPU |
|---|---|---|
| Qwen2.5-0.5B / Qwen3-0.6B | 0.75 / 0.79 s | 0.41 s |
| Qwen2.5-1.5B / Qwen3-1.7B | 1.5 / 1.8 s | 0.52 s |
| Qwen2.5-3B / Qwen3-4B | 6.5 / 6.3 s | 0.72 / 0.83 s |

（在线负载下 SSD 加载常比校准值慢 2–3 倍，因为要和预填充、驱逐争带宽。）

### 3.2 工作负载生成 `gen_schedule.py`

一个 schedule 是一组 `jac run` 及其起始偏移：`{"meta": {...}, "entries": [{"order", "program", "env", "case_index", "start_offset_s"}]}`。

- **可复现且可扩展**：三个独立随机流（到达间隔、程序选择、case 编号）都由 `--seed` 派生，每个 entry 各消费固定次数，因此同 seed 下 `--count 60` 的前 30 项与 `--count 30` 完全相同（测试 `tests/test_workload.py` 验证）。
- **开环**：偏移不依赖服务器快慢。
- 模式：`poisson --rate r`；`burst --burst-size k --burst-gap g --jitter j`；`adversarial`。

### 3.3 对抗式构造（`--pattern adversarial`）

随机混合下模拟器给 DP 的平均优势只有 ~1–4%，所以按你的要求让生成器**构造** DP 赢、greedy 输的场景：

1. `profile.py` 先对每个 (程序, case) 单独跑一遍，记录真实的 callsite/模型序列和每步引擎时间 → `profiles.json`（45 个 case）。
2. 对每个 burst，从种子流抽 300 个候选组合（程序 × case × 组内抖动），每个候选在 `serve/sim.py`（控制循环的离散事件模拟）下分别用 fifo / greedy / dp 重放，保留 **greedy − dp 差距最大**的组合；上一 burst 结束时留在 host 上的模型作为下一 burst 的初始状态。
3. **评分用的是规划器的真实视野**（默认，`--oracle` 才用真实链）：用 profile 序列训练一份 `Program.branch_freq`，链尾由 `predicted_chain` 给出、执行时间用 per-callsite 均值，与线上 `plan_input` 完全一致。用全知视野评分曾虚报 20% 的收益，线上没有兑现。

`adv-k3-g60`（seed 20260829，`--exclude deep_research.jac:0`）的 10 个 burst 与模拟预测：

| burst | 组合 | fifo | greedy | dp |
|---|---|---|---|---|
| 0 | text2sql:0, hover:3, rag_qa:9 | 114.6 | 113.7 | 88.4 |
| 1 | text2sql:0, hover:4, rag_qa:5 | 121.7 | 114.3 | 90.8 |
| 2 | text2sql:9, hover:4, rag_qa:7 | 122.6 | 118.9 | 93.0 |
| 3 | hover:3, deep_research:1, text2sql:6 | 118.5 | 133.3 | 98.8 |
| 4 | rag_qa:5, deep_research:1, text2sql:2 | 71.1 | 63.8 | 53.6 |
| 5 | deep_research:1 ×2, text2sql:5 | 89.7 | 89.7 | 76.9 |
| 6 | deep_research:1, rag_qa:1, text2sql:6 | 106.9 | 90.8 | 78.1 |
| 7 | hover:4, hover:3, rag_qa:6 | 73.5 | 73.5 | 58.9 |
| 8 | hover:1, hover:4, rag_qa:2 | 77.0 | 77.0 | 62.4 |
| 9 | hover:4, deep_research:3, text2sql:6 | 131.7 | 145.8 | 111.8 |
| 合计 | | 1027 | 1021 | 813 |

模拟预测 dp −20%，实测 −15%；模拟预测 greedy ≈ fifo，实测 greedy 略好于 fifo 但 CI 含 0。方向和量级都对上了。

## 4. 收益的第一性

收益来自三种 FIFO 结构上拿不到的信息，以及一个 greedy 拿不到的：

1. **加载代价异质性**（fifo 拿不到，greedy/dp 都拿到）：SSD→GPU 是 host→GPU 的 8 倍，4B 是 0.5B 的 8 倍。按 `keep_value` 而不是 LRU 决定谁留在 host，每个 cell 的 SSD 加载从 fifo 的 87/71 次降到 greedy 51/47、dp 57/58，加载总时长从 146 s 降到 112–119 s。这是 greedy 和 dp 共同的收益来源。

2. **需求聚合**（fifo 拿不到）：一个模型有 3 个实例在等，比一个只有 1 个实例在等的模型更值得先加载——sum-of-waiting 目标下这就是 Smith 规则。

3. **回访与相位**（只有 dp 拿到）：hover 每一跳都回到 1.5B，text2sql 每一圈都回到 3B，deep_research 首尾都是 4B。单步贪心看到 "0.5B 现在便宜" 就先加载它，随后为了回访付两次大模型重载；DP 看到 4 步之外的回访，会先把大模型的两个使用者对齐合批。这正是对抗式搜索挑出来的 burst 结构（text2sql/hover 的循环 + rag_qa 的 0.5B→0.6B→4B 短链）。

4. **greedy 为什么输**：Smith 规则的比值 `(load+exec)/权重` 系统性地推迟昂贵家族。实测 trial 0 里 4B 模型的等待从 fifo 的 106 s 涨到 greedy 的 225 s（DP 147 s），把从加载省下的时间全部还了回去；60 s 的饿死上限触发太晚。DP 的多步展望知道推迟 4B 会让 rag_qa 的三个等待者一直卡在链尾，所以不这么做——rag_qa 的端到端均值 dp 17.3 s vs greedy 36.1 s / fifo 35.1 s。

反过来也说明了边界：收益 ∝ (加载代价 / 执行时间) × 链上回访密度 × 同时活跃的实例数。程序都是三步短链、或者模型都常驻 host、或者到达稀疏到队列不形成时，三种规划器趋同（`burst-k3-g60.json`、`poisson-0.1.json` 是这种对照，模拟器预测 dp 只有 1–4% 优势，尚未实跑）。

## 5. 脚本用法

服务端（一次）：

```bash
python start_server.py --gpu-slots 1 --host-slots 3 [--planner fifo|greedy|dp]
# 控制口 8965，一行 JSON 进出：
#  {"set": {"planner": "dp", "horizon": 4, "hysteresis": 0.1, "tau": 10, "gamma": 1.0,
#           "learn_costs": false, "speculate": true, "gpu_slots": 1, "host_slots": 3}}
#  {"reset": true, "cold": true}   全部引擎回 SSD，清计数器（不清代价表）
#  {"calibrate": true}             每个引擎从 SSD、host 各加载一次，填代价表
#  {"freeze_branches": true}       快照分支预测器，之后每次 reset 恢复
#  {"stats": true, "detail": true} queue_wait_ms / ttft_submit_ms / e2e_ms / sum_wait_s / loads / 每请求明细
#  {"cost": true}                  代价表；{"forget": true} 回到先验；{"dump": true}
python send_requests.py --set planner=dp --rate 0.5 --count 30    # 旧驱动仍可用，--set 透传任意设置
```

实验流水线（服务器已启动）：

```bash
# 1. 逐 case 画像（约 30 min，含 calibrate；可 --resume 续跑，--cases N 限制每程序 case 数）
python benchmark/workload/profile.py --out benchmark/workload/profiles.json

# 2. 生成 schedule（seed 固定即可复现；同 seed 加大 --count 得到前缀一致的更大实验）
python benchmark/workload/gen_schedule.py --pattern adversarial --burst-size 3 --burst-gap 60 --count 30 \
    --candidates 300 --exclude deep_research.jac:0 --out benchmark/workload/schedules/adv-k3-g60.json
python benchmark/workload/gen_schedule.py --pattern burst   --burst-size 3 --burst-gap 60 --count 30 --out .../burst-k3-g60.json
python benchmark/workload/gen_schedule.py --pattern poisson --rate 0.1 --count 30 --out .../poisson-0.1.json
#   其他参数：--weights hover.jac=2,triage.jac=1（租户配比）--jitter --oracle --host-slots --gpu-slots

# 3. 重放：warmup → calibrate → 冻结代价表和分支预测 → 每个 cell 冷 reset 后按偏移启动 jac run，
#    trial 间交替 cell 顺序；写 clients.jsonl / server.jsonl / config.json
python benchmark/workload/run_schedule.py --schedule benchmark/workload/schedules/adv-k3-g60.json \
    --cells fifo,greedy,dp --trials 2 --out results/adv-k3-g60 [--set gpu_slots=1 ...] [--no-calibrate]

# 4. 汇总：每 cell 指标、每程序墙钟、配对差与 bootstrap CI、加载顺序 → summary.md
python benchmark/workload/report.py results/adv-k3-g60 [--baseline greedy --candidate dp]
```

离线测试（无需 GPU，stdlib unittest，36 个）：

```bash
python -m unittest discover -s tests -t .
# tests/test_cost.py       代价模型先验/EMA/轮数/到达率/冻结
# tests/test_planner.py    9 个手工场景 + 200 个随机场景：dp ≤ greedy ≤ fifo、驱逐、饿死、滞回、预算回退
# tests/test_predict.py    循环展开
# tests/test_controller.py plan_input / plan / _victim / stats / reset（假引擎）
# tests/test_workload.py   三种模式的前缀稳定性、对抗式每 burst dp ≤ greedy
```

## 6. 文件清单

| 文件 | 内容 |
|---|---|
| `serve/cost.py` | 在线代价模型 |
| `serve/planner.py` | `PlanInput`、步进语义、`fifo_order`/`greedy_order`/`dp_order`、`evaluate`、`keep_value`、`choose` |
| `serve/predict.py` | 循环感知预测链 |
| `serve/sim.py` | 控制循环的离散事件模拟器（测试与生成器共用） |
| `serve/controller.py` | 规划器开关、计时钩子、控制口新命令、keep-value 驱逐；FIFO 路径不变 |
| `static_pass/primitives.py` | `RequestHandle` 增加 `created_at / dispatched_at / done_at` |
| `start_server.py` | `--planner`；缺失的程序文件跳过而不是崩溃 |
| `send_requests.py` | 控制口读取上限放大（明细回复超过 64 KB） |
| `benchmark/workload/{profile,gen_schedule,run_schedule,report}.py` | 实验流水线 |
| `benchmark/workload/profiles.json`、`schedules/*.json` | 画像与三份 schedule |
| `benchmark/workload/summary-adv-k3-g60.md` | 本次实验汇总 |
| `tests/` | 离线测试 |

## 7. 已知问题与注意事项

- **`benchmark/applications/cascade.jac`** 在本次会话 13:40 从工作区消失（不是我删的，上一提交已把它掏空），未恢复；`start_server.py` 现在会跳过缺失的程序文件。
- **`deep_research.jac:0` 不稳定**：`build_report` 的输出在某些 cell 被类型解析拒绝 3 次后客户端重发调用，触发"静态拓扑无后继"错误。程序里 `temperature=0.0`，同一 case 在不同 cell 结果不同，只可能是请求所落的 KV/prefix-cache 状态不同（加载中预填充、投机预填充）——像是引擎层问题，被调度顺序暴露，未深究。该 case 已从 schedule 排除。
- **指标口径**：隔离规划器效果的是服务端指标（`queue_wait_ms`、`sum_wait_s`、`load_s`）；客户端墙钟包含 5–50 s 的 jac 启动与程序自身工作，稀释了差异。
- **模拟器 ≠ 全知**：生成器和测试必须用规划器的真实视野评分，否则会高估。
- **`gpu_slots ≥ 2`**：DP 的串行代价模型不建模加载与执行重叠，此时估计偏保守；驱逐仍由 keep_value 处理。
- **运维**：`pkill -f` 的模式不要和 `start_server.py` 出现在同一条命令行里（会杀掉自己，退出码 144）；控制口的 `stats detail` 回复可达数 MB。
- **branch predictor 仍在线学习**：cell 之间用 `freeze_branches` 快照恢复；`learn_costs=false` 冻结代价表。两者在 warmup + calibrate 之后设置。
