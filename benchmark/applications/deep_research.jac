"""Deep-research supervisor with four specialist sub-agents on four different models.

    decompose_task[Qwen3-4B] -> visit by router[Qwen3-0.6B] -> 1..3 of
        web_research[2.5-1.5B] | paper_research[Qwen3-1.7B] | data_research[2.5-3B] | fact_check[0.5B]
    ... $DEEP_ROUNDS rounds (default 2) ...  -> summarize[Qwen3-4B] -> build_report[Qwen3-4B]

The workers picked in a round form a partial order (the walker runs them in
queue order, the scheduler may not), so this program exercises intra-request
interleaving and router prediction. Sources are an inline slice of the
evaluation pack. Inputs: $TASK or $DEEP_TASK_INDEX."""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob PORT = int(os.environ.get("SERVE_PORT") or "8964");

glob supervisor = InterceptorLLM(model_name="Qwen/Qwen3-4B", comm_ip="localhost", comm_port=PORT, program_name="deep_research");
glob router = InterceptorLLM(model_name="Qwen/Qwen3-0.6B", comm_ip="localhost", comm_port=PORT, program_name="deep_research");
glob web_llm = InterceptorLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="deep_research");
glob paper_llm = InterceptorLLM(model_name="Qwen/Qwen3-1.7B", comm_ip="localhost", comm_port=PORT, program_name="deep_research");
glob data_llm = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="deep_research");
glob fact_llm = InterceptorLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="deep_research");

glob ROUNDS = int(os.environ.get("DEEP_ROUNDS") or "2");

glob DISPATCH_INTENT: str = "assign this round of the plan to the research sub-agents that own the open questions: the web agent for background and system documentation, the paper agent for published results, the data agent for the measured numbers, the fact-check agent for a claim that needs an authoritative source";

glob WEB: dict[str, str] = {
    "Cache (computing)": "In computing, a cache is a hardware or software component that stores data so that future requests for that data can be served faster; the data stored in a cache might be the result of an earlier computation or a copy of data stored elsewhere. A cache hit occurs when the requested data can be found in a cache, while a cache miss occurs when it cannot. Cache hits are served by reading data from the cache, which is faster than recomputing a result or reading from a slower data store; thus, the more requests that can be served from the cache, the faster the system performs.",
    "Scheduling (computing)": "In computing, scheduling is the action of assigning resources to perform tasks. The resources may be processors, network links or expansion cards. The tasks may be threads, processes or data flows.",
    "Speculative execution": "In computer science and processor design, speculative execution or speculation is a design characteristic where a processor is able to perform potentially useful work by guessing what work is needed next, compute them prematurely, and immediately applying or discarding the results when the work's necessity is eventually known. Since the 21st century, speculation is commonly seen superscalar central processors (CPUs), as an optimization technique to increase its performance. It is often used in combination with branch prediction to increase the accuracy of guessing process.",
    "API": "An application programming interface (API) is a connection between computers or between computer programs. It is a type of software interface, offering a service to other pieces of software. A document or standard that describes how to build such a connection or interface is called an API specification. A computer system that meets this standard is said to implement or expose an API. The term API may refer either to the specification or to the implementation.",
    "Attention (machine learning)": "In machine learning, attention is a method that determines the importance of each component in a sequence relative to the other components in that sequence. In natural language processing, importance is represented by \"soft\" weights assigned to each word in a sentence. More generally, attention encodes vectors called token embeddings across a fixed-width sequence that can range from tens to millions of tokens in size.",
    "Batch processing": "In computing, batch processing is the running of a software job in an automated and unattended way. A user schedules a job to run and then waits for a processing system to run it. Typically, a job is scheduled to run at a configured time of day or when an event occurs or when computer resources are available.",
    "Benchmark (computing)": "In computing, a benchmark is the act of running a computer program, a set of programs, or other operations, in order to assess the relative performance of an object, normally by running a number of standard tests and trials against it.",
    "Branch predictor": "In computer architecture, a branch predictor is a digital circuit that tries to guess which way a branch will go before this is known definitively. The purpose of the branch predictor is to improve the flow in the instruction pipeline. Branch predictors play a critical role in achieving high performance in many modern pipelined microprocessor architectures."
};
glob PAPERS: dict[str, str] = {
    "arxiv-2606-01751": "SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving (Quqing Zhang, Kai Chen, Ning Liao et al., arXiv:2606.01751v2, 2026). In long-context LLM serving, the prefill stage often dominates time-to-first-token and computational cost. Although Prefix Cache in vLLM/PagedAttention has been widely used to reuse identical prompt prefixes, repeated content in practical applications frequently appears as non-prefix, cross-request, cross-turn, and cross-agent segments, which makes conventional cache mechanisms insufficient. This paper presents SparseX, a segment-level KV Cache sharing method for common serving scenarios. SparseX uses contiguous token segments as reuse units and ",
    "arxiv-2605-24022": "Adaptive KV Cache Reuse for Fast Long-Context LLM Serving (Fei li, Song Liu, Yan Liu et al., arXiv:2605.24022v1, 2026). In long-context Large Language Model (LLM) inference, the Time-To-First-Token (TTFT) latency incurred by the prefill stage has become the foremost bottleneck limiting interactive performance and deployment cost. KV Cache reuse offers a direct path to reduce redundant prefill, yet traditional prefix caching applies only to strict-prefix scenarios; directly reusing KV Cache in non-prefix settings breaks the cross-chunk global attention relationships and causes significant degradation in generation quality. When reusable KV Cache is offloaded to GPU-external cache pools, I/O o",
    "arxiv-2502-03589": "HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference (Zeyu Zhang, Haiying Shen, Shay Vargaftik et al., arXiv:2502.03589v1, 2025). Disaggregated Large Language Model (LLM) inference has gained popularity as it separates the computation-intensive prefill stage from the memory-intensive decode stage, avoiding the prefill-decode interference and improving resource utilization. However, transmitting Key-Value (KV) data between the two stages can be a bottleneck, especially for long prompts. Additionally, the computation time overhead for prefill and decode is key for optimizing Job Completion Time (JCT), and KV data size can become prohibitive for ",
    "arxiv-2510-08055": "From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill (Gunjun Lee, Jiwon Kim, Jaiyoung Park et al., arXiv:2510.08055v2, 2025). Large Language Model (LLM) inference in production must meet stringent service-level objectives for both time-to-first-token (TTFT) and time-between-token (TBT) while maximizing throughput under fixed compute, memory, and interconnect budgets. Modern serving systems adopt stall-free scheduling techniques such as chunked prefill, which splits the processing of long prompts along the token dimension and interleaves prefill with ongoing decode iterations. While effective at stabilizing TBT, chunked prefill incurs substantial overhe",
    "arxiv-2607-09172": "Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations (Nada Zine, Tristan Coignion, Vincenzo Stoico et al., arXiv:2607.09172v2, 2026). Large Language Models are reshaping how software is developed and maintained. They are typically deployed in production using inference engines such as vLLM, which can efficiently serve pre-trained, highly configurable models. While prior work has focused on model architectures and hardware acceleration, the impact of inference engine configuration on energy consumption, performance, and output quality remains poorly understood. In this paper, we present a large-scale controlled study of three selected vLLM co",
    "arxiv-2602-09113": "Benchmarking the Energy Savings with Speculative Decoding Strategies (Rohit Dutta, Paramita Koley, Soham Poddar et al., arXiv:2602.09113v1, 2026). Speculative decoding has emerged as an effective method to reduce latency and inference cost of LLM inferences. However, there has been inadequate attention towards the energy requirements of these models. To address this gap, this paper presents a comprehensive survey of energy requirements of speculative decoding strategies, with detailed analysis on how various factors -- model size and family, speculative decoding strategies, and dataset characteristics -- influence the energy optimizations.",
    "arxiv-2607-08690": "A Practical Investigation of Training-free Relaxed Speculative Decoding (Guoxuan Xia, Luka Ribar, Paul Balanca, arXiv:2607.08690v1, 2026). Speculative decoding accelerates sampling from an autoregressive LLM by using a faster auxiliary model to draft tokens which are then verified in parallel by the LLM. Standard speculative decoding is lossless: its rejection and resampling steps exactly preserve the LLM's sampling distribution. Recent work argues that relaxing this strict guarantee can yield further speed-ups, controlled capability-speed trade-offs, or even capability gains. We practically investigate training-free relaxed speculative decoding techniques, unify existing approaches within a",
    "arxiv-1901-05758": "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads (Myeongjae Jeon, Shivaram Venkataraman, Amar Phanishayee et al., arXiv:1901.05758v2, 2019). With widespread advances in machine learning, a number of large enterprises are beginning to incorporate machine learning models across a number of products. These models are typically trained on shared, multi-tenant GPU clusters. Similar to existing cluster computing workloads, scheduling frameworks aim to provide features like high efficiency, resource isolation, fair sharing across users, etc. However Deep Neural Network (DNN) based workloads, predominantly trained on GPUs, differ in two significant ways from traditional b",
    "arxiv-1912-02322": "Perseus: Characterizing Performance and Cost of Multi-Tenant Serving for CNN Models (Matthew LeMay, Shijian Li, Tian Guo, arXiv:1912.02322v2, 2019). Deep learning models are increasingly used for end-user applications, supporting both novel features such as facial recognition, and traditional features, e.g. web search. To accommodate high inference throughput, it is common to host a single pre-trained Convolutional Neural Network (CNN) in dedicated cloud-based servers with hardware accelerators such as Graphics Processing Units (GPUs). However, GPUs can be orders of magnitude more expensive than traditional Central Processing Unit (CPU) servers. These resources could also be under-utilized f",
    "arxiv-2109-11449": "Dynamic Mixture of Experts Models for Online Prediction (Parfait Munezero, Mattias Villani, Robert Kohn, arXiv:2109.11449v2, 2021). A mixture of experts models the conditional density of a response variable using a mixture of regression models with covariate-dependent mixture weights. We extend the finite mixture of experts model by allowing the parameters in both the mixture components and the weights to evolve in time by following random walk processes. Inference for time-varying parameters in richly parameterized mixture of experts models is challenging. We propose a sequential Monte Carlo algorithm for online inference and based on a tailored proposal distribution built on ideas from lin"
};
glob METRICS: dict[str, str] = {
    "hit_rate_vs_prompt_reuse": "Synthetic benchmark series (seeded, not measured). Prefix-cache hit rate in percent as prompt reuse rises from 0 to 100 percent: 79.72, 67.20, 65.00, 45.99, 45.71, 43.87, 40.84, 7.59",
    "ttft_ms_vs_queue_depth": "Synthetic benchmark series (seeded, not measured). TTFT in milliseconds at queue depth 1 through 8: 90.45, 81.63, 75.87, 74.29, 66.50, 52.08, 48.85, 42.43",
    "ttft_ms_baseline": "Measured on this repository's RTX 3090 trial, Qwen2.5-3B-Instruct. TTFT per successor call, baseline condition, milliseconds: 34.61, 30.39, 37.58, 19.02",
    "ttft_ms_proactive": "Measured on this repository's RTX 3090 trial, Qwen2.5-3B-Instruct. TTFT per successor call, proactive prefill condition, milliseconds: 30.26, 12.84, 29.44, 16.08",
    "cached_prompt_tokens": "Measured on this repository's RTX 3090 trial. Prefix-cache hits per successor call, baseline then proactive: 32 to 96, 48 to 96, 112 to 352, 48 to 112",
    "tbt_ms": "Measured on this repository's RTX 3090 trial. Mean time between tokens, single-task mode: baseline 9.13, speculative during decode 9.90, speculative only while idle 9.21"
};
glob SOURCES: dict[str, str] = {
    "arxiv-2502-03589-claim": "Disaggregated Large Language Model (LLM) inference has gained popularity as it separates the computation-intensive prefill stage from the memory-intensive decode stage, avoiding the prefill-decode interference and improving resource utilization. (source: arXiv:2502.03589v1, HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference)",
    "arxiv-2602-09113-claim": "Speculative decoding has emerged as an effective method to reduce latency and inference cost of LLM inferences. However, there has been inadequate attention towards the energy requirements of these models. (source: arXiv:2602.09113v1, Benchmarking the Energy Savings with Speculative Decoding Strategies)",
    "arxiv-2605-24022-claim": "In long-context Large Language Model (LLM) inference, the Time-To-First-Token (TTFT) latency incurred by the prefill stage has become the foremost bottleneck limiting interactive performance and deployment cost. (source: arXiv:2605.24022v1, Adaptive KV Cache Reuse for Fast Long-Context LLM Serving)",
    "arxiv-2608-14624-claim": "Multi-agent LLM systems have emerged as an important deployment paradigm for AI services, where each user request is decomposed into a sequence of specialized agents. (source: arXiv:2608.14624v1, Learning Agent Execution for KV-Cache Management in Agentic Serving)",
    "arxiv-2312-07104-claim": "Large language models (LLMs) are increasingly used for complex tasks that require multiple generation calls, advanced prompting techniques, control flow, and structured inputs/outputs. (source: arXiv:2312.07104v2, SGLang: Efficient Execution of Structured Language Model Programs)",
    "arxiv-2604-03143-claim": "Multi-agent LLM applications organize execution in synchronized rounds where a central scheduler gathers outputs from all agents and redistributes the combined context. (source: arXiv:2604.03143v1, TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing)",
    "arxiv-2606-01751-claim": "In long-context LLM serving, the prefill stage often dominates time-to-first-token and computational cost. (source: arXiv:2606.01751v2, SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving)",
    "arxiv-2512-17910-claim": "Modern large language model (LLM) systems increasingly rely on multi-turn pipelines that are composed of multiple task-specific adapters, yet existing serving frameworks remain inefficient, incurring substantial recomputation overhead when switching between adapters. (source: arXiv:2512.17910v1, Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA)"
};
glob TASKS: list[str] = [
    "Assess whether prefix caching or recomputing the prompt is the better lever for cutting time to first token in multi-step agent workloads. Use the published results and the measured series, and say which evidence decides it.",
    "Does the series ttft_ms_vs_queue_depth support the claim that prefill-decode disaggregation helps with holding tail latency under a service level objective? Report what the numbers show and whether the sources agree.",
    "Which published work introduced chunked prefill, what problem did it solve, and what background does a reader need before the result makes sense?",
    "What does speculative decoding cost, and under what conditions does that cost outweigh its benefit for shortening end-to-end generation latency?"
];

def pick_task() -> str {
    direct = str(os.environ.get("TASK") or "");
    if direct {
        return direct;
    }
    return TASKS[int(os.environ.get("DEEP_TASK_INDEX") or "0") % len(TASKS)];
}

glob STOPWORDS: set[str] = set(
    "a an and are as at be by for from has have in into is it its of on or that the their to was were with".split()
);

def terms(text: str) -> set[str] {
    words: set[str] = set(text.lower().replace('"', " ").replace("'", " ").replace(",", " ").replace(".", " ").split());
    return words.difference(STOPWORDS);
}

def overlap(query: str, text: str) -> int {
    return len(terms(query) & terms(text));
}

def rank(query: str, table: dict[str, str], top: int) -> list[str] {
    scored: list[tuple[int, str]] = [];
    for key in table {
        scored.append((overlap(query, key + " " + table[key]) + 2 * overlap(query, key), key));
    }
    scored.sort(reverse=True);
    out: list[str] = [];
    for entry in scored[:top] {
        if entry[0] > 0 {
            out.append(entry[1]);
        }
    }
    return out;
}

def resolve(key: str, table: dict[str, str]) -> str {
    want = key.strip().lower();
    for candidate in table {
        if candidate.lower() == want {
            return candidate;
        }
    }
    for candidate in table {
        if want and (want in candidate.lower() or candidate.lower() in want) {
            return candidate;
        }
    }
    return "";
}

def lead(text: str) -> str {
    return text.split(". ")[0] + ".";
}

def web_search(query: str) -> str {
    time.sleep(0.30 * TOOL_SCALE);
    hits = rank(query, WEB, 3);
    if not hits {
        return "No page matched. Known pages: " + ", ".join(WEB.keys());
    }
    lines: list[str] = [];
    for title in hits {
        lines.append(title + ": " + lead(WEB[title]));
    }
    return "\n".join(lines);
}

def open_page(title: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    key = resolve(title, WEB);
    if not key {
        return f"No page titled '{title}'. Use web_search first.";
    }
    return key + ": " + WEB[key];
}

def paper_search(query: str) -> str {
    time.sleep(0.30 * TOOL_SCALE);
    hits = rank(query, PAPERS, 3);
    if not hits {
        return "No paper matched. Known ids: " + ", ".join(PAPERS.keys());
    }
    lines: list[str] = [];
    for pid in hits {
        lines.append(pid + ": " + lead(PAPERS[pid]));
    }
    return "\n".join(lines);
}

def read_abstract(paper_id: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    key = resolve(paper_id, PAPERS);
    if not key {
        return f"No paper with id '{paper_id}'. Use paper_search first.";
    }
    return key + ": " + PAPERS[key];
}

def metric_series(metric: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    key = resolve(metric, METRICS);
    if not key {
        return "No such metric. Known metrics: " + ", ".join(METRICS.keys());
    }
    return METRICS[key];
}

def compute_stats(series: str) -> str {
    time.sleep(0.20 * TOOL_SCALE);
    values: list[float] = [];
    for token in series.replace(",", " ").replace(":", " ").split() {
        try {
            values.append(float(token));
        } except Exception {
            continue;
        }
    }
    if not values {
        return "No numbers found in that series.";
    }
    total = 0.0;
    for value in values {
        total += value;
    }
    mean = total / len(values);
    return f"n={len(values)} min={min(values)} max={max(values)} mean={round(mean, 2)} first={values[0]} last={values[-1]}";
}

def check_source(topic: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    key = resolve(topic, SOURCES);
    if not key {
        return "No source on that topic. Known topics: " + ", ".join(SOURCES.keys());
    }
    return key + ": " + SOURCES[key];
}

sem web_search = "Search the offline web index and return the best page titles with their opening sentence.";
sem web_search.query = "keywords describing the background being researched";
sem open_page = "Read one page of the web index in full.";
sem open_page.title = "the page title, exactly as web_search reported it";
sem paper_search = "Search the paper index and return the best paper ids with their opening line.";
sem paper_search.query = "keywords describing the published work being looked for";
sem read_abstract = "Read the full record of one paper, including venue and reported result.";
sem read_abstract.paper_id = "the paper id, exactly as paper_search reported it";
sem metric_series = "Fetch one measured series from the benchmark result store.";
sem metric_series.metric = "the metric name, for example ttft_ms_baseline or cached_prompt_tokens";
sem compute_stats = "Compute count, minimum, maximum, mean, first and last over the numbers in a series.";
sem compute_stats.series = "the series text returned by metric_series";
sem check_source = "Look up the authoritative statement on one topic to confirm or refute a claim.";
sem check_source.topic = "the topic key, for example prefix-cache or chunked-prefill";


obj ResearchReport {
    has title: str;
    has mainText: str;
    has tldr: list[str];
}

node ResearchSupervisor {
    has charter: str = "run one round at a time: name the open questions, hand them to the sub-agents that own them, and stop once the questions are answered";

    def decompose_task(task: str, web: str, papers: str, data: str, facts: str) -> list[str] by supervisor(
        temperature=0.0,
        max_tokens=512,
        max_output_retries=3
    );
}

node WorkerAgent {}

node WebSearchAgent(WorkerAgent) {
    def web_research(task: str, plan: list[str]) -> str by web_llm(
        tools=[web_search, open_page],
        temperature=0.0,
        max_tokens=320,
        max_react_iterations=5
    );

    can work with DeepResearch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.web_findings = self.web_research(visitor.task, visitor.plan);
    }
}

node PaperSearchAgent(WorkerAgent) {
    def paper_research(task: str, plan: list[str]) -> str by paper_llm(
        tools=[paper_search, read_abstract],
        temperature=0.0,
        max_tokens=320,
        max_react_iterations=5
    );

    can work with DeepResearch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.paper_findings = self.paper_research(visitor.task, visitor.plan);
    }
}

node DataAnalysisAgent(WorkerAgent) {
    def data_research(task: str, plan: list[str]) -> str by data_llm(
        tools=[metric_series, compute_stats],
        temperature=0.0,
        max_tokens=320,
        max_react_iterations=5
    );

    can work with DeepResearch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.data_findings = self.data_research(visitor.task, visitor.plan);
    }
}

node FactCheckAgent(WorkerAgent) {
    def fact_check(task: str, plan: list[str]) -> str by fact_llm(
        tools=[check_source],
        temperature=0.0,
        max_tokens=256,
        max_react_iterations=4
    );

    can work with DeepResearch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.fact_findings = self.fact_check(visitor.task, visitor.plan);
    }
}

walker DeepResearch {
    has task: str = "";
    has plan: list[str] = [];
    has rounds: int = 0;
    has web_findings: str = "";
    has paper_findings: str = "";
    has data_findings: str = "";
    has fact_findings: str = "";
    has summary: str = "";
    has report: ResearchReport | None = None;

    def summarize(task: str, web: str, papers: str, data: str, facts: str) -> str by supervisor(
        temperature=0.0,
        max_tokens=384
    );

    def build_report(summary: str) -> ResearchReport by supervisor(
        temperature=0.0,
        max_tokens=384,
        max_output_retries=3
    );

    can setup with Root entry {
        if not [-->][?:ResearchSupervisor] {
            sup = ResearchSupervisor();
            here ++> sup;
            sup ++> WebSearchAgent();
            sup ++> PaperSearchAgent();
            sup ++> DataAnalysisAgent();
            sup ++> FactCheckAgent();
            sup ++> sup;
        }
        visit [-->][?:ResearchSupervisor];
    }

    can plan with ResearchSupervisor entry {
        self.rounds += 1;
        time.sleep(0.20 * TOOL_SCALE);
        self.plan = here.decompose_task(
            self.task, self.web_findings, self.paper_findings, self.data_findings, self.fact_findings
        );
        visit [-->][?:WorkerAgent] by router(select=(1, 3), intent=DISPATCH_INTENT, temperature=0.0);
        if self.rounds < ROUNDS {
            visit [-->][?:ResearchSupervisor];
        }
    }

    can merge with exit {
        time.sleep(0.20 * TOOL_SCALE);
        self.summary = self.summarize(
            self.task, self.web_findings, self.paper_findings, self.data_findings, self.fact_findings
        );
        self.report = self.build_report(self.summary);
        print("TASK:", self.task);
        print("ROUNDS:", self.rounds);
        print("SUMMARY:", self.summary);
        print("REPORT:", self.report);
    }
}

sem ResearchSupervisor = "the supervisor that owns the research plan";
sem ResearchSupervisor.decompose_task = "Name the open questions for the next research round as a JSON array of strings, one question per string, each addressed to the kind of agent that can answer it. Drop anything the findings already settle, and return an empty list when nothing is open.";
sem ResearchSupervisor.decompose_task.task = "the research task being investigated";
sem ResearchSupervisor.decompose_task.web = "what the web-search agent has found so far; empty in the first round";
sem ResearchSupervisor.decompose_task.papers = "what the paper-search agent has found so far; empty in the first round";
sem ResearchSupervisor.decompose_task.data = "what the data-analysis agent has found so far; empty in the first round";
sem ResearchSupervisor.decompose_task.facts = "what the fact-check agent has confirmed so far; empty in the first round";

sem WebSearchAgent = "the sub-agent that researches background and system documentation";
sem WebSearchAgent.web_research = "Answer the plan's background questions from the web index. Search, open the pages that matter, and finish with the findings, each attributed to its page title.";
sem WebSearchAgent.web_research.task = "the research task being investigated";
sem WebSearchAgent.web_research.plan = "this round's open questions";

sem PaperSearchAgent = "the sub-agent that researches published results";
sem PaperSearchAgent.paper_research = "Answer the plan's questions about published work. Search the paper index, read the records that matter, and finish with the findings, each attributed to its paper id and venue.";
sem PaperSearchAgent.paper_research.task = "the research task being investigated";
sem PaperSearchAgent.paper_research.plan = "this round's open questions";

sem DataAnalysisAgent = "the sub-agent that analyses the measured numbers";
sem DataAnalysisAgent.data_research = "Answer the plan's quantitative questions. Fetch the series the question needs, compute statistics over them, and finish with what the numbers show, quoting the values you used.";
sem DataAnalysisAgent.data_research.task = "the research task being investigated";
sem DataAnalysisAgent.data_research.plan = "this round's open questions";

sem FactCheckAgent = "the sub-agent that confirms claims against authoritative sources";
sem FactCheckAgent.fact_check = "Confirm or refute the plan's claims against the source store, and finish with each claim labelled confirmed or refuted with the source that decides it.";
sem FactCheckAgent.fact_check.task = "the research task being investigated";
sem FactCheckAgent.fact_check.plan = "this round's open questions";

sem DeepResearch = "the walker carrying one research task through the supervisor and its sub-agents";
sem DeepResearch.summarize = "Merge the findings of every sub-agent into one coherent summary that answers the task, attributing each fact to the agent that found it.";
sem DeepResearch.summarize.task = "the research task being investigated";
sem DeepResearch.summarize.web = "the web-search agent's findings";
sem DeepResearch.summarize.papers = "the paper-search agent's findings";
sem DeepResearch.summarize.data = "the data-analysis agent's findings";
sem DeepResearch.summarize.facts = "the fact-check agent's confirmations";
sem DeepResearch.build_report = "Turn the summary into a report with a title, the main text, and a three-bullet tldr.";
sem DeepResearch.build_report.summary = "the merged summary of the findings";


with entry:__main__ {
    DeepResearch(task=pick_task()) spawn root;
}
