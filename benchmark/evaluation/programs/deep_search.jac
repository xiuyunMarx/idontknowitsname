"""Deep-research agent with an LLM-routed supervisor graph (evaluation tenant).

    planner/supervisor  --(visit [-->][?:WorkerAgent] by llm)-->
        web-search agent | paper-search agent | data-analysis agent | fact-check agent
    ... one to three of them per round, up to $DEEP_ROUNDS rounds ...
        -> the walker merges every agent's findings and synthesizes the report

The supervisor plans with `decompose_task`, then the routing call picks which
sub-agents run: `select=(1, 3)` lets the model take one agent, or several that
the walker then runs in sequence within the round. A second planning round sees
the findings the first round produced, because every sub-agent writes its result
into walker state. The walker's exit ability is the merge point: `summarize`
reads all four slots, `build_report` turns the summary into a typed
`ResearchReport`.

Each sub-agent runs a byLLM tool loop over the offline source pack, so a round
spends most of its wall time inside tool turns. Those turns are the idle windows:
while one agent is reading a page, the topology already names its siblings and
the merge calls, and each completed agent extends the merge prefix by one more
bound argument.

Run:

    python start_server.py --model Qwen/Qwen2.5-7B-Instruct \
        --program deep_search:benchmark/evaluation/programs/deep_search.jac:8972

    DEEP_TASK_INDEX=0 jac run benchmark/evaluation/programs/deep_search.jac

Inputs: $TASK (task text) or $DEEP_TASK_INDEX into the source pack's task list.
Sources: $DEEP_SOURCES, a JSON object with `web`, `papers`, `metrics`, `sources`
and `tasks` sections; every task carries a rubric of the source ids a correct report
should cite. `datasets/deep_search_30.json` holds 30 tasks over arXiv abstracts and
Wikipedia leads (see datasets/README.md).
"""

import os;
import json;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob ROUNDS = int(os.environ.get("DEEP_ROUNDS") or "2");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=8972,
    program_name="deep_search"
);

glob DISPATCH_INTENT: str = "assign this round of the plan to the research sub-agents that own the open questions: the web agent for background and system documentation, the paper agent for published results, the data agent for the measured numbers, the fact-check agent for a claim that needs an authoritative source";


# --------------------------------------------------------------- source pack

def sources_path() -> str {
    here = os.path.dirname(os.path.abspath(__file__));
    return str(os.environ.get("DEEP_SOURCES")
               or os.path.join(here, "..", "datasets", "deep_search_30.json"));
}

def read_pack() -> any {
    path = sources_path();
    if not os.path.exists(path) {
        print(f"[deep_search] source pack missing: {path}");
        return {};
    }
    handle = open(path, "r");
    data = json.loads(handle.read());
    handle.close();
    return data;
}

glob PACK: any = read_pack();

def section(name: str) -> dict[str, str] {
    out: dict[str, str] = {};
    if name in PACK {
        for key in PACK[name] {
            out[str(key)] = str(PACK[name][key]);
        }
    }
    return out;
}

def task_list() -> list[str] {
    out: list[str] = [];
    if "tasks" in PACK {
        for item in PACK["tasks"] {
            out.append(str(item["task"]));
        }
    }
    return out;
}

glob WEB: dict[str, str] = section("web");
glob PAPERS: dict[str, str] = section("papers");
glob METRICS: dict[str, str] = section("metrics");
glob SOURCES: dict[str, str] = section("sources");
glob TASKS: list[str] = task_list();


def pick_task() -> str {
    direct = str(os.environ.get("TASK") or "");
    if direct {
        return direct;
    }
    if TASKS {
        idx = int(os.environ.get("DEEP_TASK_INDEX") or "0") % len(TASKS);
        return TASKS[idx];
    }
    return "Assess whether prefix caching or prefill-decode disaggregation is the better lever for cutting time to first token in multi-step agent workloads.";
}


# ---------------------------------------------------------------------- tools

glob STOPWORDS: set[str] = set(
    "a an and are as at be by for from has have in into is it its of on or that the their to was were with".split()
);

def terms(text: str) -> set[str] {
    words: set[str] = set(text.lower().replace(",", " ").replace(".", " ").split());
    return words.difference(STOPWORDS);
}

def rank(query: str, table: dict[str, str], top: int) -> list[str] {
    scored: list[tuple[int, str]] = [];
    for key in table {
        hits = len(terms(query) & terms(key + " " + table[key]));
        scored.append((hits + 2 * len(terms(query) & terms(key)), key));
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


# ------------------------------------------------------------ supervisor graph

obj ResearchReport {
    has title: str;
    has mainText: str;
    has tldr: list[str];
}

node ResearchSupervisor {
    has charter: str = "run one round at a time: name the open questions, hand them to the sub-agents that own them, and stop once the questions are answered";

    def decompose_task(task: str, web: str, papers: str, data: str, facts: str) -> list[str] by llm(
        temperature=0.0,
        max_tokens=1024,
        max_output_retries=3
    );
}

node WorkerAgent {}

node WebSearchAgent(WorkerAgent) {
    def web_research(task: str, plan: list[str]) -> str by llm(
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
    def paper_research(task: str, plan: list[str]) -> str by llm(
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
    def data_research(task: str, plan: list[str]) -> str by llm(
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
    def fact_check(task: str, plan: list[str]) -> str by llm(
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

    def summarize(task: str, web: str, papers: str, data: str, facts: str) -> str by llm(
        temperature=0.0,
        max_tokens=384
    );

    def build_report(summary: str) -> ResearchReport by llm(
        temperature=0.0,
        max_tokens=384,
        max_output_retries=3
    );

    can setup with Root entry {
        if not [-->][?:ResearchSupervisor] {
            supervisor = ResearchSupervisor();
            here ++> supervisor;
            supervisor ++> WebSearchAgent();
            supervisor ++> PaperSearchAgent();
            supervisor ++> DataAnalysisAgent();
            supervisor ++> FactCheckAgent();
            supervisor ++> supervisor;
        }
        visit [-->][?:ResearchSupervisor];
    }

    can plan with ResearchSupervisor entry {
        self.rounds += 1;
        time.sleep(0.20 * TOOL_SCALE);
        self.plan = here.decompose_task(
            self.task, self.web_findings, self.paper_findings, self.data_findings, self.fact_findings
        );
        # The router picks this round's sub-agents; the walker runs each of them
        # before it reaches the supervisor again.
        visit [-->][?:WorkerAgent] by llm(select=(1, 3), intent=DISPATCH_INTENT, temperature=0.0);
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
sem ResearchSupervisor.decompose_task = "Name the open questions for the next research round as a JSON array of strings, Follow the given format schema, one question per string, each addressed to the kind of agent that can answer it. Drop anything the findings already settle, and return an empty list when nothing is open.";
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

sem FactCheckAgent = "the sub-agent that confirms claims against an authoritative source";
sem FactCheckAgent.fact_check = "Check the plan's contested claims against the source index. Finish with each claim marked CONFIRMED or REFUTED, followed by the source sentence that decides it.";
sem FactCheckAgent.fact_check.task = "the research task being investigated";
sem FactCheckAgent.fact_check.plan = "this round's open questions";

sem DeepResearch.summarize = "Merge every sub-agent's findings into one account of the research task: what is established, what each source contributed, and where the sources disagree.";
sem DeepResearch.summarize.task = "the research task being investigated";
sem DeepResearch.summarize.web = "the web-search agent's findings; empty if it never ran";
sem DeepResearch.summarize.papers = "the paper-search agent's findings; empty if it never ran";
sem DeepResearch.summarize.data = "the data-analysis agent's findings; empty if it never ran";
sem DeepResearch.summarize.facts = "the fact-check agent's findings; empty if it never ran";

sem DeepResearch.build_report = "Turn the merged summary into the final research report, returned as a JSON object with the keys title, mainText and tldr.";
sem DeepResearch.build_report.summary = "the merged account of the research task";

sem ResearchReport = "the final research report";
sem ResearchReport.title = "one line naming what the research settled";
sem ResearchReport.mainText = "the report body, two or three paragraphs";
sem ResearchReport.tldr = "a JSON list of three short takeaway strings";


with entry:__main__ {
    DeepResearch(task=pick_task()) spawn root;
}
