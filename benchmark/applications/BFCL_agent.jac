import os;
import re;
import sys;
import json;
import time;
import hashlib;
import requests;
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B", extra_body={"no_header_last": False});
glob HERE: str = os.path.dirname(os.path.abspath(__file__));
glob DATA: str = os.environ.get("BF_DATA") or os.path.join(HERE, "bench_data", "BFCL", "BFCL_v4_web_search.json");
glob ANSWERS: str = os.environ.get("BF_ANSWERS") or os.path.join(HERE, "bench_data", "BFCL", "BFCL_v4_web_search_answers.json");
glob MAX_STEPS: int = int(os.environ.get("BF_MAX_STEPS") or "8");
glob PAGE_CHARS: int = int(os.environ.get("BF_PAGE_CHARS") or "6000");
glob EXCERPT_CHARS: int = int(os.environ.get("BF_EXCERPT_CHARS") or "300");
glob CACHE_DIR: str = os.environ.get("BF_CACHE_DIR") or os.path.join(HERE, "..", "reproduce", "web_cache");
glob TOOL_DELAY_S: float = float(os.environ.get("BF_TOOL_DELAY_S") or "0");
glob WIKIPEDIA_API: str = "https://en.wikipedia.org/w/api.php";

glob TOOLS: str = "\n".join([
    "search(query): keyword search; returns up to 5 results as title, URL and summary",
    "fetch(title): the plain text of one page, by its exact title, truncated",
    "get_current_time(): the current date and time",
    "answer(text): stop and return `text` as the final answer",
]);


obj AgentTools {
    def get_current_time() -> str {
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime());
    }

    def search(query: str) -> str {
        return self.cached("search", query);
    }

    def fetch(title: str) -> str {
        return self.cached("fetch", title);
    }

    def call(tool: str, argument: str) -> str {
        if tool == "search" {
            return self.search(argument);
        } elif tool == "fetch" {
            return self.fetch(argument);
        } elif tool == "get_current_time" {
            return self.get_current_time();
        }
        return f"Unknown tool: {tool}. Use one of: search, fetch, get_current_time, answer.";
    }

    def cached(kind: str, key: str) -> str {
        os.makedirs(CACHE_DIR, exist_ok=True);
        path = os.path.join(CACHE_DIR, hashlib.md5(f"{kind}:{key}".encode()).hexdigest() + ".txt");
        t0 = time.monotonic();
        if os.path.exists(path) {
            with open(path) as f {
                text = f.read();
            }
        } else {
            text = self.wiki_search(key) if kind == "search" else self.wiki_page(key);
            if not text.startswith(("Search failed", "Fetch failed")) {
                with open(path, "w") as f {
                    f.write(text);
                }
            }
        }
        remaining = TOOL_DELAY_S - (time.monotonic() - t0);
        if remaining > 0 {
            time.sleep(remaining);
        }
        return text;
    }

    def wiki(params: dict) -> dict {
        response = requests.get(
            WIKIPEDIA_API,
            params=dict(params, format="json", formatversion=2),
            headers={"User-Agent": "Jac-BFCL-Agent/1.0"},
            timeout=15
        );
        response.raise_for_status();
        return dict(dict(response.json()).get("query", {}));
    }

    def wiki_search(query: str) -> str {
        try {
            pages = list(self.wiki({
                "action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": 0,
                "gsrlimit": 5, "prop": "extracts|info", "explaintext": 1, "exintro": 1,
                "exlimit": "max", "inprop": "url"
            }).get("pages", []));
            results = [
                f"Title: {p.get('title', '')}\nURL: {p.get('fullurl', '')}\nSummary: {str(p.get('extract', '')).strip()}"
                for p in pages if str(p.get("extract", "")).strip()
            ];
            return "\n\n".join(results) if results else f"No result for: {query}";
        } except Exception as e {
            return f"Search failed for '{query}': {e}";
        }
    }

    def wiki_page(title: str) -> str {
        try {
            pages = list(self.wiki({
                "action": "query", "titles": title, "prop": "extracts", "explaintext": 1, "redirects": 1
            }).get("pages", []));
            text = str(pages[0].get("extract", "")).strip() if pages else "";
            return text[:PAGE_CHARS] if text else f"No page titled: {title}";
        } except Exception as e {
            return f"Fetch failed for '{title}': {e}";
        }
    }
}


obj Step {
    has thought: str = "";
    has tool: str = "";
    has argument: str = "";
    has observation: str = "";
}

sem Step.thought = "Why this tool was called.";
sem Step.tool = "The tool that was called.";
sem Step.argument = "The argument it was called with.";
sem Step.observation = "A short excerpt of what the tool returned; the latest full result is `observation`.";


obj Action {
    has thought: str = "";
    has tool: str = "";
    has argument: str = "";
}

sem Action.thought = "What is known so far and what the next step must find out.";
sem Action.tool = "One of the tool names listed in `tools`.";
sem Action.argument = "The tool's single argument; the final answer itself for `answer`; empty for `get_current_time`.";


walker Task {
    has question: str;
    has gold: list[str] = [];
    has history: list[Step] = [];
    has observation: str = "";
    has answer: str = "";
    has steps: int = 0;
    has errors: list[str] = [];
}

sem Task = "One session answering one multi-hop question with web tools.";
sem Task.history = "Every step taken so far, oldest first, each with an excerpt of its result.";
sem Task.observation = "The full result of the latest tool call.";


node Agent {
    def step(observation: str, tools: str, question: str, history: list[Step]) -> Action by llm(temperature=0.0, max_tokens=1024);

    can act with Task entry {
        try {
            action = self.step(visitor.observation, TOOLS, visitor.question, visitor.history);
        } except Exception as e {
            visitor.errors.append(f"step {visitor.steps + 1} failed: {e}");
            action = Action(thought="The step could not be produced.", tool="answer", argument="");
        }
        visitor.steps += 1;
        visitor.history.append(Step(thought=action.thought, tool=action.tool, argument=action.argument));
        visit [self-->][?:Executor];
    }
}

sem Agent = "Decides the next tool call from the question and the history.";
sem Agent.step = "Answer `question` by searching and reading pages. `observation` is the full result of the previous step; `history` summarizes the earlier ones. Choose the next tool from `tools`; each step should resolve one hop of the question. Call `answer` as soon as the answer is established, with the answer alone as the argument, in the format the question asks for.";
sem Agent.step.observation = "The full result of the latest tool call; empty on the first step.";
sem Agent.step.tools = "The tools available, one per line.";
sem Agent.step.question = "The question to answer.";
sem Agent.step.history = "The steps taken so far with their observations, oldest first.";


node Executor {
    has tools: AgentTools = AgentTools();

    can execute with Task entry {
        step = visitor.history[-1];
        if step.tool == "answer" {
            visitor.answer = step.argument;
        } else {
            visitor.observation = self.tools.call(step.tool, step.argument);
            step.observation = visitor.observation[:EXCERPT_CHARS];
            if visitor.steps < MAX_STEPS{
                visit [self-->][?:Agent];
                return;
            }
            visitor.errors.append(f"gave up after {visitor.steps} steps");
        }
        if not visitor.gold {
            verdict = "UNGRADED";
        } elif matches(visitor.answer, visitor.gold) {
            verdict = "PASSED";
        } else {
            verdict = "FAILED";
        }
        verdict = "UNGRADED" if not visitor.gold else ("PASSED" if matches(visitor.answer, visitor.gold) else "FAILED");
        print(f"Verdict: {verdict}");
        print(f"Answer: {visitor.answer}");
        print(f"Gold: {' | '.join(visitor.gold)}");
        print(f"Rounds: {visitor.steps}, notes: {len(visitor.history)}");
        if visitor.errors {
            print("Diagnostics:");
            for error in visitor.errors {
                print(f"- {error}");
            }
        }
    }
}

sem Executor = "Runs the tool the agent chose and records the observation.";


def normalize(text: str) -> str {
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip();
}

def matches(answer: str, gold: list[str]) -> bool {
    a = normalize(answer);
    for g in gold {
        n = normalize(g);
        if n and (n == a or n in a) {
            return True;
        }
    }
    return False;
}

def load_jsonl(path: str) -> list[dict] {
    rows: list[dict] = [];
    with open(path) as f {
        for line in f {
            if line.strip() {
                rows.append(dict(json.loads(line)));
            }
        }
    }
    return rows;
}
def load_task(task_id: str) -> Task {
    question = "";
    for row in load_jsonl(DATA) {
        if row["id"] == task_id {
            turns: list = list(row["question"]);
            first: list = list(turns[0]);
            question = str(dict(first[0])["content"]);
        }
    }
    if not question {
        raise KeyError(f"{task_id} not in {DATA}");
    }
    gold: list[str] = [];
    if os.path.exists(ANSWERS) {
        for row in load_jsonl(ANSWERS) {
            if row["id"] == task_id {
                gold = [str(g) for g in list(row["ground_truth"])];
            }
        }
    }
    return Task(question=question, gold=gold);
}

def build_graph() -> Agent {
    existing = [root-->][?:Agent];
    if existing {
        return existing[0];
    }
    agent = root ++> Agent();
    executor = Executor();
    agent ++> executor;
    executor ++> agent;
    return agent;
}


with entry {
    task_id = os.environ.get("BF_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "web_search_0");
    load_task(task_id) spawn build_graph();
}
