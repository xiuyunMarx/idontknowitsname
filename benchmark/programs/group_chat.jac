# This is a agent to show the structure that LLM decides next step:
# a group chat takes the user query, makes a triage note, routes it with
# visit-by to the matching sub-agent, and each sub-agent answers with its
# own tools (variable-length ReAct loops with realistic tool latency).
# Call chain seen by the server:
#   TriageCenter.make_primary_reply -> visit routing -> <SubAgent>.<ReAct> -> GroupChat.compose_final_answer
import os;
import sys;
import time;
import random;
import from jaclang.byllm.llm { InterceptorLLM }
glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B", call_params={"temperature": 0.0});

# Deterministic per-session randomness: same query -> same tool latency/content.
glob rng = random.Random(0);

node TriageCenter{
    def make_primary_reply(user_query: str) -> str by llm();

    def log_ticket(note: str) -> None{
        # bookkeeping between the triage reply and the routing decision, dummy here
        time.sleep(0.25);
    }
}
sem TriageCenter = "Front desk of the support group chat.";
sem TriageCenter.make_primary_reply = "One short sentence saying what kind of problem this is and what evidence would settle it. No solution.";

obj SearchResult{
    has query: str = "";
    has answer: str = "";
    has evidence: str = "";
}
sem SearchResult.query = "The query that was answered.";
sem SearchResult.answer = "The direct answer, one or two sentences.";
sem SearchResult.evidence = "The tool output line(s) the answer rests on.";

node KnowledgeBase{
    has specialty: str = "internal product knowledge base";

    # this is a tool
    def search_base(query: str) -> str{
        time.sleep(rng.uniform(0.03, 0.08));
        kb = [
            (["pricing", "price", "plan", "cost"],
             "KB-12: The pro plan is $29/seat/month; the free tier allows 3 projects."),
            (["limit", "rate", "quota", "api"],
             "KB-31: API rate limit is 600 requests/min per key; bursts up to 1200 for 10s."),
            (["login", "sso", "cookie", "auth"],
             "KB-07: Login loops are usually stale SSO cookies; clearing the *.example.com cookies fixes it."),
            (["backup", "restore", "retention"],
             "KB-44: Backups run nightly at 02:00 UTC and are kept for 35 days."),
        ];
        q = query.lower();
        hits = [v for (keys, v) in kb if any([k in q for k in keys])];
        return "\n".join(hits) if hits else "No KB entry matched; try a shorter keyword.";
    }
    def AnswerQuery(query: str) -> SearchResult by llm(
        tools = [self.search_base],
        max_react_iterations = 6
    );

    can handle_query with GroupChat entry {
        try {
            result = self.AnswerQuery(visitor.user_query);
            visitor.findings = result.answer;
        } except Exception {
            visitor.findings = "knowledge base lookup failed";
        }
    }
}
sem KnowledgeBase.AnswerQuery = "Answer the query strictly from what search_base returns; finish with a SearchResult carrying query, answer and evidence.";
sem KnowledgeBase.search_base = "Look a keyword up in the internal knowledge base.";
sem KnowledgeBase.handle_query = "answers product/account questions from the internal KB";

node WebResearcher{
    has specialty: str = "open web research";

    # this is a tool: heavy-tailed latency, snippets that invite a follow-up fetch
    def web_search(query: str) -> str{
        time.sleep(min(6.0, rng.lognormvariate(-0.5, 0.9)));
        h = hash(query) % 100;
        return (f"1. \"{query}\" - overview (https://site-{h}.example/a): partial summary, details on page.\n"
                f"2. \"{query}\" benchmarks (https://site-{h}.example/b): numbers reported on page.");
    }
    def fetch_page(url: str) -> str{
        time.sleep(min(6.0, rng.lognormvariate(-0.3, 0.8)));
        h = hash(url) % 97;
        return f"Page {url}: the measured figure is {h * 3 + 40} units, updated this year, methodology in appendix.";
    }
    def Investigate(query: str) -> str by llm(
        tools = [self.web_search, self.fetch_page],
        max_react_iterations = 6
    );

    can handle_query with GroupChat entry {
        visitor.findings = self.Investigate(visitor.user_query);
    }
}
sem WebResearcher.Investigate = "Research the query on the web; search first, fetch a page when the snippet is not enough, then answer in two sentences.";
sem WebResearcher.web_search = "Web search returning titled snippets with URLs.";
sem WebResearcher.fetch_page = "Fetch one URL from a snippet and return the page text.";
sem WebResearcher.handle_query = "researches open-ended or current-events questions on the web";

node DataAnalyst{
    has specialty: str = "calculations and quantitative checks";

    # this is a tool
    def calculate(expression: str) -> str{
        time.sleep(0.01);
        try {
            return str(eval(expression, {"__builtins__": {}}, {}));
        } except Exception as e {
            return f"error: {e}";
        }
    }
    def Analyze(query: str) -> str by llm(
        tools = [self.calculate],
        max_react_iterations = 6
    );

    can handle_query with GroupChat entry {
        visitor.findings = self.Analyze(visitor.user_query);
    }
}
sem DataAnalyst.Analyze = "Answer the quantitative question, using calculate for every piece of arithmetic.";
sem DataAnalyst.calculate = "Evaluate one Python arithmetic expression.";
sem DataAnalyst.handle_query = "handles arithmetic, unit conversions and quantitative comparisons";

walker GroupChat{
    has user_query: str;
    has triage_note: str = "";
    has findings: str = "";

    def compose_final_answer(user_query: str, triage_note: str, findings: str) -> str by llm();

    can triage with TriageCenter entry {
        self.triage_note = here.make_primary_reply(self.user_query);
        here.log_ticket(self.triage_note);
        visit [-->] by llm(
            select=1,
            intent="Send the user query to the sub-agent whose capability can actually answer it"
        );
    }

    can wrap_up with exit {
        if self.findings {
            print(self.compose_final_answer(self.user_query, self.triage_note, self.findings));
        } else {
            print("(no sub-agent produced findings)");
        }
    }
}
sem GroupChat.user_query = "The user question, verbatim.";
sem GroupChat.triage_note = "One-line triage assessment of the query.";
sem GroupChat.findings = "What the chosen sub-agent found.";
sem GroupChat.compose_final_answer = "Write the reply to the user from the findings; one short paragraph, no meta commentary.";

with entry {
    query = os.environ.get("GC_QUERY")
        or (sys.argv[1] if len(sys.argv) > 1 else "What is the API rate limit on the pro plan?");
    rng.seed(query);

    tc = root ++> TriageCenter();
    tc ++> KnowledgeBase();
    tc ++> WebResearcher();
    tc ++> DataAnalyst();
    GroupChat(user_query=query) spawn tc;
}
