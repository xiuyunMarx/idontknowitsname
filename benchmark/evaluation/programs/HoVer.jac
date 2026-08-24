"""HoVer-style many-hop claim verification agent (evaluation tenant).

The workload is one deep dataflow chain over a Wikipedia-like corpus:

    decompose_claim -> plan_query -> retrieve_evidence -> reason_hop
                    -> assess_gap -> (next hop) -> verify_claim -> synthesize_answer

Every stage after the first consumes a previous stage's return through walker
state, so the compiler pass sees a `ret` edge for almost every downstream
parameter. The two retrieval stages run byLLM tool loops (`search_wiki`,
`open_article`, `list_links`), and each tool turn is an idle window on the
serving side: the walker is blocked on a local corpus lookup while the engine
has the successor's prefix available to prefill.

Hop count is fixed by $HOVER_HOPS (default 2), not decided by the model, so the
served callsite sequence is identical across conditions and paired comparisons
stay valid.

Topology note: the static pass links successors inside one ability body, so both
tool loops are covered (`retrieve_evidence` -> `reason_hop`, `verify_claim` ->
`synthesize_answer`). The three hand-offs that cross a plain `visit` —
`decompose_claim` -> `plan_query`, `reason_hop` -> `assess_gap`, `assess_gap` ->
the next hop's `plan_query` — are recovered as dataflow (`ret`) edges but not as
may-run-next edges, so the windows after those calls stay cold.

Run:

    python start_server.py --model Qwen/Qwen2.5-7B-Instruct \
        --program hover:benchmark/evaluation/programs/HoVer.jac:8971

    HOVER_CLAIM_INDEX=0 jac run benchmark/evaluation/programs/HoVer.jac

Inputs: $CLAIM (claim text) or $HOVER_CLAIM_INDEX into $HOVER_CLAIMS.
Corpus: $HOVER_CORPUS, a JSON object of title -> {"text", "links"}. The bundled
`datasets/hover_claims_30.json` holds 30 claims from the HoVer dev release and
`datasets/hover_wiki.json` the Oct-2017 Wikipedia articles they need, gold plus
lexical distractors (see datasets/README.md).
"""

import os;
import json;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob HOPS = int(os.environ.get("HOVER_HOPS") or "2");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=8971,
    program_name="hover"
);


# --------------------------------------------------------------- corpus files
# The corpus is flattened into one string table at load time: retrieval tools
# return text, so nothing downstream needs the nested JSON shape.

def dataset_path(env_var: str, name: str) -> str {
    here = os.path.dirname(os.path.abspath(__file__));
    return str(os.environ.get(env_var) or os.path.join(here, "..", "datasets", name));
}

def read_json(path: str) -> any {
    if not os.path.exists(path) {
        print(f"[hover] dataset missing: {path}");
        return {};
    }
    handle = open(path, "r");
    data = json.loads(handle.read());
    handle.close();
    return data;
}

def load_corpus() -> dict[str, dict[str, str]] {
    out: dict[str, dict[str, str]] = {};
    raw = read_json(dataset_path("HOVER_CORPUS", "hover_wiki.json"));
    for title in raw {
        targets: list[str] = [];
        for target in raw[title]["links"] {
            targets.append(str(target));
        }
        out[str(title)] = {"text": str(raw[title]["text"]), "links": ", ".join(targets)};
    }
    return out;
}

def load_claims() -> list[str] {
    out: list[str] = [];
    raw = read_json(dataset_path("HOVER_CLAIMS", "hover_claims_30.json"));
    for record in raw {
        out.append(str(record["claim"]));
    }
    return out;
}

glob CORPUS: dict[str, dict[str, str]] = load_corpus();
glob CLAIMS: list[str] = load_claims();


def pick_claim() -> str {
    direct = str(os.environ.get("CLAIM") or "");
    if direct {
        return direct;
    }
    if CLAIMS {
        idx = int(os.environ.get("HOVER_CLAIM_INDEX") or "0") % len(CLAIMS);
        return CLAIMS[idx];
    }
    return "The director of the film whose soundtrack features the song \"Mrs. Robinson\" also directed a film adapted from a novel by Joseph Heller.";
}


# ---------------------------------------------------------------------- tools

glob STOPWORDS: set[str] = set(
    "a an and are as at be by for from has have in into is it its of on or that the their to was were with".split()
);

def terms(text: str) -> set[str] {
    words: set[str] = set(text.lower().replace('"', " ").replace("'", " ").replace(",", " ").split());
    return words.difference(STOPWORDS);
}

def overlap(query: str, text: str) -> int {
    return len(terms(query) & terms(text));
}

def resolve_title(title: str) -> str {
    want = title.strip().lower();
    for key in CORPUS {
        if key.lower() == want {
            return key;
        }
    }
    for key in CORPUS {
        if want and (want in key.lower() or key.lower() in want) {
            return key;
        }
    }
    return "";
}

def lead_sentence(text: str) -> str {
    # ". " also splits inside titles like "Mrs. Robinson", so keep taking
    # fragments until the lead reads as a whole sentence.
    lead = "";
    for part in text.split(". ") {
        if lead {
            lead = lead + ". " + part;
        } else {
            lead = part;
        }
        if len(lead) >= 60 {
            break;
        }
    }
    if not lead.endswith(".") {
        lead = lead + ".";
    }
    return lead;
}

def search_wiki(query: str) -> str {
    time.sleep(0.30 * TOOL_SCALE);
    ranked: list[tuple[int, str]] = [];
    for title in CORPUS {
        # A title hit is the strong signal in a many-hop chain: the next hop is
        # usually named outright by the query the planner wrote.
        ranked.append((3 * overlap(query, title) + overlap(query, CORPUS[title]["text"]), title));
    }
    ranked.sort(reverse=True);
    lines: list[str] = [];
    for hit in ranked[:3] {
        if hit[0] > 0 {
            lines.append(hit[1] + ": " + lead_sentence(CORPUS[hit[1]]["text"]));
        }
    }
    if not lines {
        return "No article matched that query. Try other entity names from the claim.";
    }
    return "\n".join(lines);
}

def open_article(title: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    key = resolve_title(title);
    if not key {
        return f"No article titled '{title}'. Use search_wiki to find the exact title.";
    }
    return key + ": " + CORPUS[key]["text"];
}

def list_links(title: str) -> str {
    time.sleep(0.20 * TOOL_SCALE);
    key = resolve_title(title);
    if not key {
        return f"No article titled '{title}'. Use search_wiki to find the exact title.";
    }
    return key + " links to: " + CORPUS[key]["links"];
}

sem search_wiki = "Search the encyclopedia for articles matching a query, returning the best titles with their opening sentence.";
sem search_wiki.query = "entity names or keywords taken from the claim";
sem open_article = "Read the full article body for one exact article title.";
sem open_article.title = "the article title to read, exactly as search_wiki reported it";
sem list_links = "List the articles one article links to, which is how a many-hop chain continues.";
sem list_links.title = "the article title whose outgoing links are needed";


# ---------------------------------------------------------------------- graph

node SearchData {
    def plan_query(claim: str, subclaims: str, gap: str) -> str by llm(
        temperature=0.0,
        max_tokens=96
    );

    def retrieve_evidence(claim: str, query: str) -> str by llm(
        tools=[search_wiki, open_article, list_links],
        temperature=0.0,
        max_tokens=320,
        max_react_iterations=6
    );

    def reason_hop(claim: str, query: str, passages: str, digest: str) -> str by llm(
        temperature=0.0,
        max_tokens=256
    );

    can research with HoverAgent entry {
        visitor.hops += 1;
        time.sleep(0.20 * TOOL_SCALE);
        visitor.query = self.plan_query(visitor.claim, visitor.subclaims, visitor.gap);
        visitor.passages = self.retrieve_evidence(visitor.claim, visitor.query);
        time.sleep(0.20 * TOOL_SCALE);
        visitor.digest = self.reason_hop(visitor.claim, visitor.query, visitor.passages, visitor.digest);
        visitor.notes += [f"hop {visitor.hops} query: {visitor.query}"];
        visit [-->][?:SummarizeData];
    }
}

node SummarizeData {
    def assess_gap(claim: str, subclaims: str, digest: str) -> str by llm(
        temperature=0.0,
        max_tokens=128
    );

    can digest_hop with HoverAgent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.gap = self.assess_gap(visitor.claim, visitor.subclaims, visitor.digest);
        if visitor.hops < HOPS {
            visit [-->][?:SearchData];
        } else {
            visit [-->][?:Verdict];
        }
    }
}

node Verdict {
    has rubric: str = "answer SUPPORTED only when every part of the claim is entailed by the collected evidence; otherwise answer NOT_SUPPORTED";

    def verify_claim(claim: str, subclaims: str, digest: str, rubric: str) -> str by llm(
        tools=[open_article],
        temperature=0.0,
        max_tokens=256,
        max_react_iterations=3
    );

    def synthesize_answer(claim: str, verdict: str, digest: str) -> str by llm(
        temperature=0.0,
        max_tokens=192
    );

    can decide with HoverAgent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.verdict = self.verify_claim(visitor.claim, visitor.subclaims, visitor.digest, self.rubric);
        visitor.answer = self.synthesize_answer(visitor.claim, visitor.verdict, visitor.digest);
    }
}

walker HoverAgent {
    has claim: str = "";
    has subclaims: str = "";
    has gap: str = "";
    has query: str = "";
    has passages: str = "";
    has digest: str = "";
    has verdict: str = "";
    has answer: str = "";
    has hops: int = 0;
    has notes: list[str] = [];

    def decompose_claim(claim: str) -> str by llm(
        temperature=0.0,
        max_tokens=160
    );

    can setup with Root entry {
        if not [-->][?:SearchData] {
            search = SearchData();
            summarize = SummarizeData();
            verdict = Verdict();
            here ++> search;
            search ++> summarize;
            summarize ++> search;
            summarize ++> verdict;
        }
        self.subclaims = self.decompose_claim(self.claim);
        time.sleep(0.20 * TOOL_SCALE);
        visit [-->][?:SearchData];
    }

    can wrap with exit {
        print("CLAIM:", self.claim);
        print("VERDICT:", self.verdict);
        print("ANSWER:", self.answer);
        for note in self.notes {
            print("TRAIL:", note);
        }
    }
}

sem HoverAgent.decompose_claim = "Break a multi-hop claim into the individual facts that must each be checked, one per line, naming the entity each fact is about.";
sem HoverAgent.decompose_claim.claim = "the claim under verification";

sem SearchData.plan_query = "Write the single encyclopedia search query for the next retrieval hop. Ask for the one entity that the remaining gap depends on, and return only the query text.";
sem SearchData.plan_query.claim = "the claim under verification";
sem SearchData.plan_query.subclaims = "the individual facts the claim breaks into";
sem SearchData.plan_query.gap = "what earlier hops could not yet establish; empty on the first hop";

sem SearchData.retrieve_evidence = "Collect the article evidence for one hop. Search for the query, open the most relevant article, and follow its links when the claim needs the next entity in the chain. Finish with the article sentences that bear on the claim, each prefixed by its title.";
sem SearchData.retrieve_evidence.claim = "the claim under verification";
sem SearchData.retrieve_evidence.query = "the search query planned for this hop";

sem SearchData.reason_hop = "State what this hop's passages establish about the claim and which entity the next hop must follow. Keep every fact already carried in the running evidence digest and add this hop's facts to it.";
sem SearchData.reason_hop.claim = "the claim under verification";
sem SearchData.reason_hop.query = "the query this hop retrieved for";
sem SearchData.reason_hop.passages = "the article passages this hop retrieved";
sem SearchData.reason_hop.digest = "the evidence digest carried in from earlier hops; empty on the first hop";

sem SummarizeData.assess_gap = "Name the parts of the claim the evidence still does not establish, and the entity that would settle them. Answer 'nothing missing' when the digest already covers every part.";
sem SummarizeData.assess_gap.claim = "the claim under verification";
sem SummarizeData.assess_gap.subclaims = "the individual facts the claim breaks into";
sem SummarizeData.assess_gap.digest = "the evidence digest collected so far";

sem Verdict.verify_claim = "Decide the claim against the collected evidence, opening any article you still need to confirm. Answer SUPPORTED or NOT_SUPPORTED followed by the one sentence of evidence that decides it.";
sem Verdict.verify_claim.claim = "the claim under verification";
sem Verdict.verify_claim.subclaims = "the individual facts the claim breaks into";
sem Verdict.verify_claim.digest = "the evidence digest collected over all hops";
sem Verdict.verify_claim.rubric = "the standard this verdict must meet";

sem Verdict.synthesize_answer = "Report the verdict for the claim in two sentences: the label, then the evidence chain that produced it, naming each article in hop order.";
sem Verdict.synthesize_answer.claim = "the claim under verification";
sem Verdict.synthesize_answer.verdict = "the verdict and its decisive evidence";
sem Verdict.synthesize_answer.digest = "the evidence digest collected over all hops";


with entry:__main__ {
    HoverAgent(claim=pick_claim()) spawn root;
}
