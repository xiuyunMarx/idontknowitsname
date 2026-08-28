"""Many-hop claim verification (HoVer style), five models along one chain.

    decompose_claim[1.5B] -> plan_query[1.5B] -> retrieve_evidence[Qwen3-1.7B, tools]
        -> reason_hop[2.5-3B] -> assess_gap[1.5B] -> (next hop | verify_claim[Qwen3-4B, tools])
        -> synthesize_answer[0.5B]

Per-request model sequence with HOPS=2:
    A A B C A  A B C A  D E     (A=2.5-1.5B B=3-1.7B C=2.5-3B D=3-4B E=2.5-0.5B)

Hop count is fixed by $HOVER_HOPS (default 2). Corpus and claims are inline.
Inputs: $CLAIM or $HOVER_CLAIM_INDEX."""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "0");
glob PORT = 8964;

glob planner = InterceptorLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="hover");
glob reader = InterceptorLLM(model_name="Qwen/Qwen3-1.7B", comm_ip="localhost", comm_port=PORT, program_name="hover");
glob reasoner = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="hover");
glob judge = InterceptorLLM(model_name="Qwen/Qwen3-4B", comm_ip="localhost", comm_port=PORT, program_name="hover");
glob writer = InterceptorLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="hover");

glob HOPS = int(os.environ.get("HOVER_HOPS") or "2");

glob TEXT: dict[str, str] = {
    "Mrs. Robinson": "\"Mrs. Robinson\" is a song by the American folk rock duo Simon & Garfunkel, released in 1968. It was written for the 1967 film The Graduate, where it accompanies several of the film's best known sequences. The recording won the Grammy Award for Record of the Year in 1969.",
    "The Graduate": "The Graduate is a 1967 American comedy-drama film directed by Mike Nichols. It stars Dustin Hoffman as Benjamin Braddock and Anne Bancroft as Mrs. Robinson. Its soundtrack is built from songs by Simon & Garfunkel, including \"Mrs. Robinson\". Nichols won the Academy Award for Best Director for the film.",
    "Mike Nichols": "Mike Nichols (1931-2014) was an American film and theatre director. His films include Who's Afraid of Virginia Woolf? (1966), The Graduate (1967), Catch-22 (1970), Silkwood (1983), and Closer (2004).",
    "Catch-22 (film)": "Catch-22 is a 1970 American war film directed by Mike Nichols, adapted by Buck Henry from the 1961 novel Catch-22 by Joseph Heller. It stars Alan Arkin as Captain John Yossarian.",
    "Catch-22 (novel)": "Catch-22 is a satirical war novel by the American author Joseph Heller, first published in 1961. It follows a bombardier stationed on the island of Pianosa during the Second World War.",
    "Joseph Heller": "Joseph Heller (1923-1999) was an American author. He is best known for the 1961 novel Catch-22; his later novels include Something Happened and Good as Gold.",
    "Simon & Garfunkel": "Simon & Garfunkel were an American folk rock duo of the singer-songwriter Paul Simon and the singer Art Garfunkel. Their recordings include \"The Sound of Silence\", \"Mrs. Robinson\", and \"Bridge over Troubled Water\".",
    "Dustin Hoffman": "Dustin Hoffman is an American actor. He played the lead role of Benjamin Braddock in The Graduate (1967), and won the Academy Award for Best Actor for Kramer vs. Kramer (1979) and for Rain Man (1988).",
    "Anne Bancroft": "Anne Bancroft (1931-2005) was an American actress. She played Mrs. Robinson in The Graduate (1967) and won the Academy Award for Best Actress for The Miracle Worker (1962).",
    "Rain Man": "Rain Man is a 1988 American road drama film directed by Barry Levinson. It stars Dustin Hoffman as the autistic savant Raymond Babbitt and Tom Cruise as his brother Charlie. The film won the Academy Award for Best Picture.",
    "Barry Levinson": "Barry Levinson is an American filmmaker. He directed Diner (1982), Rain Man (1988), for which he won the Academy Award for Best Director, Bugsy (1991), and Wag the Dog (1997).",
    "Who's Afraid of Virginia Woolf? (film)": "Who's Afraid of Virginia Woolf? is a 1966 American drama film and the directorial debut of Mike Nichols, adapted from the play by Edward Albee. It stars Elizabeth Taylor and Richard Burton.",
    "Blade Runner": "Blade Runner is a 1982 science fiction film directed by Ridley Scott. It is loosely based on the 1968 novel Do Androids Dream of Electric Sheep? by Philip K. Dick. The score was composed by Vangelis.",
    "Do Androids Dream of Electric Sheep?": "Do Androids Dream of Electric Sheep? is a 1968 science fiction novel by the American writer Philip K. Dick. It was the basis for the 1982 film Blade Runner.",
    "Philip K. Dick": "Philip K. Dick (1928-1982) was an American science fiction writer. His novels include Do Androids Dream of Electric Sheep?, The Man in the High Castle, and A Scanner Darkly.",
    "Ridley Scott": "Ridley Scott is an English filmmaker. He directed Alien (1979), Blade Runner (1982), Thelma & Louise (1991), and Gladiator (2000).",
    "Arrival (film)": "Arrival is a 2016 American science fiction film directed by Denis Villeneuve. It is based on the 1998 novella \"Story of Your Life\" by Ted Chiang, and stars Amy Adams as the linguist Louise Banks.",
    "Story of Your Life": "\"Story of Your Life\" is a 1998 science fiction novella by the American writer Ted Chiang. It won the Nebula Award for Best Novella and was adapted into the 2016 film Arrival.",
    "Ted Chiang": "Ted Chiang is an American science fiction writer. His short fiction includes \"Story of Your Life\", \"Exhalation\", and \"Hell Is the Absence of God\".",
    "Denis Villeneuve": "Denis Villeneuve is a French-Canadian filmmaker. He directed Sicario (2015), Arrival (2016), Blade Runner 2049 (2017), and Dune (2021).",
    "Blade Runner 2049": "Blade Runner 2049 is a 2017 science fiction film directed by Denis Villeneuve and a sequel to the 1982 film Blade Runner. Its score was composed by Hans Zimmer and Benjamin Wallfisch.",
    "Jaws (film)": "Jaws is a 1975 American thriller film directed by Steven Spielberg, based on the 1974 novel by Peter Benchley. Its score was composed by John Williams, who won the Academy Award for Best Original Score for it.",
    "John Williams": "John Williams is an American composer and conductor. He composed the scores for Jaws (1975), Star Wars (1977), E.T. the Extra-Terrestrial (1982), and Schindler's List (1993).",
    "Vangelis": "Vangelis (1943-2022) was a Greek composer of electronic and orchestral music. He wrote the score for Chariots of Fire (1981), which won the Academy Award for Best Original Score, and the score for Blade Runner (1982).",
    "Steven Spielberg": "Steven Spielberg is an American filmmaker. He directed Jaws (1975), Close Encounters of the Third Kind (1977), E.T. the Extra-Terrestrial (1982), and Schindler's List (1993).",
    "The Shawshank Redemption": "The Shawshank Redemption is a 1994 American prison drama written and directed by Frank Darabont. It is adapted from the 1982 Stephen King novella Rita Hayworth and Shawshank Redemption and stars Tim Robbins and Morgan Freeman.",
    "Rita Hayworth and Shawshank Redemption": "Rita Hayworth and Shawshank Redemption is a 1982 novella by Stephen King, published in the collection Different Seasons. It was adapted into the 1994 film The Shawshank Redemption.",
    "The Green Mile (film)": "The Green Mile is a 1999 American fantasy drama film directed by Frank Darabont, adapted from the 1996 Stephen King novel of the same name. Set on the death row of a Southern prison, it stars Tom Hanks and Michael Clarke Duncan.",
    "Frank Darabont": "Frank Darabont is an American filmmaker. He directed The Shawshank Redemption (1994), The Green Mile (1999), and The Mist (2007), and developed the television series The Walking Dead.",
    "Stephen King": "Stephen King is an American author of horror and suspense fiction. His works include Carrie, The Shining, the collection Different Seasons, and the serial novel The Green Mile."
};
glob LINKS: dict[str, str] = {
    "Mrs. Robinson": "Simon & Garfunkel, The Graduate",
    "The Graduate": "Mike Nichols, Dustin Hoffman, Anne Bancroft, Mrs. Robinson, Simon & Garfunkel",
    "Mike Nichols": "Who's Afraid of Virginia Woolf? (film), The Graduate, Catch-22 (film)",
    "Catch-22 (film)": "Mike Nichols, Catch-22 (novel), Joseph Heller",
    "Catch-22 (novel)": "Joseph Heller, Catch-22 (film)",
    "Joseph Heller": "Catch-22 (novel)",
    "Simon & Garfunkel": "Mrs. Robinson",
    "Dustin Hoffman": "The Graduate, Rain Man",
    "Anne Bancroft": "The Graduate",
    "Rain Man": "Barry Levinson, Dustin Hoffman",
    "Barry Levinson": "Rain Man",
    "Who's Afraid of Virginia Woolf? (film)": "Mike Nichols",
    "Blade Runner": "Ridley Scott, Do Androids Dream of Electric Sheep?, Vangelis",
    "Do Androids Dream of Electric Sheep?": "Philip K. Dick, Blade Runner",
    "Philip K. Dick": "Do Androids Dream of Electric Sheep?",
    "Ridley Scott": "Blade Runner",
    "Arrival (film)": "Denis Villeneuve, Story of Your Life, Ted Chiang",
    "Story of Your Life": "Ted Chiang, Arrival (film)",
    "Ted Chiang": "Story of Your Life",
    "Denis Villeneuve": "Arrival (film), Blade Runner 2049",
    "Blade Runner 2049": "Denis Villeneuve, Blade Runner",
    "Jaws (film)": "Steven Spielberg, John Williams",
    "John Williams": "Jaws (film)",
    "Vangelis": "Blade Runner",
    "Steven Spielberg": "Jaws (film)",
    "The Shawshank Redemption": "Frank Darabont, Rita Hayworth and Shawshank Redemption, Stephen King",
    "Rita Hayworth and Shawshank Redemption": "Stephen King, The Shawshank Redemption",
    "The Green Mile (film)": "Frank Darabont, Stephen King",
    "Frank Darabont": "The Shawshank Redemption, The Green Mile (film)",
    "Stephen King": "Rita Hayworth and Shawshank Redemption, The Green Mile (film)"
};
glob CLAIMS: list[str] = [
    "The director of the film whose soundtrack features the song \"Mrs. Robinson\" also directed a film adapted from a novel by Joseph Heller.",
    "The author of the novel that the film Blade Runner is based on also wrote the story that the film Arrival is based on.",
    "The 1994 prison drama adapted from a Stephen King novella and the 1999 death row drama adapted from a Stephen King novel were directed by the same person.",
    "The composer who scored the 1975 film Jaws also composed the score for the 1982 film Blade Runner.",
    "The actor who played the lead role in the 1967 film directed by Mike Nichols later starred in a film directed by Barry Levinson."
];
glob LABELS: list[str] = [
    "SUPPORTED",
    "NOT_SUPPORTED",
    "SUPPORTED",
    "NOT_SUPPORTED",
    "SUPPORTED"
];

def pick_index() -> int {
    return int(os.environ.get("HOVER_CLAIM_INDEX") or "0") % len(CLAIMS);
}

def pick_claim() -> str {
    direct = str(os.environ.get("CLAIM") or "");
    if direct {
        return direct;
    }
    return CLAIMS[pick_index()];
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

def resolve_title(title: str) -> str {
    want = title.strip().lower();
    for key in TEXT {
        if key.lower() == want {
            return key;
        }
    }
    for key in TEXT {
        if want and (want in key.lower() or key.lower() in want) {
            return key;
        }
    }
    return "";
}

def lead_sentence(text: str) -> str {
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
    for title in TEXT {
        ranked.append((3 * overlap(query, title) + overlap(query, TEXT[title]), title));
    }
    ranked.sort(reverse=True);
    lines: list[str] = [];
    for hit in ranked[:3] {
        if hit[0] > 0 {
            lines.append(hit[1] + ": " + lead_sentence(TEXT[hit[1]]));
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
    return key + ": " + TEXT[key];
}

def list_links(title: str) -> str {
    time.sleep(0.20 * TOOL_SCALE);
    key = resolve_title(title);
    if not key {
        return f"No article titled '{title}'. Use search_wiki to find the exact title.";
    }
    return key + " links to: " + LINKS[key];
}

sem search_wiki = "Search the encyclopedia for articles matching a query, returning the best titles with their opening sentence.";
sem search_wiki.query = "entity names or keywords taken from the claim";
sem open_article = "Read the full article body for one exact article title.";
sem open_article.title = "the article title to read, exactly as search_wiki reported it";
sem list_links = "List the articles one article links to, which is how a many-hop chain continues.";
sem list_links.title = "the article title whose outgoing links are needed";


node SearchData {
    def plan_query(claim: str, subclaims: str, gap: str) -> str by planner(
        temperature=0.0,
        max_tokens=96
    );

    def retrieve_evidence(claim: str, query: str) -> str by reader(
        tools=[search_wiki, open_article, list_links],
        temperature=0.0,
        max_tokens=320,
        max_react_iterations=6
    );

    def reason_hop(claim: str, query: str, passages: str, digest: str) -> str by reasoner(
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
    def assess_gap(claim: str, subclaims: str, digest: str) -> str by planner(
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

    def verify_claim(claim: str, subclaims: str, digest: str, rubric: str) -> str by judge(
        tools=[open_article],
        temperature=0.0,
        max_tokens=256,
        max_react_iterations=3
    );

    def synthesize_answer(claim: str, verdict: str, digest: str) -> str by writer(
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

    def decompose_claim(claim: str) -> str by planner(
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
    if not os.environ.get("CLAIM") {
        print("GOLD:", LABELS[pick_index()]);
    }
}
