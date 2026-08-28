"""Retrieval-augmented QA with a rewrite / rerank / generate pipeline.

    rewrite_query[0.5B] -> lexical retrieve (tool, top-5) -> rerank[Qwen3-0.6B]
        -> generate_answer[Qwen3-4B]

Every request has the same fixed three-model sequence (E R G), no branch and no
loop, so the sequence is known exactly at compile time: the upper bound for
proactive load and prefill. Passages are an inline encyclopedia slice.
Inputs: $QUESTION or $RAG_INDEX."""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob PORT = int(os.environ.get("SERVE_PORT") or "8964");

glob rewriter = InterceptorLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="rag_qa");
glob reranker = InterceptorLLM(model_name="Qwen/Qwen3-0.6B", comm_ip="localhost", comm_port=PORT, program_name="rag_qa");
glob generator = InterceptorLLM(model_name="Qwen/Qwen3-4B", comm_ip="localhost", comm_port=PORT, program_name="rag_qa");

glob PASSAGES: dict[str, str] = {
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
glob QUESTIONS: list[str] = [
    "Who directed the 1967 film The Graduate?",
    "Which novel is the film Blade Runner based on?",
    "Who composed the score for the 1975 film Jaws?",
    "Which director made both The Shawshank Redemption and The Green Mile?",
    "Which novella was the film Arrival adapted from?",
    "Who wrote the song Mrs. Robinson?",
    "Who directed Rain Man?",
    "Which 1970 war film did Mike Nichols adapt from a Joseph Heller novel?",
    "Who composed the score for Blade Runner?",
    "Which actor played Benjamin Braddock in The Graduate?",
    "Who directed Blade Runner 2049?",
    "Which Stephen King collection contains Rita Hayworth and Shawshank Redemption?"
];
glob ANSWERS: list[str] = [
    "Mike Nichols",
    "Do Androids Dream of Electric Sheep?",
    "John Williams",
    "Frank Darabont",
    "Story of Your Life",
    "Simon & Garfunkel",
    "Barry Levinson",
    "Catch-22",
    "Vangelis",
    "Dustin Hoffman",
    "Denis Villeneuve",
    "Different Seasons"
];

def pick_index() -> int {
    return int(os.environ.get("RAG_INDEX") or "0") % len(QUESTIONS);
}

def pick_question() -> str {
    direct = str(os.environ.get("QUESTION") or "");
    if direct {
        return direct;
    }
    return QUESTIONS[pick_index()];
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

def retrieve(query: str, top: int) -> str {
    time.sleep(0.30 * TOOL_SCALE);
    ranked: list[tuple[int, str]] = [];
    for title in PASSAGES {
        ranked.append((3 * overlap(query, title) + overlap(query, PASSAGES[title]), title));
    }
    ranked.sort(reverse=True);
    lines: list[str] = [];
    for hit in ranked[:top] {
        if hit[0] > 0 {
            lines.append("[" + hit[1] + "] " + PASSAGES[hit[1]]);
        }
    }
    if not lines {
        return "No passage matched the query.";
    }
    return "\n".join(lines);
}

def passage_of(title: str) -> str {
    want = title.strip().strip("[]").lower();
    for key in PASSAGES {
        if key.lower() == want {
            return "[" + key + "] " + PASSAGES[key];
        }
    }
    for key in PASSAGES {
        if want and (want in key.lower() or key.lower() in want) {
            return "[" + key + "] " + PASSAGES[key];
        }
    }
    return "";
}


def rewrite_query(question: str) -> str by rewriter(
    temperature=0.0,
    max_tokens=48
);

def rerank(question: str, candidates: str) -> str by reranker(
    temperature=0.0,
    max_tokens=32
);

def generate_answer(question: str, passage: str) -> str by generator(
    temperature=0.0,
    max_tokens=160
);

sem rewrite_query = "Rewrite the question as a short keyword search query naming the entities it asks about. Return only the query.";
sem rewrite_query.question = "the user's question";
sem rerank = "Pick the one candidate passage that answers the question and return its bracketed title exactly as written, nothing else.";
sem rerank.question = "the user's question";
sem rerank.candidates = "the candidate passages, each starting with its bracketed title";
sem generate_answer = "Answer the question in one sentence using only the passage, and quote the passage phrase that supports the answer.";
sem generate_answer.question = "the user's question";
sem generate_answer.passage = "the passage the answer must come from";


walker RagAgent {
    has question: str = "";
    has query: str = "";
    has candidates: str = "";
    has best: str = "";
    has passage: str = "";
    has answer: str = "";

    can run with Root entry {
        self.query = rewrite_query(self.question);
        self.candidates = retrieve(self.query, 5);
        self.best = rerank(self.question, self.candidates);
        self.passage = passage_of(self.best);
        if not self.passage {
            self.passage = self.candidates.split("\n")[0];
        }
        time.sleep(0.10 * TOOL_SCALE);
        self.answer = generate_answer(self.question, self.passage);
    }

    can report with exit {
        print("QUESTION:", self.question);
        print("QUERY:", self.query);
        print("BEST:", self.best);
        print("ANSWER:", self.answer);
    }
}


with entry:__main__ {
    RagAgent(question=pick_question()) spawn root;
    if not os.environ.get("QUESTION") {
        print("GOLD:", ANSWERS[pick_index()]);
    }
}
