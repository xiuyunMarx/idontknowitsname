"""ReAct tool-loop QA agent over a closed synthetic multi-hop corpus.

Topology this app exercises that evaluations/multi_intent cannot:
  - a LOOP: `reason` re-enters itself through a data-dependent for/break, so the
    call site is its own successor (continuation back-edge) instead of a link in
    a fixed chain;
  - an append-only MUTABLE walker field (`scratchpad`) that every turn both reads
    and extends, so turn i+1's binding is turn i's binding plus a suffix;
  - a VARIABLE, question-dependent number of byLLM calls per request, so the exit
    call (`finalize`) is speculated after every turn but served only once.

Corpus is fictional on purpose: the model cannot answer from parametric memory,
so every hop must come from a real retrieval and the loop actually iterates.
"""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }
import from langchain_community.vectorstores { FAISS }
import from langchain_huggingface { HuggingFaceEmbeddings }
import from sentence_transformers { CrossEncoder }

glob llm = InterceptorLLM(model_name="Qwen/Qwen3-4B");

obj RagStore {
    has corpus_dir: str;
    has embedding_model: str = "BAAI/bge-small-en-v1.5";
    has cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    has top_k: int = 1;
    has fetch_k: int = 4;
    has vectorstore: any = None;
    has reranker: any = None;

    def postinit {
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={"device": "cpu"});
        self.reranker = CrossEncoder(self.cross_encoder_model, device="cpu");
        texts: list = [];
        metas: list = [];
        for name in sorted(os.listdir(self.corpus_dir)) {
            if not name.endswith(".txt") { continue; }
            f = open(os.path.join(self.corpus_dir, name), "r");
            raw = f.read();
            f.close();
            for para in raw.split("\n\n") {
                p = para.strip();
                if p {
                    texts.append(p);
                    metas.append({"source": name});
                }
            }
        }
        self.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metas);
        print(f"[rag] indexed {len(texts)} passages from {self.corpus_dir}");
    }
    
    def search(query: str) -> str {
        time.sleep(0.3);  # simulate tool latency -> the idle window we prefill into
        docs = self.vectorstore.similarity_search(query, k=self.fetch_k);
        if not docs {
            return "No relevant passages found.";
        }
        pairs = [(query, d.page_content) for d in docs];
        scores = self.reranker.predict(pairs);
        scored: list = [];
        for (i, d) in enumerate(docs) {
            scored.append((float(scores[i]), d.page_content));
        }
        scored.sort(key=lambda t: tuple -> float : -t[0]);
        out: list = [];
        for (j, item) in enumerate(scored[:self.top_k]) {
            out.append(f"R{j + 1}. {item[1]}");
        }
        return "\n".join(out);
    }
}

glob rag = RagStore(corpus_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus"));

def search_docs(query: str) -> str {
    return rag.search(query);
}

enum ActionKind { SEARCH = "SEARCH", ANSWER = "ANSWER" }

obj ReactStep {
    has thought: str;
    has action: ActionKind;
    has query: str;
}
sem ReactStep = "one decision of the research analyst's think-search-observe loop";
sem ReactStep.thought = "one or two sentences naming which facts the notes already establish and which single fact is still missing";
sem ReactStep.action = "SEARCH when a required fact is still missing from the notes, ANSWER when every required fact is already present in the notes";
sem ReactStep.query = "the archive query for the single missing fact when the action is SEARCH, and an empty string when the action is ANSWER";

enum Support { GROUNDED = "GROUNDED", PARTIAL = "PARTIAL", UNSUPPORTED = "UNSUPPORTED" }

obj QABrief {
    has answer: str;
    has citation: str;
    has support: Support;
}
sem QABrief = "the final record of a closed-corpus question answering case";
sem QABrief.answer = "the direct answer to the question in one or two sentences";
sem QABrief.citation = "the one sentence from the notes' retrieved passages that supports the answer, quoted verbatim";
sem QABrief.support = "GROUNDED when every required link is stated explicitly in the retrieved passages, PARTIAL when the answer had to be inferred by combining them, UNSUPPORTED when a required link is missing entirely";

walker QA_Agent {
    has question: str;
    has scratchpad: str = "";
    has answer: str = "";
    has citation: str = "";
    has support: str = "";
    has steps_taken: int = 0;
    has searches: int = 0;
    has max_steps: int = 5;

    can start with Root entry {
        visit [-->];
    }
}
sem QA_Agent = "A research walker carrying one closed-corpus question through a think-search-observe loop to a signed-off answer.";
sem QA_Agent.question = "the customer's original question";
sem QA_Agent.scratchpad = "the running investigation notes: every thought, search query and retrieved passage so far";
sem QA_Agent.answer = "the final answer to the question";
sem QA_Agent.citation = "the passage sentence supporting the final answer";
sem QA_Agent.support = "how well the retrieved passages support the final answer";

node Researcher {
    def reason(question: str, notes: str) -> ReactStep by llm(temperature=0.0, max_tokens=320);
    def integrate(notes: str, query: str, passage: str) -> str by llm(temperature=0.0, max_tokens=800);
    def finalize(question: str, notes: str) -> QABrief by llm(temperature=0.0, max_tokens=320);

    can investigate with QA_Agent entry {
        notes = "";
        for i=0 to i<visitor.max_steps by i+=1 {
            step = self.reason(visitor.question, notes);
            visitor.steps_taken = i + 1;
            if step.action == ActionKind.ANSWER {
                break;
            }
            observation = search_docs(step.query);
            visitor.searches = visitor.searches + 1;
            notes = self.integrate(notes, step.query, observation);
        }
        visitor.scratchpad = notes;
        brief = self.finalize(visitor.question, notes);
        visitor.answer = brief.answer;
        visitor.citation = brief.citation;
        visitor.support = str(brief.support);
    }
}
sem Researcher = "The closed-corpus research desk: answers a question by issuing one archive search at a time and reasoning over what comes back, then signing off a cited answer.";
sem Researcher.reason = "You are the research analyst of a closed-corpus question answering desk, working one step at a time in a strict think-search-observe loop. You are given the customer's question and the running notes of every step taken so far; for each earlier step the notes contain the thought you recorded, the search query you issued, and the single passage the archive returned. The archive describes one organization and nothing else, and it is the only source of truth you have: never answer from your own knowledge, never invent an entity, and treat any name you do not find in the notes as unknown rather than guessing at it. Decide the single next action. Set action to SEARCH whenever any fact the question requires is still missing from the notes, and set action to ANSWER only when every fact the question requires is already present in the notes as a retrieved passage. When you choose SEARCH, write a query that targets exactly ONE missing fact, because the archive returns only a single passage per search: name the specific entity you need the passage about and the attribute you need from it, and never fold two hops into one query. If the question asks which building houses the instrument used by a named person's project, that is three separate searches taken in order, first the person to learn their project, then the project to learn its instrument, then the instrument to learn its building, and you must issue only the first of them until its answer is in the notes. Never repeat a query that already appears in the notes; if a search returned nothing useful, rephrase it around a different entity name or a different attribute rather than reissuing it. Record in thought, in one or two sentences, which facts the notes already establish and which single fact you are seeking next, naming that fact concretely. When action is ANSWER, leave query as an empty string.";
sem Researcher.integrate = "You are the record keeper of a closed-corpus question answering desk, maintaining the investigation notes that every later step of the case will read. You are given the notes as they stand, the query that was just issued to the archive, and the single passage the archive returned for it. Produce the updated notes. Reproduce the existing notes first, unchanged and in their original order, then append one new numbered entry for the search just performed. The new entry states the query that was issued, then quotes the returned passage in full and verbatim under the label Passage, and closes with one sentence under the label Learned that names the specific fact this passage contributes to the case, written as a concrete statement such as the Tidewater Survey takes its measurements with the Halcyon spectrometer. Never shorten, paraphrase, summarise or re-order an existing entry, and never drop a passage that is already recorded: a later step has to be able to quote any of these passages word for word, so the notes only ever grow. If the returned passage contains nothing that bears on the case, still record the entry and write under Learned that the search returned nothing relevant, so that a later step does not repeat the same query. Add no commentary of your own, do not answer the customer's question here, and never introduce a fact that is not in one of the recorded passages. Reply with the updated notes directly and do not reason out loud before them. /no_think";
sem Researcher.finalize = "You are the senior editor of the closed-corpus question answering desk writing the final record of the case. You are given the customer's question and the complete investigation notes containing every thought, every search query and every passage the archive returned during the run. Answer the question in one or two sentences using only facts that appear in the retrieved passages recorded in the notes, following the chain of intermediate facts the investigation established in the order it found them; never introduce a fact that is not in a retrieved passage and never fall back on your own knowledge of the world. Choose as the citation the single sentence from those retrieved passages that most directly supports the answer, quoted verbatim and unaltered, including its original wording and punctuation. Grade the support by this rubric: GROUNDED only when every link in the chain the question requires appears in a retrieved passage and the final fact is stated there explicitly; PARTIAL when the answer had to be assembled by combining passages or when one link in the chain is implied rather than stated outright; UNSUPPORTED when a required link is missing from the notes entirely, in which case the answer must say plainly that the archive does not contain the information rather than guessing it.";

with entry:__main__ {
    import json;
    if not [root-->][?:Researcher] {
        root ++> Researcher();
    }
    one: str = str(os.environ.get("QUESTION") or "");
    if one {
        w = root spawn QA_Agent(question=one);
        print("STEPS:", w.steps_taken, "SEARCHES:", w.searches);
        print("NOTES:");
        print(w.scratchpad);
        print("ANSWER:", w.answer);
        print("CITATION:", w.citation);
        print("SUPPORT:", w.support);
    } else {
        qpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "questions.jsonl");
        limit = int(os.environ.get("LIMIT") or "20");
        fh = open(qpath, "r");
        lines = fh.readlines();
        fh.close();
        hit = 0;
        n = 0;
        for line in lines[:limit] {
            rec = json.loads(line);
            w = root spawn QA_Agent(question=rec["question"]);
            ok = rec["answer"].lower() in w.answer.lower();
            if ok { hit = hit + 1; }
            n = n + 1;
            print(f"[case] {rec['id']} hops={rec['hops']} steps={w.steps_taken} searches={w.searches} correct={ok} answer={w.answer[:80]}");
        }
        print(f"[eval] accuracy {hit}/{n}");
    }
}
