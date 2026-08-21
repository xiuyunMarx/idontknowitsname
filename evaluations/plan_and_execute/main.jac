"""Plan-and-Execute research agent over the closed synthetic Aldermere corpus.

Topology this app exercises that chain / multi_intent / QA_agent cannot:
  - a runtime-determined FAN-OUT: `generate_plan` decides how many execution
    steps there are and what each one asks for, so the executor call site is
    served N times per request with N unknown at compile time;
  - a LONG-LIVED binding: the plan text is produced ONCE at the entry call and
    then consumed at distance 1..N+1 by every verify/execute call and by the
    final synthesis, so one ret recording pays off on every later call;
  - an inner RETRY loop with a data-dependent break (verify -> re-search),
    so the same site re-enters itself an unpredictable number of times;
  - an append-only ret-carried dossier: `execute_task` reads the dossier and
    returns it extended, so turn i+1's binding is turn i's binding plus a
    suffix (append-only prefix = APC-friendly by construction).

Provenance the compiler should recover (all `ret`, none `unknown`, for the
bindings that matter):
    verify_data.plan       <- ret(PlanAndExecute.generate_plan)   dist 1..N
    execute_task.plan      <- ret(PlanAndExecute.generate_plan)   dist 1..N
    execute_task.dossier   <- ret(PlanAndExecute.execute_task)    self edge
    synthesize_report.plan <- ret(PlanAndExecute.generate_plan)   dist N+1
    synthesize_report.dossier <- ret(PlanAndExecute.execute_task) last hop

The 0.3s archive search in front of every verify call is the idle window the
speculative prefill lands in.  Corpus is shared with evaluations/QA_agent so
that corpus difficulty is held constant across the two agent topologies.
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
    has top_k: int = 2;
    has fetch_k: int = 6;
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

glob rag = RagStore(corpus_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "QA_agent", "data", "corpus"));

obj TaskPlan {
    has task: str;
    has query: str;
}

# Deterministically turn the planner's `task | query` lines into steps. Kept
# non-LLM on purpose: the plan TEXT stays the ret-carried binding that every
# later call consumes, while control flow reads a parsed view of it.
def parse_plan(text: str, max_tasks: int) -> list[TaskPlan] {
    out: list = [];
    for raw in text.split("\n") {
        line = raw.strip();
        if not line { continue; }
        head = line.lstrip("0123456789.)-* \t");
        if "|" not in head { continue; }
        parts = head.split("|", 1);
        task = parts[0].strip();
        query = parts[1].strip();
        if not task { continue; }
        if not query { query = task; }
        out.append(TaskPlan(task=task, query=query));
        if len(out) >= max_tasks { break; }
    }
    if not out {
        for raw in text.split("\n") {
            line = raw.strip().lstrip("0123456789.)-* \t");
            if not line { continue; }
            out.append(TaskPlan(task=line, query=line));
            if len(out) >= max_tasks { break; }
        }
    }
    if not out {
        stub = text.strip()[:200];
        out.append(TaskPlan(task=stub, query=stub));
    }
    return out;
}

# Display/scoring helper only. NEVER applied to a value that is passed to a byLLM
# call: the server records the RAW return value as the binding, so cleaning a value
# before handing it downstream would make the fed prefix diverge from the served one.
def strip_think(text: str) -> str {
    out = text;
    while "<think>" in out and "</think>" in out {
        a = out.index("<think>");
        b = out.index("</think>");
        if b < a { break; }
        out = out[:a] + out[b + 8:];
    }
    return out.replace("</think>", "").strip();
}

enum Confidence { COMPLETE = "COMPLETE", PARTIAL = "PARTIAL", INSUFFICIENT = "INSUFFICIENT" }

obj Report {
    has summary: str;
    has evidence: str;
    has confidence: Confidence;
}
sem Report = "the final briefing produced from a completed plan-and-execute run";
sem Report.summary = "the briefing answering the request in three to five sentences, covering every step of the plan in the plan's own order";
sem Report.evidence = "the single sentence from the dossier's recorded passages that carries the most load in the briefing, quoted verbatim";
sem Report.confidence = "COMPLETE when every step of the plan produced a usable passage, PARTIAL when at least one step is missing but the briefing still holds together, INSUFFICIENT when the dossier does not support a briefing at all";

walker Executor {
    has request: str;
    has database: RagStore;
    has plan_text: str = "";
    has dossier: str = "";
    has summary: str = "";
    has evidence: str = "";
    has confidence: str = "";
    has n_tasks: int = 0;
    has n_searches: int = 0;
    has max_tasks: int = 4;
    has max_retries: int = 2;

    can setup with Root entry {
        visit [-->][?:PlanAndExecute];
    }

    can handle with PlanAndExecute entry {
        plan_text = here.generate_plan(self.request);
        self.plan_text = plan_text;
        tasks = parse_plan(plan_text, self.max_tasks);
        self.n_tasks = len(tasks);
        dossier = "";
        for t in tasks {
            # the plan is written before any entity is known, so the executor has to
            # resolve this step's query against what the dossier established so far
            resolved = here.resolve_query(plan_text, dossier, t.task);
            variants = [resolved, t.query, t.task];
            for k=0 to k<self.max_retries by k+=1 {
                passage = self.database.search(variants[k]);  # tool call: the temporal gap we prefill into
                self.n_searches = self.n_searches + 1;
                if here.verify_data(plan_text, t.task, variants[k], passage) {
                    dossier = here.execute_task(plan_text, dossier, t.task, passage);
                    break;
                }
            }
        }
        self.dossier = dossier;
        report_out = here.synthesize_report(plan_text, dossier, self.request);
        self.summary = report_out.summary;
        self.evidence = report_out.evidence;
        self.confidence = str(report_out.confidence);
    }
}
sem Executor = "A research walker that carries one briefing request through a written plan, a step by step archive investigation, and a final synthesis.";
sem Executor.request = "the briefing request as the customer wrote it";
sem Executor.database = "the archive the walker searches";
sem Executor.plan_text = "the written plan the run is following";
sem Executor.dossier = "the accumulated evidence dossier of every completed step";
sem Executor.summary = "the final briefing";
sem Executor.evidence = "the quoted sentence supporting the briefing";
sem Executor.confidence = "how completely the plan was carried out";

node PlanAndExecute {
    def generate_plan(request: str) -> str by llm(temperature=0.0, max_tokens=400);
    def resolve_query(plan: str, dossier: str, task: str) -> str by llm(temperature=0.0, max_tokens=120);
    def verify_data(plan: str, task: str, query: str, passage: str) -> bool by llm(temperature=0.0, max_tokens=64);
    def execute_task(plan: str, dossier: str, task: str, passage: str) -> str by llm(temperature=0.0, max_tokens=1000);
    def synthesize_report(plan: str, dossier: str, request: str) -> Report by llm(temperature=0.0, max_tokens=500);
}
sem PlanAndExecute = "The briefing desk: writes a research plan for a request, works the plan one step at a time against a closed archive, and synthesises the result.";
sem PlanAndExecute.generate_plan = "You are the planner of a closed-corpus briefing desk. You are given a customer's briefing request. Write the plan the desk will follow to answer it. The archive you are planning against describes one research institute and nothing else, it returns exactly ONE passage per search, and each of its passages describes a single entity, so every step of your plan must ask for exactly one fact about exactly one named entity and must never fold two facts into one step. Order the steps so that a step whose subject is only discovered by an earlier step comes after that earlier step: if the request asks which building houses the instrument used by a named person's project, the plan is first the person's project, then that project's instrument, then that instrument's building. Write exactly four steps and nothing else, one step per line, each line in the exact form: the goal of the step, then a space, then a vertical bar character, then a space, then the archive query for that step. The goal names the fact being sought in a few words, and the query names the entity and the attribute in the plain wording the archive is likely to use. Emit no heading, no numbering, no blank lines, no commentary before or after the four lines, and never answer the request yourself. /no_think";
sem PlanAndExecute.resolve_query = "You are the executor of a closed-corpus briefing desk, turning one step of a plan into the query that will actually be sent to the archive. You are given the plan the desk is following, the evidence dossier of the steps already completed, and the single step to be worked now. The plan was written before anything was known, so a later step often refers to an entity only by description, such as the instrument used by the project, and it is your job to replace that description with the concrete name the dossier has since established: if the dossier records that the Tidewater Survey takes its measurements with the Halcyon spectrometer and the step asks for the building housing that instrument, the query must name the Halcyon spectrometer and not the survey. Ask for exactly one attribute of exactly one named entity, in the plain wording the archive is likely to use, and never fold two hops into one query. If the dossier does not yet name the entity the step needs, fall back to the wording of the step itself rather than inventing a name. Reply with the query text on a single line and nothing else: no quotation marks, no label, no explanation, and never answer the step yourself. /no_think";
sem PlanAndExecute.verify_data = "You are the acceptance check of a closed-corpus briefing desk. You are given the plan the desk is following, the single step being worked right now, the query that was issued to the archive for it, and the one passage the archive returned. Answer whether that passage actually supplies the fact this step asks for. Answer True only when the passage names the step's subject entity and states the attribute the step is seeking; answer False when the passage is about a different entity, when it is about the right entity but does not state the attribute, or when the archive returned nothing relevant. Judge only the step you were given and ignore whether the other steps of the plan are satisfied, and never accept a passage because it looks generally related to the request.";
sem PlanAndExecute.execute_task = "You are the record keeper of a closed-corpus briefing desk, maintaining the evidence dossier that the final synthesis will read. You are given the plan the desk is following, the dossier as it stands, the single step that was just completed, and the passage the archive returned for it. Produce the updated dossier. The plan is context only and is never part of the dossier: if the dossier you are given is empty then the updated dossier is the new entry alone, and you must not copy the plan or any of its lines into it. Reproduce the existing dossier first, unchanged, complete and in its original order, then append one new entry for the step just completed. The new entry states the step's goal on its own line, then quotes the returned passage in full and verbatim under the label Passage, and closes with one sentence under the label Learned that states the concrete fact this passage contributes, written as a plain statement such as the Tidewater Survey takes its measurements with the Halcyon spectrometer. Never shorten, paraphrase, summarise, re-order or drop an existing entry: the synthesis has to be able to quote any recorded passage word for word, so the dossier only ever grows. Add no commentary of your own, do not attempt the customer's briefing here, and never introduce a fact that is not in one of the recorded passages. Reply with the updated dossier directly and do not reason out loud before it. /no_think";
sem PlanAndExecute.synthesize_report = "You are the senior editor of the closed-corpus briefing desk writing the deliverable. You are given the plan that was followed, the complete evidence dossier of every step that was carried out, and the customer's original request. Write the briefing using only facts that appear in the passages recorded in the dossier, walking through the plan in its own order so that each step's finding is stated and the chain between them is visible; never introduce a fact that is not in a recorded passage and never fall back on your own knowledge of the world. If a step of the plan has no entry in the dossier, say plainly in the briefing that the archive did not supply that fact rather than guessing it. Choose as the evidence the single sentence from the dossier's recorded passages that carries the most load in the briefing, quoted verbatim and unaltered including its original wording and punctuation. Grade confidence as COMPLETE when every step of the plan has a dossier entry that states its fact, PARTIAL when at least one step is missing or its fact had to be inferred by combining other entries, and INSUFFICIENT when so little was recorded that the briefing cannot be written.";

with entry:__main__ {
    import json;
    if not [root-->][?:PlanAndExecute] {
        root ++> PlanAndExecute();
    }
    one: str = str(os.environ.get("REQUEST") or "");
    if one {
        w = root spawn Executor(request=one, database=rag);
        print("PLAN:");
        print(strip_think(w.plan_text));
        print("TASKS:", w.n_tasks, "SEARCHES:", w.n_searches);
        print("DOSSIER:");
        print(strip_think(w.dossier));
        print("SUMMARY:", w.summary);
        print("EVIDENCE:", w.evidence);
        print("CONFIDENCE:", w.confidence);
    } else {
        rpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "requests.jsonl");
        limit = int(os.environ.get("LIMIT") or "12");
        fh = open(rpath, "r");
        lines = fh.readlines();
        fh.close();
        hit = 0;
        tot = 0;
        for line in lines[:limit] {
            rec = json.loads(line);
            w = root spawn Executor(request=rec["request"], database=rag);
            blob = (w.summary + " " + w.evidence + " " + w.dossier).lower();
            found = 0;
            for f in rec["facts"] {
                if f.lower() in blob { found = found + 1; }
            }
            hit = hit + found;
            tot = tot + len(rec["facts"]);
            print(f"[case] {rec['id']} tasks={w.n_tasks} searches={w.n_searches} facts={found}/{len(rec['facts'])} conf={w.confidence}");
        }
        print(f"[eval] fact coverage {hit}/{tot}");
    }
}
