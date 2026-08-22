import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }
import from langchain_community.vectorstores { FAISS }
import from langchain_huggingface { HuggingFaceEmbeddings }
import from sentence_transformers { CrossEncoder }
import sys;
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

    def overview(query: str) -> str {
        # orientation retrieval for the request as a whole, kept distinct from the
        # per-subtask search so the two do not share one value identity
        return self.search(query);
    }
}

obj TaskPlan{
    has task: str;
    has query: str;
    has acceptance: str;
}

walker Executor{
    has request: str;
    has database: RagStore;
    has collected_data: str = "";
    can setup with Root entry{
        visit [-->][?:PlanAndExecute];
    }
    can handle with PlanAndExecute entry{
        request = self.request;
        notes: str = "";
        plans:list[TaskPlan] = here.generate_plan(request);
        max_retries: int = 3;
        for plan in plans{
            max_retries = 3;
            while (max_retries > 0){
                retrieved_data:str = self.database.search(plan.query); # tool calling, temporal gap exists here
                if here.verify_data(plan, retrieved_data) {
                    notes = here.integrate(plan, notes, retrieved_data); # the record grows through a call, so it is a value the runtime can follow
                    break;
                }else{
                    max_retries -= 1;
                }
            }
        }
        background: str = self.database.overview(request); # last tool gap: everything else the report needs is already known when it opens
        summary: str = here.synthesize_report(request, plans, notes, background);
        self.collected_data = notes;
        print(f"Summary Report:\n{summary}");
    }
}
node PlanAndExecute{
    def generate_plan(request: str) -> list[TaskPlan] by llm();
    def verify_data(original_plan: TaskPlan, retrieved_data: str) -> bool by llm();
    def integrate(original_plan: TaskPlan, notes: str, retrieved_data: str) -> str by llm();
    def synthesize_report(request: str, plans: list[TaskPlan], notes: str, background: str) -> str by llm();
}

sem TaskPlan = "one retrieval subtask of a decomposed research request";
sem TaskPlan.task = "what this subtask is meant to establish, in one short sentence";
sem TaskPlan.query = "the single archive query that retrieves the passage for this subtask, naming the entity in full and in the wording the archive itself would use";
sem TaskPlan.acceptance = "the concrete condition the retrieved passage must satisfy for this subtask to count as answered, written as one checkable statement";

sem PlanAndExecute.generate_plan = "You are the planner of a closed-corpus research desk. You are given one oversight request that asks for several distinct facts about the same subject. Break it into the smallest set of independent retrieval subtasks that together cover the request, one subtask per fact the request names, in the order the request names them. The archive returns a single passage per query, so each subtask must target exactly one entity and one attribute of it and must never fold two hops into one query. Write each query the way the archive's own passages are worded, naming the entity in full rather than by pronoun or abbreviation, and state for each subtask the condition a passage must meet to count as an answer. Do not answer the request here, do not merge two of its points into one subtask, and do not invent subtasks the request did not ask for.";
sem PlanAndExecute.generate_plan.request = "the oversight request to be decomposed into retrieval subtasks";

sem PlanAndExecute.verify_data = "You are the verifier of a closed-corpus research desk. You are given one retrieval subtask and the single passage the archive returned for its query. Decide whether that passage actually satisfies the subtask's acceptance condition. Answer true only when the passage states the required fact explicitly and states it about the entity the subtask names; answer false when the passage is about a different entity, when it merely implies the fact or leaves it to be inferred, or when the fact is absent altogether. Judge the passage in front of you and nothing else: never draw on your own knowledge of the world, never credit a passage because it is topically related to the subtask, and never accept a passage on the grounds that a later search would probably find the missing part.";
sem PlanAndExecute.verify_data.original_plan = "the retrieval subtask being checked, including the condition its passage must satisfy";
sem PlanAndExecute.verify_data.retrieved_data = "the passage the archive returned for this subtask's query";

sem PlanAndExecute.integrate = "You are the record keeper of a closed-corpus research desk, maintaining the case record every later step will read. You are given the record as it stands, the subtask that was just answered, and the passage the archive returned for it. Produce the updated record. Reproduce the existing record first, unchanged, in its original order and word for word, then append one new entry for the subtask just answered. The new entry states the subtask on one line under the label Task, then quotes the returned passage in full and verbatim under the label Passage. Never shorten, paraphrase, summarise or re-order an entry that is already in the record, and never drop a passage: the editor who writes the final briefing has to be able to quote any of these passages word for word, so the record only ever grows. Add no commentary of your own, do not answer the request here, and reply with the updated record directly. /no_think";
sem PlanAndExecute.integrate.notes = "the case record as it stands, to be reproduced verbatim before the new entry";
sem PlanAndExecute.integrate.original_plan = "the subtask that was just answered";
sem PlanAndExecute.integrate.retrieved_data = "the passage the archive returned for that subtask, to be quoted in full";

sem PlanAndExecute.synthesize_report = "You are the senior editor of a closed-corpus research desk writing the final oversight briefing. You are given the original request, the subtasks it was decomposed into, the case record naming every subtask that was answered together with the archive passage accepted for it, and one further passage retrieved for the request as a whole. Write the briefing as continuous prose that answers the request point by point in the order the request asked, grounding every statement in what the collected passages say and never introducing a fact that is not in them. Where the record leaves one of the request's points unanswered, say so plainly instead of filling the gap from your own knowledge. Do not reproduce the record as a list, do not describe your own process, and do not close with a summary of what you have just written.";
sem PlanAndExecute.synthesize_report.request = "the original oversight request the briefing must answer";
sem PlanAndExecute.synthesize_report.plans = "the subtasks the request was decomposed into, in the order the briefing must answer them";
sem PlanAndExecute.synthesize_report.notes = "the case record: every answered subtask with the archive passage accepted for it, in the order they were answered";
sem PlanAndExecute.synthesize_report.background = "one further passage retrieved for the request as a whole, for orientation only";
glob CORPUS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus");

with entry {
    question: str = os.environ.get("QUESTION", "Prepare an oversight briefing on the Tidewater Survey covering: its scope and current status, who leads it and what their background is, which instrument it depends on and what that instrument's known failure modes are, which program funds it and under what conditions, and what the archive retention rules mean for its data.");
    if len(sys.argv) > 1 { question = sys.argv[1]; }
    if not [root-->][?:PlanAndExecute] {
        root ++> PlanAndExecute();
    }
    root spawn Executor(request=question, database=RagStore(corpus_dir=CORPUS_DIR));
}
