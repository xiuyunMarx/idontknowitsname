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
}

obj TaskPlan{
    has task: str;
    has query: str;
}

walker Executor{
    has request: str;
    has database: RagStore;
    has collected_data: str = "";
    can setup with Root entry{
        visit [-->][?:PlanAndExecute];
    }
    can handle with PlanAndExecute entry{
        plans:list[TaskPlan] = here.generate_plan(self.request);
        max_retries: int = 3;
        for plan in plans{
            max_retries = 3;
            while (max_retries > 0){
                retrieved_data:str = self.database.search(plan.query); # tool calling, temporal gap exists here
                if here.verify_data(plan, retrieved_data) {
                    self.collected_data += f"Task: {plan.task}, Data: {retrieved_data}\n";
                    break;
                }else{
                    max_retries -= 1;
                }
            }
        }
        summary: str = here.synthesize_report(self.request, self.collected_data);
        print(f"Summary Report:\n{summary}");
    }
}
node PlanAndExecute{
    def generate_plan(request: str) -> list[TaskPlan] by llm();
    def verify_data(original_plan: TaskPlan, retrieved_data: str) -> bool by llm();
    def synthesize_report(request: str, retrieved_data: str) -> str by llm();
}
with entry {
    question:str = sys.argv[1];
    if not [root-->][?:PlanAndExecute] {
        root ++> PlanAndExecute();
    }
    root spawn Executor(request=question, database=RagStore(corpus_dir="Not prepared yet"));
}
