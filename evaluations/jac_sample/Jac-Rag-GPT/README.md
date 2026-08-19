# Jac-Rag-GPT (Jac-GPT core)

Core of Jac-GPT rebuilt as a minimal Jac project: a RAG-grounded, LLM-routed
multi-agent chatbot that answers questions about the Jac language using the
documentation bundled in `docs/`. No frontend, no dashboard — just the engine.

## Architecture

```
interact walker (message, session_id)
  └─ Root: find/create Session, record user turn
      └─ Router: visit [-->] by llm(select=1)   ← LLM picks the agent
          ├─ RagChat       docs Q&A       (ReAct + search_docs tool)
          ├─ CodingChat    write Jac code (ReAct + search_docs tool)
          ├─ DebuggerChat  fix Jac code   (ReAct + search_docs tool)
          ├─ QAChat        small talk
          └─ OffTopicChat  polite redirect
  └─ Root exit: save assistant turn, report {session_id, agent, response}
```

- `main.jac` — agent nodes, router, session persistence, `interact` /
  `get_session` walkers, CLI REPL.
- `rag_engine.jac` — FAISS vector store over `docs/` (OpenAI embeddings,
  chunking, metadata-enriched chunks) with CrossEncoder reranking.
- `config/faiss_reranking.json` — model + RAG parameters (chunking, k,
  reranker); loaded at startup, falls back to built-in defaults.

## Setup

```bash
export OPENAI_API_KEY=sk-...   # or put it in .env
jac install                    # installs Python deps + byllm capability
```

First run builds the FAISS index from `docs/` (one-time embedding cost);
subsequent runs load it from `faiss_index/`. The CrossEncoder reranker model
downloads from HuggingFace on first use; if unavailable, reranking is
disabled automatically.

## Run

Terminal chat:

```bash
jac run main.jac
```

As a service (REST):

```bash
jac start main.jac
# POST /walker/interact     {"message": "...", "session_id": "s1"}
# POST /walker/get_session  {"session_id": "s1"}
```

`MODEL=gpt-4o jac run main.jac` overrides the LLM from the environment.
