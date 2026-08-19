"""RAG engine for Jac-GPT core: FAISS vector search over the bundled docs
with CrossEncoder reranking. Degrades gracefully (search disabled) when the
embedding API key or the reranker model is unavailable."""

import os;
import json;
import time;
import from typing { cast }
import from langchain_community.document_loaders { PyPDFDirectoryLoader, PyPDFLoader, DirectoryLoader }
import from langchain_community.document_loaders.text { TextLoader }
import from langchain_text_splitters { RecursiveCharacterTextSplitter }
import from langchain_huggingface { HuggingFaceEmbeddings }
import from langchain_community.vectorstores { FAISS }
import from sentence_transformers { CrossEncoder }


"""Typed application configuration; defaults mirror config/faiss_reranking.json."""
obj AppConfig {
    has model_name: str = "gpt-4.1-mini";
    has chunk_size: int = 800;
    has chunk_overlap: int = 100;
    has chunk_k: int = 30;
    has rerank_top_n: int = 7;
    has cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2";
    has embedding_model: str = "BAAI/bge-small-en-v1.5";
    has docs_path: str = "docs";
    has faiss_path: str = "faiss_index";
}


"""Load config/<config_name>.json into an AppConfig, falling back to defaults."""
def load_config(config_name: str) -> AppConfig {
    cfg = AppConfig();
    config_path = os.path.join(os.path.dirname(__file__), "config", f"{config_name}.json");
    try {
        with open(config_path, 'r', encoding='utf-8') as f {
            raw = cast(dict[str, any], json.load(f));
        }
        llm_cfg = cast(dict[str, any], raw.get("llm", {}));
        rag_cfg = cast(dict[str, any], raw.get("rag_engine", {}));
        search_cfg = cast(dict[str, any], rag_cfg.get("similarity_search", {}));
        rerank_cfg = cast(dict[str, any], rag_cfg.get("reranking", {}));
        paths_cfg = cast(dict[str, any], raw.get("paths", {}));

        cfg.model_name = cast(str, llm_cfg.get("model_name", cfg.model_name));
        cfg.chunk_size = cast(int, rag_cfg.get("chunk_size", cfg.chunk_size));
        cfg.chunk_overlap = cast(int, rag_cfg.get("chunk_overlap", cfg.chunk_overlap));
        cfg.chunk_k = cast(int, search_cfg.get("k", cfg.chunk_k));
        cfg.rerank_top_n = cast(int, rerank_cfg.get("top_n", cfg.rerank_top_n));
        cfg.cross_encoder_model = cast(str, rerank_cfg.get("model", cfg.cross_encoder_model));
        cfg.embedding_model = cast(str, rag_cfg.get("embedding_model", cfg.embedding_model));
        cfg.docs_path = cast(str, paths_cfg.get("file_path", cfg.docs_path));
        cfg.faiss_path = cast(str, paths_cfg.get("faiss_path", cfg.faiss_path));
    } except Exception as e {
        print(f"[config] Could not load {config_path} ({e}); using built-in defaults.");
    }
    return cfg;
}


obj RagEngine {
    has file_path: str;
    has faiss_path: str;
    has chunk_size: int;
    has chunk_overlap: int;
    has chunk_nos: int;
    has cross_encoder_model: str;
    has embedding_model: str;
    has top_n: int;
    # Runtime handles to third-party objects (FAISS store, CrossEncoder);
    # typed `any` since these libraries expose no Jac type info.
    has vectorstore: any = None;
    has cross_encoder: any = None;
    has use_reranking: bool = True;

    def postinit {
        try {
            self.cross_encoder = CrossEncoder(self.cross_encoder_model);
            print(f"CrossEncoder reranking enabled with model: {self.cross_encoder_model}");
        } except Exception as e {
            print(f"Warning: Failed to load CrossEncoder model: {e}");
            print("Reranking will be disabled.");
            self.use_reranking = False;
        }
        print("RAG Engine initialized");
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}, Chunk Nos: {self.chunk_nos}");

        if not os.path.exists(self.file_path) {
            os.makedirs(self.file_path);
        }

        embeddings = self.get_embedding_function();
        needs_build = True;
        if embeddings is None {
            needs_build = False;
        } elif os.path.exists(self.faiss_path) and os.path.isdir(self.faiss_path) {
            try {
                print(f"Loading existing FAISS index from {self.faiss_path}");
                self.vectorstore = FAISS.load_local(
                    self.faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                );
                needs_build = False;
            } except Exception as e {
                # Most likely a dimension mismatch after changing embedding_model.
                print(f"Could not load index at {self.faiss_path} ({e}); rebuilding from docs.");
            }
        }
        if needs_build {
            print("Creating new FAISS index");
            documents = self.load_documents();
            chunks = self.split_documents(documents);
            self.create_faiss_index(chunks);
        }
    }

    def load_documents -> list {
        documents: list = [];

        try {
            pdf_loader = PyPDFDirectoryLoader(self.file_path);
            pdf_docs = pdf_loader.load();
            documents.extend(pdf_docs);
            print(f"Loaded {len(pdf_docs)} PDF documents");
        } except Exception as e {
            print(f"No PDF files found or error loading PDFs: {e}");
        }

        try {
            md_loader = DirectoryLoader(
                self.file_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            );
            md_docs = md_loader.load();
            documents.extend(md_docs);
            print(f"Loaded {len(md_docs)} Markdown documents");
        } except Exception as e {
            print(f"No Markdown files found or error loading Markdown: {e}");
        }

        print(f"Total documents loaded: {len(documents)}");
        return documents;
    }

    def load_document(file_path: str) -> list {
        if file_path.endswith('.pdf') {
            pdf_loader = PyPDFLoader(file_path);
            return pdf_loader.load();
        } elif file_path.endswith('.md') {
            text_loader = TextLoader(file_path, encoding="utf-8");
            return text_loader.load();
        }
        raise ValueError(f"Unsupported file type: {file_path}");
    }

    def add_file(file_path: str) -> None {
        documents = self.load_document(file_path);
        chunks = self.split_documents(documents);
        self.add_to_faiss(chunks);
    }

    def split_documents(documents: list) -> list {
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        );
        return text_splitter.split_documents(documents);
    }

    """Local sentence-transformers embeddings; building and searching needs no API key."""
    def get_embedding_function -> any {
        try {
            return HuggingFaceEmbeddings(model_name=self.embedding_model);
        } except Exception as e {
            print(f"Failed to load embedding model {self.embedding_model}: {e}. RAG search disabled.");
            return None;
        }
    }

    """Content string enriched with metadata for better semantic matching."""
    def enrich_content_with_metadata(chunk: any) -> str {
        source = chunk.metadata.get('source', 'unknown');
        page = chunk.metadata.get('page');
        chunk_id = chunk.metadata.get('id', '');

        metadata_prefix = f"Source: {source}";
        if page is not None {
            metadata_prefix += f", Page: {page}";
        }
        if chunk_id {
            metadata_prefix += f", ChunkID: {chunk_id}";
        }

        return f"{metadata_prefix}\n\n{chunk.page_content}";
    }

    def add_chunk_id(chunks: list) -> list {
        last_page_id = "";
        current_chunk_index = 0;

        for chunk in chunks {
            source = chunk.metadata.get('source');
            page = chunk.metadata.get('page');

            # Markdown files carry no page numbers
            if page is None {
                current_page_id = f'{source}:0';
            } else {
                current_page_id = f'{source}:{page}';
            }

            if current_page_id == last_page_id {
                current_chunk_index += 1;
            } else {
                current_chunk_index = 0;
            }

            chunk_id = f'{current_page_id}:{current_chunk_index}';
            last_page_id = current_page_id;

            chunk.metadata['id'] = chunk_id;
            # Keep the original text; embed the metadata-enriched version
            chunk.metadata['original_content'] = chunk.page_content;
            chunk.page_content = self.enrich_content_with_metadata(chunk);
        }

        return chunks;
    }

    def create_faiss_index(chunks: list) -> None {
        if len(chunks) == 0 {
            print("No documents to index");
            return;
        }

        # Release notes describe old versions; keep them out of the index
        filtered_chunks = [
            chunk for chunk in chunks
            if "release_notes" not in cast(str, chunk.metadata.get('source', '')).lower()
        ];

        if len(filtered_chunks) == 0 {
            print("No documents to index after filtering");
            return;
        }

        print(f"Creating FAISS index with {len(filtered_chunks)} chunks (with metadata enrichment)");
        filtered_chunks = self.add_chunk_id(filtered_chunks);
        vs = FAISS.from_documents(filtered_chunks, self.get_embedding_function());
        vs.save_local(self.faiss_path);
        self.vectorstore = vs;
        print(f"FAISS index saved to {self.faiss_path}");
    }

    def add_to_faiss(chunks: list) -> None {
        vs = self.vectorstore;
        if vs is None {
            self.create_faiss_index(chunks);
        } elif len(chunks) > 0 {
            print(f"Adding {len(chunks)} new documents to FAISS index");
            vs.add_documents(chunks);
            vs.save_local(self.faiss_path);
            print("FAISS index updated");
        } else {
            print("No new documents to add");
        }
    }

    """Rerank (doc, score) pairs with the CrossEncoder; keep the top_n best."""
    def rerank_results(query: str, documents: list, top_n: int | None = None) -> list {
        ce = self.cross_encoder;
        if not self.use_reranking or ce is None {
            return documents;
        }

        try {
            rerank_start = time.perf_counter();
            docs_only = [pair[0] for pair in documents];
            pairs = [
                (query, doc.metadata.get('original_content', doc.page_content))
                for doc in docs_only
            ];

            scores = ce.predict(
                pairs,
                show_progress_bar=False,
                batch_size=30,
                convert_to_numpy=True,
                convert_to_tensor=False
            );

            doc_score_pairs: list = [];
            i = 0;
            for doc in docs_only {
                doc_score_pairs.append((doc, scores[i]));
                i += 1;
            }

            def neg_score(pair: any) -> float {
                return -cast(float, pair[1]);
            }
            results = sorted(doc_score_pairs, key=neg_score);

            if top_n is not None {
                results = results[:top_n];
            }

            rerank_elapsed = time.perf_counter() - rerank_start;
            print(f"Reranking time: {rerank_elapsed:.4f}s");
            print(f"Reranked {len(results)} documents using CrossEncoder");
            return results;
        } except Exception as e {
            print(f"Error during reranking: {e}");
            print("Falling back to original FAISS results");
            return documents;
        }
    }

    def get_from_faiss(query: str) -> list {
        vs = self.vectorstore;
        if vs is None {
            print("FAISS index not initialized");
            return [];
        }
        search_start = time.perf_counter();
        results = cast(list, vs.similarity_search_with_score(query, k=self.chunk_nos));
        search_elapsed = time.perf_counter() - search_start;
        print(f"Search time: {search_elapsed:.4f}s");

        if self.use_reranking and len(results) > 0 {
            results = self.rerank_results(query, results, top_n=self.top_n);
        }

        return results;
    }

    """Retrieve relevant chunks for `query` and format them as one summary string."""
    def search(query: str) -> str {
        print(f'Searching RAG Engine with query: {query}');
        rag_start = time.perf_counter();
        results = self.get_from_faiss(query=query);

        print(f"\n{'='*60}");
        print(f"Retrieved {len(results)} chunks for query: {query}");
        print(f"{'='*60}");

        summary = "";
        idx = 0;
        for pair in results {
            idx += 1;
            doc: any = pair[0];
            score: any = pair[1];
            page = doc.metadata.get('page');
            source = doc.metadata.get('source');
            chunk_id = doc.metadata.get('id');
            chunk_txt = doc.page_content;

            print(f"\nChunk {idx}:");
            print(f"  Score: {score:.4f}");
            print(f"  Source: {source}");
            print(f"  ID: {chunk_id}");
            if page is not None {
                print(f"  Page: {page}");
            }
            print(f"  Content preview: {chunk_txt[:100]}...");

            original_content = doc.metadata.get('original_content', chunk_txt);
            summary += f"{source} (Relevance Score: {score:.4f}) : {original_content}\n";
        }

        print(f"\n{'='*60}\n");
        rag_elapsed = time.perf_counter() - rag_start;
        print(f"RAG time: {rag_elapsed:.4f}s");
        return summary;
    }
}
