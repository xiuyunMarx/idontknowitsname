"""Entry point for multi_intent."""

import os;
import from jaclang.byllm.llm { InterceptorLLM }
import os;
import from langchain_community.vectorstores { FAISS }
import from langchain_huggingface { HuggingFaceEmbeddings }
import from sentence_transformers { CrossEncoder }
import time;
glob llm = InterceptorLLM(model_name="Qwen/Qwen3-4B");
obj RagStore {
    has corpus_dir: str;
    has embedding_model: str = "BAAI/bge-small-en-v1.5";
    has cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    has top_k: int = 1;
    has fetch_k: int = 2;
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

    # Top passages of one corpus file for a query: embed + FAISS over the whole
    # index, keep the requested source, rerank with the cross-encoder.
    def search(query: str, source: str) -> str {
        time.sleep(0.3);  # simulate tool latency
        docs = self.vectorstore.similarity_search(query, k=32);
        docs = [d for d in docs if d.metadata.get("source") == source][:self.fetch_k];
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

glob ROUTER_INTENT: str = "Route the normalized service ticket to the single desk best equipped to resolve it: quantitative word problems and anything requiring arithmetic go to the math desk, requests to design or write program code go to the code desk, questions whose answer must be looked up in the product documentation go to the docs desk, and requests to compose, rewrite or polish prose go to the writing desk.";

def normalize(request: str) -> str by llm(temperature=0.0, max_tokens=96);
sem normalize = "You are the intake clerk of a multi-desk service counter. Rewrite the customer's free-text message as one normalized service ticket of at most three sentences: state the kind of task being requested, the concrete deliverable the customer expects, and every number, name, constraint, quotation or deadline the message contains. Preserve every factual detail exactly; add nothing and answer nothing. Output only the ticket text.";

walker Desk {
    has request: str;
    has profile: str = "";
    has response: str = "";

    can triage with Root entry {
        self.profile = normalize(self.request);
        visit [-->] by llm(select=1, intent=ROUTER_INTENT);
    }
}
sem Desk = "A service walker carrying one customer request from intake through routing to the desk that resolves it.";
sem Desk.request = "the customer's original free-text message";
sem Desk.profile = "the normalized service ticket produced by intake";
sem Desk.response = "the resolving desk's final brief";
enum CodeVerdict { READY = "READY", NEEDS_REVIEW = "NEEDS_REVIEW", BLOCKED = "BLOCKED" }

obj CodeBrief {
    has signature: str;
    has summary: str;
    has verdict: CodeVerdict;
}
sem CodeBrief = "the signed-off case brief of the software implementation desk";
sem CodeBrief.signature = "the exact def line of the delivered function";
sem CodeBrief.summary = "at most three sentences stating what was implemented and how the specified edge cases are handled";
sem CodeBrief.verdict = "READY when every requirement and edge case in the specification is covered by the implementation, NEEDS_REVIEW when any is doubtful, BLOCKED when the ticket cannot be implemented as specified";

def code_snippets(analysis: str) -> str {
    return rag.search(analysis, "code_library.txt");
}

def code_review_refs(implementation: str) -> str {
    return rag.search(implementation, "code_library.txt");
}

node CodeDesk {
    def analyze(ticket: str) -> str by llm(temperature=0.0, max_tokens=420);
    def implement(analysis: str, snippets: str) -> str by llm(temperature=0.0, max_tokens=560);
    def finalize(analysis: str, implementation: str, review_refs: str) -> CodeBrief by llm(temperature=0.0, max_tokens=240);

    can handle with Desk entry {
        analysis = self.analyze(visitor.profile);
        snippets = code_snippets(analysis);
        implementation = self.implement(analysis, snippets);
        review_refs = code_review_refs(implementation);
        brief = self.finalize(analysis, implementation, review_refs);
        visitor.response = str(brief);
    }
}
sem CodeDesk = "The software implementation desk: turns programming tickets into a written specification, a working Python implementation, and a signed-off review brief.";
sem CodeDesk.analyze = "You are the specification analyst of the software implementation desk. Restate the programming ticket as a numbered requirements specification in plain prose, at least ten numbered sentences each stating exactly one requirement, ordered as follows. Sentences one and two give the exact function name and full signature with parameter names and their types. The following sentences give the domain of every input, including whether it may be empty, contain duplicates, be unsorted, hold negative values, or mix types. Then the output contract: the return type and the exact value returned in every situation, including empty and degenerate inputs. Then at least three edge cases, each as its own sentence naming a concrete literal input and the expected observable result, chosen from: empty input, single element, all elements equal, boundary values, and structurally malformed input. Then error handling: for each class of invalid input, state whether the function raises, returns a sentinel, or normalizes it, and which. Every requirement must be concrete and testable — name literal example values rather than categories wherever possible, and never write a requirement whose satisfaction could not be checked by a single assert. If the ticket names an expected example or a sample invocation, restate it verbatim as its own requirement sentence. Do not write any code and do not sketch an algorithm; specify only observable behavior. Plain prose, no markdown.";
sem CodeDesk.implement = "You are the implementer of the software implementation desk. Following the numbered requirements specification exactly and respecting the conventions shown in the team's retrieved reference snippets, write one self-contained Python function that fulfils every requirement: standard library only, a docstring stating the contract in one or two lines, explicit handling of every edge case the specification lists, and no behavior the specification does not require. Respect the team conventions concretely: return early on invalid input instead of nesting, never mutate an argument the caller still owns — copy before modifying, prefer a plain loop over a clever one-liner when it reads better, and when a retrieved pattern matches the task, follow its structure rather than inventing a new one. Choose variable names that echo the specification's own vocabulary so the reviewer can trace requirements into code. Output the complete function source first, as plain text with normal indentation and no markdown fences, then a numbered explanation with exactly one sentence per specified edge case stating where in the code, described by its logic rather than a line number, that case is handled; if an edge case is handled implicitly by the algorithm's structure, say so and name the structural reason. Add no example usage, no tests, no prints, and no commentary beyond the numbered explanation.";
sem CodeDesk.finalize = "You are the senior reviewer of the software implementation desk signing off the case. Compare the delivered implementation against the numbered requirements specification requirement by requirement, applying the retrieved review references — the team's review checklist and known pitfalls — to the code as written: extract the exact def line as the signature, summarize in at most three sentences what was implemented and how the specified edge cases are handled, and issue the verdict — READY only when every numbered requirement and every listed edge case is verifiably covered and no checklist item or known pitfall from the review references applies unaddressed, NEEDS_REVIEW when any requirement's coverage is doubtful or a referenced pitfall may apply, BLOCKED when the specification itself cannot be satisfied.";


enum DocSupport { FULL = "FULL", PARTIAL = "PARTIAL", NONE = "NONE" }

obj DocBrief {
    has answer: str;
    has citation: str;
    has support: DocSupport;
}
sem DocBrief = "the signed-off case brief of the documentation desk";
sem DocBrief.answer = "the direct answer to the customer's question in one or two sentences";
sem DocBrief.citation = "the one documentation sentence that supports the answer, quoted verbatim";
sem DocBrief.support = "FULL when the documentation states the answer explicitly, PARTIAL when it must be inferred, NONE when the documentation does not contain it";

def doc_retrieve(analysis: str) -> str {
    return rag.search(analysis, "meridian_handbook.txt");
}

def doc_recheck(findings: str) -> str {
    return rag.search(findings, "meridian_handbook.txt");
}

node DocDesk {
    def analyze(question: str) -> str by llm(temperature=0.0, max_tokens=420);
    def answer(analysis: str, passages: str) -> str by llm(temperature=0.0, max_tokens=560);
    def finalize(analysis: str, findings: str, recheck: str) -> DocBrief by llm(temperature=0.0, max_tokens=240);

    can handle with Desk entry {
        analysis = self.analyze(visitor.profile);
        passages = doc_retrieve(analysis);
        findings = self.answer(analysis, passages);
        recheck = doc_recheck(findings);
        brief = self.finalize(analysis, findings, recheck);
        visitor.response = str(brief);
    }
}
sem DocDesk = "The documentation desk: answers product questions strictly from retrieved documentation passages, never from memory.";
sem DocDesk.analyze = "You are the research planner of the documentation desk. Decompose the customer's question in the service ticket into an explicit lookup plan written as numbered plain-prose sentences: the precise fact or facts that must be found, the entities and attributes each fact concerns, the section of a product handbook where each fact would normally live such as pricing, retention, security or exports, the exact search terms you would use for each fact, and what a complete answer must contain to fully satisfy the question. Write at least eight numbered sentences with exactly one statement each. Do not answer the question and do not speculate about what the documentation might say. Plain prose, no markdown.";
sem DocDesk.answer = "You are the researcher of the documentation desk. Execute the lookup plan strictly against the retrieved passages provided and nothing else: for each fact the plan requires, name the passage that contains it, quote the supporting sentence verbatim inside single quotes, and state the fact in your own words; where the passages do not contain a required fact, say so explicitly instead of guessing, and never import knowledge from outside the passages. After covering every fact in the plan, state in one closing sentence whether the plan's requirements for a complete answer are now fully met, partially met, or unmet. Number every sentence. Plain prose, no markdown.";
sem DocDesk.finalize = "You are the senior editor of the documentation desk signing off the case. From the lookup plan, the researcher's findings, and the independently re-retrieved verification passages, produce the final brief for the customer record: state the direct answer to the customer's question in one or two sentences, select the single verbatim quotation from the findings that best supports it as the citation, and grade the support — FULL only when the quoted documentation states the answer explicitly and the verification passages do not contradict it, PARTIAL when the answer required combining or inferring from passages or the verification passages differ in any detail, NONE when the findings show the documentation does not contain the answer, in which case the answer must say the documentation does not cover it.";

enum MathConfidence { LOW = "LOW", MEDIUM = "MEDIUM", HIGH = "HIGH" }

obj MathBrief {
    has answer: float;
    has method: str;
    has confidence: MathConfidence;
}
sem MathBrief = "the signed-off case brief of the quantitative reasoning desk";
sem MathBrief.answer = "the final numeric answer as a plain number, no units and no thousands separators";
sem MathBrief.method = "at most three sentences summarizing how the answer was derived";
sem MathBrief.confidence = "confidence that the worked solution answers exactly the quantity the customer asked for";

def math_reference(analysis: str) -> str {
    return rag.search(analysis, "math_handbook.txt");
}

def math_audit(solution: str) -> str {
    return rag.search(solution, "math_handbook.txt");
}

node MathDesk {
    def analyze(problem: str) -> str by llm(temperature=0.0, max_tokens=420);
    def solve(analysis: str, reference: str) -> str by llm(temperature=0.0, max_tokens=560);
    def finalize(analysis: str, solution: str, audit: str) -> MathBrief by llm(temperature=0.0, max_tokens=240);

    can handle with Desk entry {
        analysis = self.analyze(visitor.profile);
        reference = math_reference(analysis);
        solution = self.solve(analysis, reference);
        audit = math_audit(solution);
        brief = self.finalize(analysis, solution, audit);
        visitor.response = str(brief);
    }
}
sem MathDesk = "The quantitative reasoning desk: resolves arithmetic and word-problem tickets by restating the problem, deriving a step-by-step solution, and signing off a numeric answer.";
sem MathDesk.analyze = "You are the intake analyst of the quantitative reasoning desk. Restate the word problem in the service ticket as a complete inventory of its mathematical content, written as numbered plain-prose sentences: every given quantity with its unit, every relationship between quantities that is stated or implied, every intermediate quantity that will have to be computed on the way to the answer, and finally the exact quantity the customer asks for. Write at least ten numbered sentences, each containing exactly one factual statement. Expand every implicit relationship into an explicit arithmetic relation between named quantities, using this catalog: twice or double means multiply by 2, half means divide by 2, a dozen means 12 units, per unit phrasing means a multiplication by a count, fewer or less than means a subtraction from the referenced quantity, more than means an addition, and percent means per hundred applied to the stated base. Give every quantity a short reusable name such as monday_count or unit_price and use that name consistently in every later sentence that references it. State the unit of every quantity every time it appears; if the ticket mixes units, add one sentence per required conversion naming the conversion factor. If the ticket contains a number that is irrelevant to the asked quantity, include one sentence explicitly marking it irrelevant so the solver does not use it. If the customer's message itself asserts a total or an answer, record it as a claim to be verified, never as a given quantity. Close with one final sentence that names, in computation order, the intermediate quantities the solver must produce. Do not solve the problem, do not perform any arithmetic beyond restating what is given, and add no assumption that is not in the ticket. Plain prose, no markdown, no formulas in special notation.";
sem MathDesk.solve = "You are the solver of the quantitative reasoning desk. Using the structured problem inventory and the retrieved reference material provided, derive the answer step by step: write one numbered sentence per arithmetic step. Each step must name its operands by the inventory's quantity names, state the operation in words, give the intermediate result with its unit, and restate the computation once in words as a self-check, in the style of: multiplying monday_count of 24 muffins by unit_price of 3 dollars gives monday_revenue of 72 dollars. Use the retrieved reference material as follows: apply a formula from it only where the inventory's relationships call for it, and when a worked example in the material matches the problem's shape, mirror its step structure but never copy its numbers. Follow the inventory's list of intermediate quantities in their stated order and compute nothing the inventory does not require; if the inventory marked a number as irrelevant, do not use it in any step. If the inventory recorded a claimed total from the ticket, add one sentence near the end comparing your computed result against that claim and state plainly whether the claim is consistent or wrong; your own computation always takes precedence over the claim. Carry units through every step and never mix units within one operation without first converting. Every five steps, summarize in one sentence what has been established so far. Conclude with a single sentence that begins with the words FINAL RESULT: followed by the numeric answer and its unit. Plain prose, no markdown.";
sem MathDesk.finalize = "You are the senior reviewer of the quantitative reasoning desk signing off the case. From the problem inventory, the worked solution, and the retrieved audit material containing similar worked examples with reference answers, produce the final brief for the customer record. The answer field must contain the number from the solution's FINAL RESULT sentence as a plain number with no units, no currency symbols and no thousands separators; if the solution lacks a FINAL RESULT sentence, extract the last computed total instead and grade confidence LOW. The method field summarizes the derivation in at most three sentences naming the key steps by their inventory quantity names. Grade confidence by this rubric: HIGH only when every intermediate quantity the inventory listed was computed, no step used a quantity the inventory marked irrelevant, units are consistent from givens to answer, the step structure is consistent with the pattern the audit material's comparable worked examples establish, and the final quantity is exactly the one the inventory says the customer asked for; MEDIUM when the arithmetic chain is complete but one of those checks is doubtful, for example a unit stated inconsistently, a claimed total in the ticket contradicting the result without the solution addressing it, or a deviation from the audit material's pattern without a reason grounded in the inventory; LOW when a required intermediate quantity was never computed, a step contradicts the inventory, or the solution answered a different quantity than the one asked. Never recompute the mathematics yourself and never blend the solution's result with the ticket's claimed total: your job is extraction and audit, not solving.";

enum WriteRegister { FORMAL = "FORMAL", NEUTRAL = "NEUTRAL", CASUAL = "CASUAL" }

obj WriteBrief {
    has final_text: str;
    has register: WriteRegister;
    has summary: str;
}
sem WriteBrief = "the signed-off case brief of the writing desk";
sem WriteBrief.final_text = "the polished final text, complete and ready to deliver to the customer";
sem WriteBrief.register = "the register of the delivered text";
sem WriteBrief.summary = "at most two sentences stating what was delivered and for which audience";

def style_lookup(analysis: str) -> str {
    return rag.search(analysis, "style_guide.txt");
}

def style_audit(drafted: str) -> str {
    return rag.search(drafted, "style_guide.txt");
}

node WriteDesk {
    def analyze(ticket: str) -> str by llm(temperature=0.0, max_tokens=420);
    def draft(analysis: str, style: str) -> str by llm(temperature=0.0, max_tokens=560);
    def finalize(analysis: str, drafted: str, audit_refs: str) -> WriteBrief by llm(temperature=0.0, max_tokens=400);

    can handle with Desk entry {
        analysis = self.analyze(visitor.profile);
        style = style_lookup(analysis);
        drafted = self.draft(analysis, style);
        audit_refs = style_audit(drafted);
        brief = self.finalize(analysis, drafted, audit_refs);
        visitor.response = str(brief);
    }
}
sem WriteDesk = "The writing desk: turns composition tickets into an editorial brief, a full draft, and a signed-off final text.";
sem WriteDesk.analyze = "You are the editor of the writing desk preparing an editorial brief. Restate the composition ticket as numbered plain-prose sentences covering: the audience the text is for, the purpose it must achieve with that audience, the register it should be written in and why, the target length in words taken from the ticket or a stated sensible default, every point of content the ticket requires the text to include, every constraint on tone or wording the ticket imposes, and a one-sentence outline of the structure the draft should follow. Write at least eight numbered sentences with exactly one editorial decision each, and ground every decision in something the ticket actually says. Do not write any of the text itself. Plain prose, no markdown.";
sem WriteDesk.draft = "You are the staff writer of the writing desk. Write the full text the editorial brief specifies, following the brief's audience, purpose, register, length target, required content points and structure exactly, and obeying every rule in the house style guide provided. Every required content point must appear; nothing the brief does not call for may be added; the length must stay within the brief's target. Output only the text itself with no preamble, no commentary and no markdown.";
sem WriteDesk.finalize = "You are the senior editor of the writing desk signing off the case. Review the draft against the editorial brief point by point, applying the retrieved audit references — the house revision checklist and register rules that match this piece — to the draft as written: correct any violation of the brief's register, length or required content and any breach the audit references identify, keep everything else unchanged, classify the register of the delivered text, and summarize in at most two sentences what was delivered and for which audience. The final text must be complete and ready to send.";

with entry:__main__ {
    import os;
    os.environ["JAC_ROUTE_CACHE_LAYOUT"] = "1";
    if not [root-->][?:MathDesk] {
        root ++> MathDesk();
        root ++> CodeDesk();
        root ++> DocDesk();
        root ++> WriteDesk();
    }

    request: str = str(os.environ.get("REQUEST"));
    w = root spawn Desk(request=request);
    print("PROFILE:", w.profile);
    print("RESPONSE:", w.response);
}
