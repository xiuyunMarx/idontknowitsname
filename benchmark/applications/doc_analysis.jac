import os;
import re;
import sys;
import json;
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B");
glob HERE: str = os.path.dirname(os.path.abspath(__file__));
glob DATA: str = os.environ.get("DA_DATA") or os.path.join(HERE, "bench_data", "finicial_bench", "financebench_open_source.jsonl");
glob PAGES_DIR: str = os.environ.get("DA_PAGES_DIR") or os.path.join(HERE, "bench_data", "finicial_bench", "pages");
glob DOC_CHARS: int = int(os.environ.get("DA_DOC_CHARS") or "40000");
glob PAGE_WINDOW: int = int(os.environ.get("DA_PAGE_WINDOW") or "4");


obj Fact {
    has statement: str = "";
    has quote: str = "";
    has page: int = 0;
}

sem Fact.statement = "One fact needed to answer the question: the figure, its unit and fiscal period.";
sem Fact.quote = "The sentence or table row of the document the fact comes from, verbatim.";
sem Fact.page = "The page number of the quote, from the page markers in the document.";


obj Analysis {
    has reasoning: str = "";
    has result: str = "";
}

sem Analysis.reasoning = "The computation or comparison performed on the facts, step by step.";
sem Analysis.result = "The resulting value or conclusion, with its unit.";


obj Audit {
    has passed: bool = False;
    has issues: list[str] = [];
}

sem Audit.passed = "Whether every fact is faithful to the document and the analysis follows from the facts.";
sem Audit.issues = "Each discrepancy found, or empty when the analysis holds.";


obj Answer {
    has answer: str = "";
    has rationale: str = "";
}

sem Answer.answer = "The final answer: a number with its unit, or a short phrase.";
sem Answer.rationale = "One paragraph justifying the answer from the facts.";


walker DocTask {
    has question: str;
    has document: str;
    has gold: str = "";
    has facts: list[Fact] = [];
    has analysis: Analysis | None = None;
    has audit: Audit | None = None;
    has errors: list[str] = [];
}

sem DocTask = "One session answering one question about one financial filing.";
sem DocTask.document = "The relevant pages of the filing, each preceded by a page marker.";


node Extractor {
    def extract(document: str, question: str) -> list[Fact] by llm(temperature=0.0, max_tokens=4096);

    can run with DocTask entry {
        try {
            visitor.facts = self.extract(visitor.document, visitor.question);
        } except Exception as e {
            visitor.errors.append(f"extraction failed: {e}");
        }
        visit [self-->][?:Analyst];
    }
}

sem Extractor = "Reads the filing pages and pulls out the facts the question needs.";
sem Extractor.extract = "List every fact from `document` needed to answer `question`: the figures with their units and fiscal periods, each quoted from the document with its page. Do not compute and do not answer.";
sem Extractor.extract.document = "The relevant pages of the filing.";
sem Extractor.extract.question = "The question to be answered.";


node Analyst {
    def compute(document: str, question: str, facts: list[Fact]) -> Analysis by llm(temperature=0.0, max_tokens=4096);

    can run with DocTask entry {
        try {
            visitor.analysis = self.compute(visitor.document, visitor.question, visitor.facts);
        } except Exception as e {
            visitor.errors.append(f"analysis failed: {e}");
            visitor.analysis = Analysis(reasoning="The analysis could not be produced.");
        }
        visit [self-->][?:Auditor];
    }
}

sem Analyst = "Derives the answer from the extracted facts.";
sem Analyst.compute = "Using `facts` and, where a fact is missing, `document`, perform the computation or comparison `question` asks for. Show every step with its figures. State the result with its unit.";
sem Analyst.compute.document = "The relevant pages of the filing.";
sem Analyst.compute.question = "The question to be answered.";
sem Analyst.compute.facts = "The facts the extractor found, in document order.";


node Auditor {
    def check(document: str, question: str, facts: list[Fact], analysis: Analysis) -> Audit by llm(temperature=0.0, max_tokens=4096);

    can run with DocTask entry {
        analysis = visitor.analysis or Analysis();
        try {
            visitor.audit = self.check(visitor.document, visitor.question, visitor.facts, analysis);
        } except Exception as e {
            visitor.errors.append(f"audit failed: {e}");
            visitor.audit = Audit(passed=False, issues=["The audit could not be produced."]);
        }
        visit [self-->][?:Writer];
    }
}

sem Auditor = "Checks the facts against the filing and the analysis against the facts.";
sem Auditor.check = "Verify each of `facts` against `document` and verify that `analysis` answers `question` from those facts without arithmetic or unit errors. Report every discrepancy.";
sem Auditor.check.document = "The relevant pages of the filing.";
sem Auditor.check.question = "The question to be answered.";
sem Auditor.check.facts = "The facts the extractor found.";
sem Auditor.check.analysis = "The analyst's computation and result.";


node Writer {
    def answer(question: str, facts: list[Fact], analysis: Analysis, audit: Audit) -> Answer by llm(temperature=0.0, max_tokens=4096);

    can run with DocTask entry {
        analysis = visitor.analysis or Analysis();
        audit = visitor.audit or Audit();
        try {
            final = self.answer(visitor.question, visitor.facts, analysis, audit);
        } except Exception as e {
            visitor.errors.append(f"writing failed: {e}");
            final = Answer(answer=analysis.result);
        }
        ok = matches(final.answer, visitor.gold) if visitor.gold else False;
        print(f"Verdict: {'PASSED' if ok else 'FAILED'}");
        print(f"Answer: {final.answer}");
        print(f"Gold: {visitor.gold}");
        print(f"Rounds: 1, notes: {len(visitor.facts)}");
        if visitor.errors {
            print("Diagnostics:");
            for error in visitor.errors {
                print(f"- {error}");
            }
        }
    }
}

sem Writer = "Writes the final answer.";
sem Writer.answer = "Answer `question` from `facts` and `analysis`. If `audit` reports issues, correct the answer accordingly. Give the number with its unit, or the phrase the question asks for.";
sem Writer.answer.question = "The question to be answered.";
sem Writer.answer.facts = "The facts the extractor found.";
sem Writer.answer.analysis = "The analyst's computation and result.";
sem Writer.answer.audit = "The auditor's verdict and issues.";


def numbers(text: str) -> list[float] {
    out: list[float] = [];
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text) {
        try {
            out.append(float(str(tok).replace(",", "")));
        } except ValueError {}
    }
    return out;
}

def close(a: float, g: float) -> bool {
    d = a - g;
    if d < 0.0 {
        d = -d;
    }
    tol = 0.005 * (g if g >= 0.0 else -g) + 0.005;
    return d <= tol;
}

def matches(answer: str, gold: str) -> bool {
    wanted = numbers(gold);
    if not wanted {
        return gold.strip().lower() in answer.lower();
    }
    got = numbers(answer);
    for g in wanted {
        if not any([close(a, g) for a in got]) {
            return False;
        }
    }
    return True;
}

def page_window(doc_name: str, evidence_pages: list[int]) -> dict[int, str] {
    path = os.path.join(PAGES_DIR, f"{doc_name}.json");
    if not os.path.exists(path) {
        return {};
    }
    with open(path) as f {
        texts: list = list(json.load(f));
    }
    out: dict[int, str] = {};
    for d in range(PAGE_WINDOW + 1) {
        for p in evidence_pages {
            for q in (p - d, p + d) {
                if 0 <= q < len(texts) and q not in out {
                    out[q] = str(texts[q]);
                }
            }
        }
    }
    return out;
}

def load_task(task_id: str) -> DocTask {
    rows: list[dict] = [];
    with open(DATA) as f {
        for line in f {
            if line.strip() {
                rows.append(dict(json.loads(line)));
            }
        }
    }
    row: dict = {};
    for r in rows {
        if r["financebench_id"] == task_id {
            row = dict(r);
        }
    }
    if not row {
        raise KeyError(f"{task_id} not in {DATA}");
    }
    evidence_pages: list[int] = [];
    pages: dict[int, str] = {};
    for ev in list(row["evidence"]) {
        p = int(ev["evidence_page_num"]);
        evidence_pages.append(p);
        pages.setdefault(p, str(ev["evidence_text_full_page"]));
    }
    window = page_window(str(row["doc_name"]), evidence_pages);
    if window {
        pages = window;
    } else {
        for r in rows {
            if r["doc_name"] == row["doc_name"] {
                for ev in list(r["evidence"]) {
                    pages.setdefault(int(ev["evidence_page_num"]), str(ev["evidence_text_full_page"]));
                }
            }
        }
    }
    used = 0;
    keep: list[int] = [];
    for p in evidence_pages + [p for p in pages.keys() if p not in evidence_pages] {
        used += len(pages[p]);
        if used > DOC_CHARS and keep {
            break;
        }
        keep.append(p);
    }
    parts = [f"[page {p}]\n{pages[p]}" for p in sorted(keep)];
    return DocTask(question=str(row["question"]), document="\n\n".join(parts), gold=str(row["answer"]));
}

def build_graph() -> Extractor {
    existing = [root-->][?:Extractor];
    if existing {
        return existing[0];
    }
    extractor = root ++> Extractor();
    analyst = Analyst();
    auditor = Auditor();
    writer = Writer();
    extractor ++> analyst;
    analyst ++> auditor;
    auditor ++> writer;
    return extractor;
}


with entry {
    task_id = os.environ.get("DA_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "financebench_id_03029");
    load_task(task_id) spawn build_graph();
}
