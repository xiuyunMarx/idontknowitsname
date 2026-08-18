import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct");

def analyze(text: str) -> str by llm(temperature=0.2, max_tokens=96);
sem analyze = """
Deep technical analyst for incoming bug reports on a GPU training platform. Read the raw report and produce a structured analysis with these exact sections: SYMPTOM (one line restating the observable failure in neutral language), SUBSYSTEM (which of: kernel driver, CUDA runtime, cooling/hardware, training framework, user code — pick exactly one and justify in one clause), SEVERITY (P0 data-loss/fire, P1 blocked training, P2 degraded performance, P3 cosmetic — one line), EVIDENCE (bullet list of every concrete fact present in the report, no speculation), MISSING (bullet list of the three most important facts the reporter did not provide that a triage engineer would need next). Never propose fixes at this stage; never exceed 150 words; never invent facts not present in the report; keep bullets to one line each and use plain ASCII dashes for bullets.
""";

def draft(analysis: str) -> str by llm(temperature=0.4, max_tokens=96);
sem draft = """
Support-response drafter for the same GPU training platform. Input is a structured triage analysis with SYMPTOM/SUBSYSTEM/SEVERITY/EVIDENCE/MISSING sections. Produce a first-draft reply to the customer that: opens with a one-sentence acknowledgement mirroring their SYMPTOM in empathetic but non-alarmist language; states the current triage classification (subsystem and severity) in plain words a non-engineer understands; asks for exactly the facts listed under MISSING as a numbered list of questions, each answerable in one sentence or one attached file; closes with the concrete next step and a realistic response-time expectation. Hard constraints: no apology theater beyond the single opening sentence, no speculation about root cause, no promises of fixes or timelines for fixes, no exclamation marks, under 180 words total, and every question must map to one MISSING item — no extra questions.
""";

def polish(draft_text: str) -> str by llm(temperature=0.3, max_tokens=96);
sem polish = """
Final-pass editor for outgoing customer communication. Input is a draft support reply. Rewrite it so that: sentences average under 20 words; passive voice is eliminated except where the actor is genuinely unknown; jargon is replaced by plain terms (say "graphics card" not "GPU SKU", "error message" not "stack trace") while keeping any exact error strings verbatim in quotes; the numbered questions from the draft survive with identical meaning and order; tone lands professional-warm, never chirpy, never robotic. Preserve all factual content exactly — you may reorder sentences within a paragraph but never across the acknowledgement/classification/questions/next-step structure. Output only the final reply text with no meta-commentary, no subject line, and no signature block.
""";

def count_words(text: str) -> int by llm(temperature=0.0);
sem count_words = "Count the number of words in the text. Return only the count.";

with entry:__main__ {
    a = analyze("My GPU catches fire every time I run the training job!!");
    d = draft(a);
    p = polish(d);
    print("POLISHED:", p);
    n = count_words(text=p);
    print("TYPED:", type(n).__name__, n);
}
