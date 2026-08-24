import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=8964,
    program_name="research"
);

def search_archive(topic: str) -> str {
    time.sleep(0.35 * TOOL_SCALE);
    records = {
        "retention": "Audit records are retained for seven years.",
        "encryption": "Customer exports use AES-256 encryption at rest.",
        "access": "Production access requires hardware-backed MFA."
    };
    return str(records.get(topic.lower(), "No matching archive record."));
}

def verify_archive(claim: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    return f"Archive verification completed for: {claim}";
}

sem search_archive = "Search the internal policy archive for one topic.";
sem search_archive.topic = "one of retention, encryption, or access";
sem verify_archive = "Verify one proposed claim against the archive index.";
sem verify_archive.claim = "the exact claim that needs independent verification";

def investigate(question: str) -> str by llm(
    tools=[search_archive, verify_archive],
    temperature=0.0,
    max_tokens=320,
    max_react_iterations=4
);

def draft_answer(question: str, evidence: str) -> str by llm(
    temperature=0.0,
    max_tokens=320
);

def audit_answer(answer: str) -> str by llm(
    temperature=0.0,
    max_tokens=160
);

sem investigate = "Research the question using the archive tools. Call search_archive for the relevant topic, verify the key claim, then finish with a concise evidence report.";
sem investigate.question = "the policy question to investigate";
sem draft_answer = "Answer the policy question using only the supplied evidence. Include the decisive archive sentence verbatim.";
sem draft_answer.question = "the original policy question";
sem draft_answer.evidence = "the completed archive investigation";
sem audit_answer = "Check whether the proposed answer is fully supported and return PASS or FAIL followed by one sentence.";
sem audit_answer.answer = "the answer awaiting support review";

with entry:__main__ {
    question = str(os.environ.get("QUESTION") or "How long are audit records retained?");
    evidence = investigate(question);
    time.sleep(0.20 * TOOL_SCALE);
    answer = draft_answer(question, evidence);
    time.sleep(0.20 * TOOL_SCALE);
    verdict = audit_answer(answer);
    print("ANSWER:", answer);
    print("AUDIT:", verdict);
}
