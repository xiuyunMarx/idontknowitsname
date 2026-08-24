import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"),
    comm_ip="localhost",
    comm_port=8969,
    program_name="moderation"
);

def fetch_history(author: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    return "Author has 2 prior warnings for spam, none in the last 90 days, account age 3 years.";
}

sem fetch_history = "Fetch the moderation history of one author.";
sem fetch_history.author = "the author handle mentioned in the post";

def classify_content(post: str) -> str by llm(
    temperature=0.0,
    max_tokens=32
);

def escalate_case(post: str, category: str) -> str by llm(
    tools=[fetch_history],
    temperature=0.0,
    max_tokens=280,
    max_react_iterations=3
);

def draft_warning(post: str, category: str) -> str by llm(
    temperature=0.0,
    max_tokens=200
);

def archive_note(post: str, category: str) -> str by llm(
    temperature=0.0,
    max_tokens=120
);

def log_decision(decision: str) -> str by llm(
    temperature=0.0,
    max_tokens=100
);

sem classify_content = "Classify the reported post as exactly one of: threat, harassment, spam, benign.";
sem classify_content.post = "the reported post text";
sem escalate_case = "Write the escalation ticket for a post that may be a threat or harassment. Check the author's history first, then state the risk and the recommended action.";
sem escalate_case.post = "the reported post text";
sem escalate_case.category = "the category the classifier assigned";
sem draft_warning = "Write the warning message sent to the author of a spam post: name the rule broken and the consequence of repeating it.";
sem draft_warning.post = "the reported post text";
sem draft_warning.category = "the category the classifier assigned";
sem archive_note = "Write the one-line archive note for a benign report: why no action is needed.";
sem archive_note.post = "the reported post text";
sem archive_note.category = "the category the classifier assigned";
sem log_decision = "Compress the moderation decision into one audit-log line: category, action taken, reason.";
sem log_decision.decision = "the full decision text produced for this report";

with entry:__main__ {
    post = str(os.environ.get("POST") or "Buy cheap followers now at follow-blast dot com, DM @growthguru, limited offer!!!");
    category = classify_content(post);
    time.sleep(0.15 * TOOL_SCALE);
    if ("threat" in category or "harassment" in category) {
        decision = escalate_case(post, category);
    } elif ("spam" in category) {
        decision = draft_warning(post, category);
    } else {
        decision = archive_note(post, category);
    }
    time.sleep(0.15 * TOOL_SCALE);
    print("DECISION:", decision);
    print("AUDIT:", log_decision(decision));
}
