import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"),
    comm_ip="localhost",
    comm_port=8968,
    program_name="pipeline"
);

def extract_fields(record: str) -> str by llm(
    temperature=0.0,
    max_tokens=160
);

def validate_fields(fields: str) -> str by llm(
    temperature=0.0,
    max_tokens=120
);

def classify_record(fields: str) -> str by llm(
    temperature=0.0,
    max_tokens=32
);

def summarize_record(fields: str, category: str) -> str by llm(
    temperature=0.0,
    max_tokens=120
);

def translate_summary(summary: str) -> str by llm(
    temperature=0.0,
    max_tokens=160
);

def format_output(summary: str, translation: str) -> str by llm(
    temperature=0.0,
    max_tokens=200
);

sem extract_fields = "Pull the named fields out of the raw record as 'name: value' lines: customer, product, quantity, incident date, severity.";
sem extract_fields.record = "the raw free-text record";
sem validate_fields = "Check the extracted lines for missing or malformed values and return OK or a list of problems.";
sem validate_fields.fields = "the extracted 'name: value' lines";
sem classify_record = "Classify the record as exactly one of: complaint, order, incident, inquiry.";
sem classify_record.fields = "the extracted 'name: value' lines";
sem summarize_record = "Summarize the record in two sentences for the daily digest.";
sem summarize_record.fields = "the extracted 'name: value' lines";
sem summarize_record.category = "the record's assigned category";
sem translate_summary = "Translate the summary into German, keeping names and dates unchanged.";
sem translate_summary.summary = "the English summary";
sem format_output = "Format the final digest entry: the English summary, then the German translation, each prefixed with its language tag.";
sem format_output.summary = "the English summary";
sem format_output.translation = "the German translation";

with entry:__main__ {
    record = str(os.environ.get("RECORD") or "Customer Meyer GmbH reports that 4 units of the RackServe X2 arrived dented on 2026-08-15, production line down, wants replacement this week.");
    fields = extract_fields(record);
    time.sleep(0.15);
    verdict = validate_fields(fields);
    time.sleep(0.10);
    category = classify_record(fields);
    time.sleep(0.10);
    summary = summarize_record(fields, category);
    time.sleep(0.15);
    translation = translate_summary(summary);
    time.sleep(0.10);
    entry_text = format_output(summary, translation);
    print("VALIDATION:", verdict);
    print("DIGEST:", entry_text);
}
