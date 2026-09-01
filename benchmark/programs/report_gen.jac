# Report synthesis over one large document: the long-prefill application.
# Call chain seen by the server:
#   ReportStudio.outline_document -> (ReportStudio.draft_section × N, a learned loop)
#   -> ReportStudio.finalize_report
# Nearly the whole prompt of every call is the document body — dynamic per
# session (it arrives inside the first request, so it is no static anchor),
# yet byte-recoverable from value flow: the copy rule lets the server prefill
# each next call's multi-KB prompt during the think-time gaps.
import os;
import sys;
import time;
import from jaclang.byllm.llm { InterceptorLLM }
glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B", call_params={"temperature": 0.0});

glob MAX_SECTIONS = 4,
     THINK_S = 0.3;   # note-gathering pause between calls: the prefill window

obj ReportStudio{
    has style: str = "concise executive tone";

    def outline_document(document: str) -> list[str] by llm();
    def draft_section(document: str, section_title: str) -> str by llm();
    def finalize_report(document: str, drafts: list[str]) -> str by llm();
}
sem ReportStudio = "A studio that turns one source document into a short review report.";
sem ReportStudio.style = "The tone every produced text must keep.";
sem ReportStudio.outline_document = "Read the document and return 3 to 5 short section titles for a review report.";
sem ReportStudio.draft_section = "Write one paragraph for the given section, using only facts from the document.";
sem ReportStudio.finalize_report = "Merge the drafted paragraphs into the final report and add a two-sentence executive summary; stay consistent with the document.";

with entry {
    default_doc = os.path.join(os.path.dirname(__file__), "..", "datasets", "docs", "doc_01.txt");
    path = os.environ.get("RG_DOC") or (sys.argv[1] if len(sys.argv) > 1 else default_doc);
    f = open(path, "r");
    document = f.read();
    f.close();

    studio = ReportStudio();
    titles = studio.outline_document(document);
    drafts: list = [];
    for title in titles[:MAX_SECTIONS] {
        time.sleep(THINK_S);
        drafts.append(studio.draft_section(document, title));
    }
    time.sleep(THINK_S);
    print(studio.finalize_report(document, drafts));
}
