"""Download the FinanceBench filings and extract their page texts.

    python -m benchmark.applications.prepare_financebench [--workers 8]

Writes bench_data/finicial_bench/pdfs/<doc>.pdf and pages/<doc>.json (a list of
page strings, zero-indexed like evidence_page_num). Existing files are kept.
"""
import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pypdf import PdfReader

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_data", "finicial_bench")
JSONL = os.path.join(HERE, "financebench_open_source.jsonl")
PDF_DIR = os.path.join(HERE, "pdfs")
PAGES_DIR = os.path.join(HERE, "pages")
URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"


def prepare(doc: str) -> str:
    pdf = os.path.join(PDF_DIR, f"{doc}.pdf")
    pages = os.path.join(PAGES_DIR, f"{doc}.json")
    if os.path.exists(pages):
        return f"{doc}: cached"
    if not os.path.exists(pdf):
        urllib.request.urlretrieve(URL.format(doc), pdf + ".part")
        os.replace(pdf + ".part", pdf)
    texts = [(page.extract_text() or "") for page in PdfReader(pdf).pages]
    with open(pages, "w") as f:
        json.dump(texts, f)
    return f"{doc}: {len(texts)} pages, {sum(len(t) for t in texts) // 1000}k chars"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PAGES_DIR, exist_ok=True)
    with open(JSONL) as f:
        docs = sorted({json.loads(line)["doc_name"] for line in f if line.strip()})
    failed = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for doc, result in zip(docs, pool.map(lambda d: _safe(d), docs)):
            print(result, flush=True)
            failed += result.endswith("FAILED")
    print(f"{len(docs) - failed}/{len(docs)} documents ready", flush=True)
    return 1 if failed else 0


def _safe(doc: str) -> str:
    try:
        return prepare(doc)
    except Exception as e:
        return f"{doc}: {e} FAILED"


if __name__ == "__main__":
    sys.exit(main())
