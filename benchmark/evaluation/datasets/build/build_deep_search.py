"""Build the Deep Research source pack for benchmark/evaluation/programs/deep_search.jac.

No public deep-research benchmark ships offline sources for all four sub-agents (web,
papers, data, fact-check), so the pack is assembled from real content:

  papers  arXiv API (export.arxiv.org), one relevance query per topic cluster
  web     Wikipedia REST summaries for the background entities of each cluster
  data    measured series from this repository plus seeded synthetic series, each
          labelled in its own text so the agent can see which is which
  sources fact-check statements quoted verbatim from the fetched abstracts and leads,
          each carrying the id of the document it came from

The 30 tasks are instantiated from six templates over the clusters, and each carries a
rubric naming the source ids a correct report should cite — the non-LLM quality signal
for this tenant, matching what grade() does for the SQL tenant.

    python benchmark/evaluation/datasets/build/build_deep_search.py
"""

import argparse
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "deep_search_30.json")
SEED = 20260824
N_CASES = 30
PAPERS_PER_CLUSTER = 6
ABSTRACT_CAP = 900
ARXIV_DELAY = 3.5          # export.arxiv.org asks for one request every few seconds
WIKI_DELAY = 1.0

ATOM = "{http://www.w3.org/2005/Atom}"

# Twelve clusters covering the systems literature this tenant researches. `pair` names
# the two mechanisms a comparison task contrasts; `goal` completes the question.
CLUSTERS = [
    {
        "id": "prefix-caching",
        "topic": "KV cache reuse and prefix caching",
        "arxiv": 'abs:"prefix caching" OR abs:"KV cache reuse" OR abs:"RadixAttention"',
        "wiki": ["Cache (computing)", "Memoization", "Trie"],
        "pair": ("prefix caching", "recomputing the prompt"),
        "goal": "cutting time to first token in multi-step agent workloads",
        "workload": "an agent that repeats a long system prompt on every step",
    },
    {
        "id": "disaggregation",
        "topic": "prefill and decode disaggregation",
        "arxiv": 'abs:"disaggregated" AND abs:"LLM inference"',
        "wiki": ["Distributed computing", "Latency (engineering)"],
        "pair": ("prefill-decode disaggregation", "colocated prefill and decode"),
        "goal": "holding tail latency under a service level objective",
        "workload": "a shared endpoint mixing long prompts with long generations",
    },
    {
        "id": "batching",
        "topic": "continuous batching and inference scheduling",
        "arxiv": 'abs:"continuous batching" OR abs:"iteration-level scheduling" OR abs:"chunked prefill"',
        "wiki": ["Batch processing", "Scheduling (computing)", "Throughput"],
        "pair": ("chunked prefill", "first-come-first-served prefill"),
        "goal": "trading time between tokens against throughput",
        "workload": "a serving cluster with bursty arrivals",
    },
    {
        "id": "speculative-decoding",
        "topic": "speculative decoding",
        "arxiv": 'abs:"speculative decoding"',
        "wiki": ["Speculative execution", "Branch predictor"],
        "pair": ("speculative decoding", "a larger batch size"),
        "goal": "shortening end-to-end generation latency",
        "workload": "an interactive assistant with short prompts and long answers",
    },
    {
        "id": "quantization",
        "topic": "quantized LLM serving",
        "arxiv": 'abs:"quantization" AND abs:"LLM inference"',
        "wiki": ["Quantization (signal processing)", "Half-precision floating-point format"],
        "pair": ("weight quantization", "KV cache quantization"),
        "probe": "quantization quantized weights bits precision LLM inference memory",
        "goal": "fitting a larger model on one accelerator without losing accuracy",
        "workload": "a single-GPU deployment of a mid-sized model",
    },
    {
        "id": "long-context",
        "topic": "long-context attention and memory",
        "arxiv": 'abs:"long context" AND abs:"attention" AND abs:"inference"',
        "wiki": ["Attention (machine learning)", "Transformer (deep learning architecture)"],
        "pair": ("sparse attention", "KV cache eviction"),
        "goal": "serving hundred-thousand-token prompts within a memory budget",
        "workload": "a document assistant that keeps whole reports in context",
    },
    {
        "id": "moe-serving",
        "topic": "mixture-of-experts serving",
        "arxiv": 'abs:"mixture-of-experts" AND abs:"inference"',
        "wiki": ["Mixture of experts", "Sparse matrix"],
        "pair": ("expert offloading", "expert parallelism"),
        "goal": "keeping expert routing from dominating inference latency",
        "workload": "a sparse model whose experts do not fit in device memory",
    },
    {
        "id": "rag",
        "topic": "retrieval-augmented generation pipelines",
        "arxiv": 'abs:"retrieval-augmented generation" AND abs:"efficiency"',
        "wiki": ["Information retrieval", "Vector database", "Okapi BM25"],
        "pair": ("retrieval-augmented generation", "long-context prompting"),
        "goal": "grounding answers without paying for a very long prompt",
        "workload": "a support assistant over a large document base",
    },
    {
        "id": "agents",
        "topic": "LLM agents and tool use",
        "arxiv": 'abs:"LLM agent" AND (abs:"tool use" OR abs:"tool calling")',
        "wiki": ["Software agent", "Application programming interface"],
        "pair": ("multi-agent decomposition", "a single agent with more tools"),
        "goal": "completing multi-step tasks reliably",
        "workload": "a research assistant that calls several tools per step",
    },
    {
        "id": "compiler-runtime",
        "topic": "compiler and runtime co-design for LLM programs",
        "arxiv": 'abs:"LLM" AND (abs:"compiler" OR abs:"program analysis") AND abs:"serving"',
        "wiki": ["Compiler", "Static program analysis", "Intermediate representation"],
        "pair": ("compiler-guided prefill", "runtime prediction from execution history"),
        "probe": "compiler intermediate representation static analysis LLM program serving",
        "goal": "warming a cache before the first execution of a workflow",
        "workload": "a workflow deployed for the first time, with no execution history",
    },
    {
        "id": "multi-tenant",
        "topic": "multi-tenant GPU sharing",
        "arxiv": 'abs:"multi-tenant" AND (abs:"GPU" OR abs:"inference serving")',
        "wiki": ["Multitenancy", "Graphics processing unit", "Quality of service"],
        "pair": ("fair-share scheduling", "priority scheduling"),
        "goal": "keeping one tenant's burst from hurting the others",
        "workload": "several independent programs sharing one inference engine",
    },
    {
        "id": "agent-evaluation",
        "topic": "evaluation of LLM agents and serving systems",
        "arxiv": 'abs:"benchmark" AND abs:"LLM agent" AND abs:"evaluation"',
        "wiki": ["Benchmark (computing)", "Reproducibility"],
        "pair": ("end-to-end task success", "per-call systems metrics"),
        "goal": "separating model quality from systems performance in a benchmark",
        "workload": "a study comparing two serving configurations on the same agent",
    },
]

# Measured series from this repository's own two-tenant trial (README.md).
MEASURED = {
    "ttft_ms_baseline": "Measured on this repository's RTX 3090 trial, Qwen2.5-3B-Instruct. "
                        "TTFT per successor call, baseline condition, milliseconds: 34.61, 30.39, 37.58, 19.02",
    "ttft_ms_proactive": "Measured on this repository's RTX 3090 trial, Qwen2.5-3B-Instruct. "
                         "TTFT per successor call, proactive prefill condition, milliseconds: 30.26, 12.84, 29.44, 16.08",
    "cached_prompt_tokens": "Measured on this repository's RTX 3090 trial. Prefix-cache hits per successor "
                            "call, baseline then proactive: 32 to 96, 48 to 96, 112 to 352, 48 to 112",
    "tbt_ms": "Measured on this repository's RTX 3090 trial. Mean time between tokens, single-task mode: "
              "baseline 9.13, speculative during decode 9.90, speculative only while idle 9.21",
    "uncached_tokens_per_workflow": "Measured on this repository's RTX 3090 trial, seven tenants, single "
                                    "mode. Uncached prompt tokens per workflow, baseline then proactive: "
                                    "741 to 645, 229 to 139, 747 to 371, 592 to 133, 644 to 472, 926 to 616, 796 to 188",
}

# Synthetic series, one per cluster: the text says so, so the data agent can qualify it.
SYNTHETIC_SPECS = [
    ("hit_rate_vs_prompt_reuse", "Synthetic benchmark series (seeded, not measured). Prefix-cache hit rate "
                                 "in percent as prompt reuse rises from 0 to 100 percent", 8, 5, 95),
    ("ttft_ms_vs_queue_depth", "Synthetic benchmark series (seeded, not measured). TTFT in milliseconds at "
                               "queue depth 1 through 8", 8, 20, 140),
    ("throughput_tok_s_vs_batch", "Synthetic benchmark series (seeded, not measured). Output tokens per "
                                  "second at batch size 1, 2, 4, 8, 16, 32", 6, 90, 1400),
    ("accept_rate_vs_draft_len", "Synthetic benchmark series (seeded, not measured). Speculative acceptance "
                                 "rate in percent for draft lengths 1 through 7", 7, 25, 85),
    ("accuracy_drop_vs_bits", "Synthetic benchmark series (seeded, not measured). Accuracy drop in points at "
                              "8, 6, 5, 4, and 3 bit weights", 5, 0, 12),
    ("kv_gb_vs_context_len", "Synthetic benchmark series (seeded, not measured). KV cache footprint in "
                             "gigabytes at 8k, 16k, 32k, 64k, 128k context", 5, 2, 60),
    ("expert_load_skew", "Synthetic benchmark series (seeded, not measured). Fraction of tokens routed to "
                         "each of eight experts, in percent", 8, 4, 30),
    ("recall_at_k", "Synthetic benchmark series (seeded, not measured). Retrieval recall in percent at k = "
                    "1, 3, 5, 10, 20", 5, 40, 95),
    ("tool_calls_per_task", "Synthetic benchmark series (seeded, not measured). Tool calls per completed "
                            "task across ten runs", 10, 3, 18),
    ("cold_start_uncached_tokens", "Synthetic benchmark series (seeded, not measured). Uncached prompt "
                                   "tokens on the first five executions of a new workflow", 5, 120, 800),
    ("p99_ms_vs_tenants", "Synthetic benchmark series (seeded, not measured). p99 request latency in "
                          "milliseconds with 1, 2, 4, 8, 16 tenants", 5, 60, 900),
    ("task_success_vs_turns", "Synthetic benchmark series (seeded, not measured). Task success rate in "
                              "percent after 1 through 6 repair turns", 6, 30, 88),
]

TEMPLATES = [
    ("compare",
     "Assess whether {a} or {b} is the better lever for {goal}. Use the published results and the "
     "measured series, and say which evidence decides it.",
     {"papers": 2, "pages": 1, "metrics": 1, "sources": 0}),
    ("evidence",
     "Does the series {metric} support the claim that {a} helps with {goal}? Report what the numbers "
     "show and whether the sources agree.",
     {"papers": 1, "pages": 0, "metrics": 1, "sources": 2}),
    ("provenance",
     "Which published work introduced {a}, what problem did it solve, and what background does a reader "
     "need before the result makes sense?",
     {"papers": 2, "pages": 2, "metrics": 0, "sources": 0}),
    ("tradeoff",
     "What does {a} cost, and under what conditions does that cost outweigh its benefit for {goal}?",
     {"papers": 2, "pages": 1, "metrics": 0, "sources": 1}),
    ("survey",
     "Summarise how {topic} is handled across the published systems, and name the point where they "
     "disagree.",
     {"papers": 3, "pages": 1, "metrics": 0, "sources": 1}),
    ("applicability",
     "Would {a} help {workload}? Weigh the published evidence against the measured series before "
     "answering.",
     {"papers": 2, "pages": 1, "metrics": 1, "sources": 1}),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def first_sentences(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sum(len(s) + 1 for s in kept) + len(sentence) > limit and kept:
            break
        kept.append(sentence)
    return " ".join(kept)


# ---------------------------------------------------------------------- fetching

CACHE_DIR = os.environ.get("BUILD_CACHE", "/tmp/deep_search_build")
USER_AGENT = "proactive-prefill-benchmark/1.0 (research workload builder)"


def http_get(url: str, retries: int = 5) -> bytes:
    """Fetch with a disk cache and backoff. Both APIs answer 429 under bursts, and a
    half-built pack is worse than a slow one, so every response is kept."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    import hashlib

    path = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".cache")
    if os.path.exists(path):
        return open(path, "rb").read()
    delay = 5.0
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            open(path, "wb").write(payload)
            return payload
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = float(err.headers.get("Retry-After") or delay)
            print(f"[http] {err.code} on {url[:70]}…, retrying in {wait:.0f}s")
            time.sleep(wait)
            delay *= 2
    raise RuntimeError(f"unreachable: {url}")


def arxiv_search(query: str, count: int) -> list:
    url = ("https://export.arxiv.org/api/query?"
           + urllib.parse.urlencode({"search_query": query, "start": 0,
                                     "max_results": count, "sortBy": "relevance"}))
    feed = ET.fromstring(http_get(url))
    papers = []
    for entry in feed.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", "")
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        published = entry.findtext(f"{ATOM}published", "")
        authors = [clean(a.findtext(f"{ATOM}name", "")) for a in entry.findall(f"{ATOM}author")]
        papers.append({
            "id": f"arxiv-{arxiv_id.replace('.', '-').split('v')[0]}",
            "arxiv_id": arxiv_id,
            "title": clean(entry.findtext(f"{ATOM}title", "")),
            "year": published[:4],
            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            "abstract": first_sentences(clean(entry.findtext(f"{ATOM}summary", "")), ABSTRACT_CAP),
        })
    return papers


def wiki_summary(title: str) -> dict:
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + urllib.parse.quote(title.replace(" ", "_"), safe=""))
    payload = json.loads(http_get(url))
    return {"title": payload.get("title", title),
            "text": first_sentences(clean(payload.get("extract", "")), ABSTRACT_CAP)}


def synthetic_series(rng, count, low, high) -> str:
    values = sorted(rng.uniform(low, high) for _ in range(count))
    if rng.random() < 0.5:
        values.reverse()
    return ", ".join(f"{v:.2f}" for v in values)


# ------------------------------------------------------------------------ build

# Clusters whose questions are about measured serving behaviour also get this
# repository's real series; the others rely on their own synthetic one.
MEASURED_CLUSTERS = {"prefix-caching", "compiler-runtime", "multi-tenant", "agent-evaluation"}


def build_sections(clusters, rng):
    web, papers, metrics, sources = {}, {}, dict(MEASURED), {}
    metrics_meta = {name: {"synthetic": False} for name in MEASURED}
    cluster_metrics = {}

    for position, cluster in enumerate(clusters):
        print(f"[cluster] {cluster['id']}")
        for paper in arxiv_search(cluster["arxiv"], PAPERS_PER_CLUSTER):
            paper["cluster"] = cluster["id"]
            papers.setdefault(paper["id"], paper)
        time.sleep(ARXIV_DELAY)
        for title in cluster["wiki"]:
            page = wiki_summary(title)
            if page["text"]:
                page["cluster"] = cluster["id"]
                web.setdefault(page["title"], page)
            time.sleep(WIKI_DELAY)
        name, description, count, low, high = SYNTHETIC_SPECS[position % len(SYNTHETIC_SPECS)]
        metrics[name] = f"{description}: {synthetic_series(rng, count, low, high)}"
        metrics_meta[name] = {"synthetic": True, "cluster": cluster["id"]}
        cluster_metrics[cluster["id"]] = [name] + (
            sorted(MEASURED) if cluster["id"] in MEASURED_CLUSTERS else [])

    # Fact-check statements: the opening claim of a real document, attributed to it.
    for paper in papers.values():
        sentence = first_sentences(paper["abstract"], 260)
        if len(sentence) < 80:
            continue
        sources[f"{paper['id']}-claim"] = f"{sentence} (source: arXiv:{paper['arxiv_id']}, {paper['title']})"
    for page in web.values():
        sentence = first_sentences(page["text"], 260)
        if len(sentence) < 80:
            continue
        key = re.sub(r"[^a-z0-9]+", "-", page["title"].lower()).strip("-") + "-claim"
        sources[key] = f"{sentence} (source: Wikipedia, {page['title']})"
    return web, papers, metrics, metrics_meta, sources, cluster_metrics


# The tenant ranks its indexes with this exact formula (see `rank`/`terms` in
# deep_search.jac). The rubric is chosen with it so that every source a task requires
# is one the agent's own search actually returns.
STOPWORDS = set(
    "a an and are as at be by for from has have in into is it its of on or that the their "
    "to was were with".split())
SEARCH_TOP = 3


def terms(text: str) -> set:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    return words - STOPWORDS


def rank(query: str, table: dict, top: int = SEARCH_TOP) -> list:
    scored = []
    for key, text in table.items():
        hits = len(terms(query) & terms(key + " " + text))
        scored.append((hits + 2 * len(terms(query) & terms(key)), key))
    scored.sort(reverse=True)
    return [key for score, key in scored[:top] if score > 0]


def build_tasks(clusters, web, papers, sources, cluster_metrics):
    """One task per (cluster, template) slot, with a rubric of evidence the tenant's
    own tools return for that task text."""
    paper_table = {pid: f"{p['title']} {p['abstract']}" for pid, p in papers.items()}
    page_table = {title: page["text"] for title, page in web.items()}
    paper_cluster = {pid: p["cluster"] for pid, p in papers.items()}
    page_cluster = {title: page["cluster"] for title, page in web.items()}

    tasks = []
    for index in range(N_CASES):
        cluster = clusters[index % len(clusters)]
        kind, template, need = TEMPLATES[(index // len(clusters) + index) % len(TEMPLATES)]
        metric_names = cluster_metrics[cluster["id"]][: max(1, need["metrics"])]
        text = template.format(a=cluster["pair"][0], b=cluster["pair"][1], goal=cluster["goal"],
                               topic=cluster["topic"], workload=cluster["workload"],
                               metric=metric_names[0])

        # The rubric is whatever the agent's own search returns for a query a planner
        # would write from this task — the topic and the two mechanisms it contrasts —
        # restricted to the cluster, so every required source is reachable in one hop.
        probe = cluster.get("probe") or f"{cluster['topic']} {cluster['pair'][0]} {cluster['pair'][1]}"
        found_papers = [pid for pid in rank(probe, paper_table)
                        if paper_cluster[pid] == cluster["id"]][: need["papers"]]
        found_pages = [title for title in rank(probe, page_table)
                       if page_cluster[title] == cluster["id"]][: need["pages"]]
        if not found_papers:
            # The cluster's own papers lose the global ranking to a neighbouring
            # cluster on this probe; require the ones a search inside the cluster
            # surfaces, which a refined query reaches.
            local = {pid: paper_table[pid] for pid in paper_table
                     if paper_cluster[pid] == cluster["id"]}
            found_papers = rank(probe, local)[: need["papers"]]
        rubric_sources = [f"{pid}-claim" for pid in found_papers
                          if f"{pid}-claim" in sources][: need["sources"]]
        tasks.append({
            "id": f"dr-{index:02d}-{cluster['id']}",
            "cluster": cluster["id"],
            "kind": kind,
            "task": text,
            "probe": probe,
            "rubric": {
                "papers": found_papers,
                "pages": found_pages,
                "metrics": metric_names if need["metrics"] else [],
                "sources": rubric_sources,
            },
        })
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--cache", default=CACHE_DIR)
    args = ap.parse_args()
    globals()["CACHE_DIR"] = args.cache
    rng = random.Random(SEED)

    web, papers, metrics, metrics_meta, sources, cluster_metrics = build_sections(CLUSTERS, rng)
    tasks = build_tasks(CLUSTERS, web, papers, sources, cluster_metrics)

    # A task nobody can answer is not a case: every rubric id must resolve in the pack.
    kept = []
    for task in tasks:
        rubric = task["rubric"]
        if (all(pid in papers for pid in rubric["papers"])
                and all(title in web for title in rubric["pages"])
                and all(name in metrics for name in rubric["metrics"])
                and all(key in sources for key in rubric["sources"])):
            kept.append(task)
        else:
            print(f"[drop] {task['id']}: rubric does not resolve")
    assert len(kept) == N_CASES, f"{len(kept)} tasks after validation, expected {N_CASES}"

    pack = {
        "tasks": kept,
        "web": {title: page["text"] for title, page in sorted(web.items())},
        "papers": {pid: f"{p['title']} ({p['authors']}, arXiv:{p['arxiv_id']}, {p['year']}). {p['abstract']}"
                   for pid, p in sorted(papers.items())},
        "metrics": metrics,
        "sources": sources,
        "meta": {
            "clusters": [{"id": c["id"], "topic": c["topic"]} for c in CLUSTERS],
            "metrics": metrics_meta,
            "papers": {pid: {"arxiv_id": p["arxiv_id"], "cluster": p["cluster"], "year": p["year"]}
                       for pid, p in papers.items()},
            "pages": {title: {"cluster": page["cluster"]} for title, page in web.items()},
        },
    }
    with open(args.out, "w") as handle:
        json.dump(pack, handle, indent=1, ensure_ascii=False)
    size = os.path.getsize(args.out) / 1024
    print(f"[pack] {args.out} | {len(kept)} tasks | {len(pack['papers'])} papers | "
          f"{len(pack['web'])} pages | {len(metrics)} series | {len(sources)} sources | {size:.0f} KB")


if __name__ == "__main__":
    main()
