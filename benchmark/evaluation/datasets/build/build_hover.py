"""Build the HoVer claim pack and Wikipedia corpus for benchmark/evaluation/programs/HoVer.jac.

Sources (downloaded at build time, nothing vendored by hand):
  hover-nlp/hover   data/hover/hover_dev_release_v1.1.json   4000 labelled dev claims
  BeIR/hotpotqa     corpus parquet (~976 MB)                 5.23M Wikipedia intro
                                                             paragraphs from the Oct-2017
                                                             dump HoVer was annotated on

The claims keep their gold labels because the corpus is the same vintage they were
written against. Only what the selected claims need is shipped: their gold articles
plus lexical distractors, so retrieval stays a real search problem without carrying
5 million paragraphs.

    python benchmark/evaluation/datasets/build/build_hover.py

The parquet is downloaded into --cache (default /tmp/hover_build) and is not deleted
automatically; remove that directory when the build is done.
"""

import argparse
import heapq
import json
import os
import re
import urllib.request
from collections import Counter, defaultdict

import pyarrow.compute as pc
import pyarrow.parquet as pq

HOVER_URL = ("https://raw.githubusercontent.com/hover-nlp/hover/main/"
             "data/hover/hover_dev_release_v1.1.json")
CORPUS_URL = ("https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/"
              "corpus/corpus-00000-of-00001.parquet")

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMS_OUT = os.path.join(HERE, "..", "hover_claims_30.json")
WIKI_OUT = os.path.join(HERE, "..", "hover_wiki.json")

SEED = 20260824
N_CASES = 30
N_CANDIDATES = 300        # claims considered before the gold-coverage filter
DISTRACTORS = 20          # per selected claim
TEXT_CAP = 1500           # characters, cut at a sentence boundary
LINKS_CAP = 8

STOPWORDS = set(
    "a an and are as at be been by for from has have in into is it its of on or that the "
    "their to was were with which who whose what when where while also but not this these "
    "those he she they his her them there than then over under after before both same".split())


def tokens(text: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def cap_text(text: str, limit: int = TEXT_CAP) -> str:
    if len(text) <= limit:
        return text.strip()
    kept = []
    for sentence in sentences(text):
        if sum(len(s) + 1 for s in kept) + len(sentence) > limit and kept:
            break
        kept.append(sentence)
    return " ".join(kept).strip()


# --------------------------------------------------------------------- download

def fetch(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url.split("?")[0]))
    if not os.path.exists(path):
        print(f"[fetch] {url}")
        urllib.request.urlretrieve(url, path)
    return path


# -------------------------------------------------------------------- selection

def spread(records, seed, count):
    """A seeded spread over (num_hops, label) buckets, round-robin across buckets."""
    import random

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for record in records:
        buckets[(str(record["num_hops"]), record["label"])].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    picked, index = [], 0
    while len(picked) < count and any(buckets[k] for k in keys):
        key = keys[index % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop())
        index += 1
    return picked


def gold_titles(record) -> list:
    seen = []
    for title, _sent_id in record["supporting_facts"]:
        if title not in seen:
            seen.append(title)
    return seen


# ----------------------------------------------------------------------- corpus

def scan_titles(parquet_path, claims):
    """Pass one: titles only. Returns the distractor shortlist per claim.

    A distractor is an article whose *title* shares vocabulary with the claim or with
    one of its gold titles — the articles a lexical retriever would actually confuse
    with the right one.
    """
    index = defaultdict(list)
    for position, record in enumerate(claims):
        vocab = tokens(record["claim"]) | tokens(" ".join(gold_titles(record)))
        for token in vocab:
            index[token].append(position)
    gold_all = {t for record in claims for t in gold_titles(record)}

    shortlists = [[] for _ in claims]
    counter = Counter()
    reader = pq.ParquetFile(parquet_path)
    seen_rows = 0
    for batch in reader.iter_batches(batch_size=100_000, columns=["title"]):
        for title in batch.column("title").to_pylist():
            seen_rows += 1
            if title in gold_all:
                continue
            hits = counter
            hits.clear()
            for token in tokens(title):
                for position in index.get(token, ()):  # most titles hit nothing
                    hits[position] += 1
            for position, score in hits.items():
                if score < 2:
                    continue
                shortlist = shortlists[position]
                item = (score, title)
                if len(shortlist) < DISTRACTORS * 3:
                    heapq.heappush(shortlist, item)
                elif item > shortlist[0]:
                    heapq.heapreplace(shortlist, item)
        if seen_rows % 1_000_000 < 100_000:
            print(f"[scan] {seen_rows:,} titles")
    print(f"[scan] {seen_rows:,} titles scanned")
    return shortlists


def fetch_articles(parquet_path, wanted: set) -> dict:
    """Pass two: decode text only for the titles that made the cut."""
    import pyarrow as pa

    value_set = pa.array(sorted(wanted), type=pa.large_string())
    out = {}
    reader = pq.ParquetFile(parquet_path)
    for batch in reader.iter_batches(batch_size=100_000, columns=["title", "text"]):
        mask = pc.is_in(batch.column("title"), value_set=value_set)
        if not pc.any(mask).as_py():
            continue
        table = batch.filter(mask)
        for title, text in zip(table.column("title").to_pylist(),
                               table.column("text").to_pylist()):
            out.setdefault(title, text)
    return out


DISAMBIGUATION = re.compile(r"\s*\([^)]*\)\s*$")


def title_variants(title: str) -> list:
    """The surface forms an article is referred to by. Wikipedia titles carry a
    disambiguator the running text drops: `Life Goes On (Fergie song)` is written as
    `Life Goes On`."""
    base = DISAMBIGUATION.sub("", title).strip()
    variants = [title]
    if base != title and len(base) >= 4:
        variants.append(base)
    return variants


def build_links(corpus: dict) -> dict:
    """Link graph for `list_links`: an article links to another article in the pack when
    that article's title (or its undisambiguated form) is mentioned in its text. BEIR
    ships no link structure, and a mention is the hop a many-hop claim actually
    follows."""
    forms = []
    for title in corpus:
        for variant in title_variants(title):
            forms.append((variant, title,
                          re.compile(rf"(?<![A-Za-z0-9]){re.escape(variant)}(?![A-Za-z0-9])")))
    links = {}
    for title, text in corpus.items():
        found, seen = [], {title}
        for variant, target, pattern in forms:
            if target in seen or variant not in text:
                continue
            match = pattern.search(text)
            if match is None:
                continue
            found.append((match.start(), target))
            seen.add(target)
        found.sort()
        links[title] = [name for _pos, name in found[:LINKS_CAP]]
    return links


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("BUILD_CACHE", "/tmp/hover_build"))
    ap.add_argument("--claims-out", default=CLAIMS_OUT)
    ap.add_argument("--wiki-out", default=WIKI_OUT)
    args = ap.parse_args()

    dev = json.load(open(fetch(HOVER_URL, args.cache)))
    print(f"[hover] {len(dev)} dev claims")

    candidates = spread(dev, SEED, N_CANDIDATES)
    # The corpus scan is two passes over 5.2M rows; cache its result so a change to
    # selection, capping or link derivation does not pay for it again.
    scan_cache = os.path.join(args.cache, f"hover_scan_{SEED}_{N_CANDIDATES}.json")
    if os.path.exists(scan_cache):
        cached = json.load(open(scan_cache))
        shortlists = [[tuple(item) for item in row] for row in cached["shortlists"]]
        articles = cached["articles"]
        print(f"[scan] reusing {scan_cache}: {len(articles):,} articles")
    else:
        parquet_path = fetch(CORPUS_URL, args.cache)
        shortlists = scan_titles(parquet_path, candidates)
        wanted = {t for record in candidates for t in gold_titles(record)}
        for shortlist in shortlists:
            wanted.update(title for _score, title in shortlist)
        print(f"[corpus] fetching {len(wanted):,} articles")
        articles = fetch_articles(parquet_path, wanted)
        print(f"[corpus] {len(articles):,} found")
        json.dump({"shortlists": shortlists, "articles": articles}, open(scan_cache, "w"))

    # Keep only claims whose whole gold chain is present, with every gold sentence
    # still inside the capped text — a claim we cannot ground is not a usable case.
    usable = []
    for record, shortlist in zip(candidates, shortlists):
        titles = gold_titles(record)
        if any(t not in articles for t in titles):
            continue
        needed = defaultdict(int)
        for title, sent_id in record["supporting_facts"]:
            needed[title] = max(needed[title], int(sent_id))
        if any(len(sentences(cap_text(articles[t]))) <= needed[t] for t in titles):
            continue
        usable.append((record, shortlist))
    print(f"[select] {len(usable)}/{len(candidates)} candidates fully grounded")

    chosen = spread([record for record, _ in usable], SEED, N_CASES)
    chosen_ids = {record["uid"] for record in chosen}
    shortlist_of = {record["uid"]: shortlist for record, shortlist in usable}

    corpus_titles = set()
    claims = []
    for record in chosen:
        titles = gold_titles(record)
        corpus_titles.update(titles)
        ranked = sorted(shortlist_of[record["uid"]], reverse=True)[:DISTRACTORS]
        corpus_titles.update(title for _score, title in ranked)
        claims.append({
            "uid": record["uid"],
            "claim": record["claim"],
            "label": record["label"],
            "num_hops": int(record["num_hops"]),
            "supporting_titles": titles,
            "hpqa_id": record["hpqa_id"],
        })

    corpus = {title: cap_text(articles[title]) for title in sorted(corpus_titles)
              if title in articles}
    links = build_links(corpus)
    wiki = {title: {"text": text, "links": links[title]} for title, text in corpus.items()}

    json.dump(claims, open(args.claims_out, "w"), indent=1, ensure_ascii=False)
    json.dump(wiki, open(args.wiki_out, "w"), indent=1, ensure_ascii=False)

    linked = sum(1 for v in wiki.values() if v["links"])
    reachable = 0
    for claim in claims:
        titles = claim["supporting_titles"]
        pool = set()
        for title in titles:
            pool.update(wiki[title]["links"])
        reachable += sum(1 for t in titles[1:] if t in pool) / max(1, len(titles) - 1)
    print(f"[pack] {args.claims_out} | {len(claims)} claims | "
          f"hops {dict(Counter(c['num_hops'] for c in claims))} | "
          f"labels {dict(Counter(c['label'] for c in claims))}")
    print(f"[pack] {args.wiki_out} | {len(wiki)} articles | "
          f"{os.path.getsize(args.wiki_out) / 1024:.0f} KB | "
          f"{linked} with links | mean gold-chain link reachability "
          f"{reachable / len(claims):.2f}")
    assert len(chosen_ids) == len(claims) == N_CASES


if __name__ == "__main__":
    main()
