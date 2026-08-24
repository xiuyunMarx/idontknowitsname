# Evaluation datasets

Three case packs, 30 cases each, one per tenant in `../programs/`. Every pack is
produced by a script in `build/` from a public source, so it can be rebuilt rather
than trusted. A tenant selects its case with an index environment variable, which is
what lets a mixture experiment cycle cases the way `microbench/experiments` already
cycles its inputs.

| pack | tenant | cases | selector | size |
|---|---|---|---:|---:|
| `hover_claims_30.json` + `hover_wiki.json` | `HoVer.jac` | 30 claims | `HOVER_CLAIM_INDEX` | 12 KB + 250 KB |
| `intercode_sql_30.json` | `intercode_sql.jac` | 30 questions | `SQL_TASK_INDEX` | 328 KB |
| `deep_search_30.json` | `deep_search.jac` | 30 tasks | `DEEP_TASK_INDEX` | 141 KB |

`hover_mini_*.json`, `intercode_sql_tasks.json` and `deep_search_sources.json` are the
small hand-written fixtures the programs were developed against. They are kept as smoke
tests; point a tenant at one with its `$HOVER_CORPUS` / `$INTERCODE_TASKS` /
`$DEEP_SOURCES` override.

Selection is seeded (`SEED = 20260824` in every builder), so a rebuild reproduces the
same 30 cases from the same inputs. Which cases run together, and in what proportion,
is a question for the mixture experiments — each pack carries per-case metadata
(hop count, label, hardness, cluster, rubric) for that purpose.

---

## 1. HoVer — many-hop claim verification

**Sources.** Claims from the HoVer dev release
(`hover-nlp/hover`, `data/hover/hover_dev_release_v1.1.json`, 4000 labelled claims).
Articles from `BeIR/hotpotqa`, the 5,233,329 Wikipedia intro paragraphs of the
October 2017 dump that HoVer was annotated against — so the shipped text is the text
the gold labels were written for.

**Build.**

```bash
python build/build_hover.py            # downloads ~976 MB into --cache (default /tmp/hover_build)
```

300 candidate claims are drawn round-robin over the (hop count × label) buckets, the
corpus is scanned twice (titles first, then text for the titles that matter), and the
30 final claims are those whose entire gold chain is present with every gold sentence
inside the capped text. The scan result is cached in `--cache`, so re-running after a
change to selection or link derivation costs nothing.

**Contents.** 30 claims — 10 each at 2, 3 and 4 hops; 15 SUPPORTED, 15 NOT_SUPPORTED —
and 675 articles: their 90 gold articles plus 20 lexical distractors per claim, chosen
by title-token overlap with the claim. Article text is cut at a sentence boundary near
1500 characters (mean 291). Each claim carries `uid`, `label`, `num_hops`,
`supporting_titles` and its HotpotQA id.

**Links.** BEIR ships no link graph, so `list_links` is served by mentions: an article
links to another article in the pack when that article's title, or its title without
the disambiguating parenthesis (`Life Goes On (Fergie song)` → `Life Goes On`), occurs
in its text at a word boundary. 167 of 675 articles carry links.

**Checked after the build**, through the tenant's own tools:

- 27 of 30 claims surface a gold article from a search on the claim text alone;
- 89 of 90 gold articles come back from a search on their own title;
- 46 of 90 gold articles are mentioned by another gold article of the same claim.

Retrieval is therefore a real search problem — 675 candidates, ~22 lexically close
distractors per claim — but every claim has a reachable evidence chain.

## 2. InterCode-SQL — execution-feedback repair

**Sources.** `princeton-nlp/intercode`, `data/sql/spider/ic_spider_dev.json` (the 1034
Spider 1.0 dev questions InterCode-SQL serves) and `ic_spider_dbs.sql` (its MySQL dump
of the 19 dev databases).

**Build.**

```bash
python build/build_intercode_sql.py    # ~1.5 MB of downloads, seconds to run
```

InterCode runs those databases in MySQL; the tenant runs SQLite in memory, so the dump
is converted once: `/*! */` directives, `LOCK TABLES`, user and database statements are
dropped, backticks become double quotes (Spider has columns named `18_49_Rating_Share`
and `Official_ratings_(millions)`), `AUTO_INCREMENT` / `ENGINE=` / charset and collation
clauses are stripped, MySQL index prefix lengths (`PRIMARY KEY ("Year_awarded"(255))`)
are removed, `KEY` index lines inside `CREATE TABLE` are dropped, and `\'` becomes `''`.
Foreign-key constraints are kept.

**Filtering.** Every one of the 1034 gold queries is executed against the converted
database. 888 survive; 91 are dropped because the gold query returns no rows on this
data, and 55 because the gold answer is a single `0` or `1`, which execution match
cannot distinguish from a lucky guess. All 20 databases build cleanly, which is the
evidence that the conversion is faithful.

**Contents.** 30 questions — 8 easy, 8 medium, 7 hard, 7 extra, at most 3 per database —
over the 16 databases they need (136 setup statements). Each task carries `id` (its
index in the InterCode dev file), `db`, `question`, `gold` and `hardness`.

**Checked after the build**: every one of the 30 gold queries returns `MATCH` through
the tenant's own `grade()`, a wrong query returns `MISMATCH`, and a broken one `ERROR`.

## 3. Deep Research — supervisor routing and fan-out

No public deep-research benchmark ships offline sources for all four sub-agents, so the
pack is assembled from real documents around 12 topic clusters in LLM serving and ML
systems (prefix caching, disaggregation, batching, speculative decoding, quantization,
long context, MoE, RAG, agents, compiler-runtime co-design, multi-tenancy, evaluation).

**Sources.** Papers from the arXiv API, one relevance query per cluster: id, title,
authors, year and abstract. Background pages from the Wikipedia REST summary endpoint.
Measured series from this repository's own two-tenant trial (see the top-level
`README.md`). Synthetic series generated from the build seed, one per cluster, each
labelled *Synthetic benchmark series (seeded, not measured)* in its own text so the data
agent can see what it is holding. Fact-check statements are quoted from the fetched
abstracts and leads, each ending in its source id.

**Build.**

```bash
python build/build_deep_search.py      # ~150 API calls, rate-limited, responses cached
```

**Contents.** 30 tasks over the 12 clusters, five each from six templates (`compare`,
`evidence`, `provenance`, `tradeoff`, `survey`, `applicability`); 70 papers, 29 pages,
17 series (5 measured, 12 synthetic) and 99 fact-check statements.

**Rubric.** Every task names the evidence a correct report should cite:
`{papers, pages, metrics, sources}`. It is not chosen freely — it is what the tenant's
own `paper_search` / `web_search` return for the task's `probe` query (the cluster topic
and the two mechanisms it contrasts), restricted to that cluster. A report can therefore
be scored by string-matching the rubric ids against it, without an LLM judge, and
nothing in a rubric is unreachable.

**Checked after the build**: all 30 rubrics resolve inside the pack and every rubric
paper and page comes back from the tenant's search on the task's probe. 30 tasks require
at least one paper (22 require two), 19 require a page, 15 a metric series and 20 a
fact-check source. 15 of 30 already surface a rubric paper from the raw task text; the
rest need the sub-question a planner writes, which is the work the tenant is there to do.

---

## Provenance and licences

- HoVer — Jiang et al., *HoVer: A Dataset for Many-Hop Fact Extraction And Claim
  Verification*, Findings of EMNLP 2020. Claims derive from HotpotQA (CC BY-SA 4.0).
- Wikipedia article text (via `BeIR/hotpotqa`, October 2017 dump) — CC BY-SA.
- Spider — Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Domain Semantic Parsing and Text-to-SQL Task*, EMNLP 2018 — CC BY-SA 4.0.
  InterCode — Yang et al., *InterCode: Standardizing and Benchmarking Interactive Coding
  with Execution Feedback*, NeurIPS 2023 (code MIT); the SQL environment's data is
  derived from Spider.
- arXiv abstracts and metadata are fetched through the arXiv API and stored with their
  arXiv id; copyright remains with the authors.
- The synthetic series and the task text of the Deep Research pack are generated by
  `build/build_deep_search.py` in this repository.

## Rebuilding everything

```bash
cd benchmark/evaluation/datasets
python build/build_intercode_sql.py
python build/build_deep_search.py
python build/build_hover.py            # largest download; run it last
rm -rf /tmp/hover_build                # the 976 MB corpus parquet, once the pack exists
```

Then confirm the tenants still load and compile:

```bash
cd ../../..
jac check benchmark/evaluation/programs/*.jac
python -m utils.jac_static_parser benchmark/evaluation/programs/HoVer.jac
```
