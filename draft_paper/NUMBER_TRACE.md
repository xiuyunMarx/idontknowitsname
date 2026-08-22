# Per-number trace: every numeric literal in main.pdf

Source of truth for all system-under-test numbers: `../story.md` (line numbers as of
the current working-tree version; the one uncommitted edit to story.md is prose-only
and changes no number).

Method, two independent passes, both per-occurrence rather than per-distinct-value
(a distinct-value check can hide a stray literal behind a value that legitimately
occurs elsewhere):

1. **Rendered pass.** `pdftotext main.pdf` + regex over the extracted text: **448
   numeric occurrences**. Of these, **49 body occurrences** carry a literal that does
   not appear anywhere in story.md; each of the 49 was printed with 125 characters of
   surrounding context and classified by hand (section E).
2. **Source pass.** The same regex over `main.tex`, `sections/*.tex` and the two
   generated bar figures --- LaTeX sources contain no auto-generated numbers, since
   sections, floats and citations are `\ref`/`\cite` keys, so this pass sees exactly
   the numbers an author wrote: **582 occurrences**, of which **224** are not literally
   present in story.md. All 224 fall into the four classes enumerated in section E.

Zero untraceable numbers remain under either pass.

## A. Case study 1 (story.md lines 76-94) — exact copies

Table `tab:percomponent` (sections/eval.tex:69-84) and figure `fig_percomponent.tex`
reproduce the story.md table cell for cell. Each row of the paper table maps to one
story.md line:

| paper table row | story.md line | status |
|---|---|---|
| `normalize (entry)` 20 / 193 / 37.3 / 37.6 / 35.3 / 0.99x / n/a | 78 | exact |
| `route (visit-by-llm)` 20 / 549 / 65.8 / 65.9 / 37.8 / 1.00x / 0% | 79 | exact |
| `MathDesk.analyze` 8 / 500 / 65.8 / 66.1 / 37.8 / 1.00x / 0% | 80 | exact |
| `CodeDesk.analyze` 4 / 423 / 64.0 / 64.3 / 34.5 / 1.00x / 0% | 81 | exact |
| `DocDesk.analyze` 3 / 283 / 52.3 / 52.3 / 35.2 / 1.00x / 0% | 82 | exact |
| `WriteDesk.analyze` 5 / 294 / 52.3 / 52.5 / 36.6 / 1.00x / 0% | 83 | exact |
| `MathDesk.solve` 8 / 879 / 119.8 / 67.8 / 37.7 / 1.77x / 63% | 84 | exact |
| `CodeDesk.implement` 4 / 835 / 117.9 / 49.7 / 36.2 / 2.37x / 83% | 85 | exact |
| `DocDesk.answer` 3 / 707 / 102.3 / 62.3 / 37.1 / 1.64x / 61% | 86 | exact |
| `WriteDesk.draft` 5 / 676 / 102.0 / 50.7 / 37.9 / 2.01x / 80% | 87 | exact |
| `MathDesk.finalize` 8 / 1453 / 216.3 / 42.7 / 40.1 / 5.07x / 99% | 88 | exact |
| `CodeDesk.finalize` 4 / 1292 / 187.0 / 42.6 / 37.2 / 4.39x / 96% | 89 | exact |
| `DocDesk.finalize` 3 / 1296 / 185.0 / 41.5 / 39.2 / 4.46x / 98% | 90 | exact |
| `WriteDesk.finalize` 5 / 1258 / 171.1 / 41.9 / 39.6 / 4.09x / 98% | 91 | exact |
| **sum per request** 20 / ~3300 / 469.9 / 265.1 / 186.2 / 1.77x / 72% | 92 | exact |

Setup numbers: Qwen3-4B, RTX 3090, 20 problems, GSM8K (story.md:66); "median over 20
cold-start requests" and the Oracle replay-twice / 100% cache-hit protocol
(story.md:94); four subagents, `select=1, 4 candidates` (story.md:57,59).

Same numbers re-appear verbatim in: abstract (469.9, 265.1, 1.77x, 72%, 186.2,
5.07x), conclusion (469.9, 265.1, 72%, 5.07x), threats (72%), `gen_figures.py:ROWS`
and `SUM`, `figures/fig_percomponent.tex` labels.

## B. Case study 2 (story.md lines 107-118) — exact copies

| paper item | story.md line | status |
|---|---|---|
| `reason` 74/84, 39.3, 39.3, 1.00x, 448 -> 448 | 113 | exact |
| `integrate` 58/71, 39.4, 38.3, 1.03x, 368 -> 544 | 114 | exact |
| `finalize` 20/20, 66.8, 39.2, 1.71x, 304 -> 496 | 115 | exact |
| 20 multi-hop questions, 37 passages | 107 | exact |
| loop, 2-5 turns (`fig_react.tex`, eval "two to five turns") | 103 | exact |
| ~0.3 s retrieval gap (`fig_react.tex`, `fig:react` caption) | 101 | exact |

Abstract (66.8, 39.2, 1.71x) and conclusion (1.71x) repeat these.

## C. Derived quantities — recomputed exactly from story.md values

| paper location | claim | recomputation | check |
|---|---|---|---|
| eval.tex:100 | 37.3 ms is 2.0 ms above the 35.3 ms Oracle | 37.3 - 35.3 | = 2.0 |
| eval.tex:113 | middle stages improve 1.64-2.37x | min/max of {1.77, 2.37, 1.64, 2.01} (L84-87) | ok |
| eval.tex:113 | capture 61-83% of headroom | min/max of {63, 83, 61, 80} (L84-87) | ok |
| eval.tex:117 | terminal stages 4.09-5.07x | min/max of {5.07, 4.39, 4.46, 4.09} (L88-91) | ok |
| eval.tex:117 | capturing 96-99% | min/max of {99, 96, 98, 98} (L88-91) | ok |
| eval.tex:117 | within 2.3-5.4 ms of Oracle | 42.7-40.1=2.6; 42.6-37.2=5.4; 41.5-39.2=2.3; 41.9-39.6=2.3 | min 2.3, max 5.4 |
| eval.tex:119 | longest prompts, 1258-1453 tokens | min/max of {1453, 1292, 1296, 1258} | ok |
|  eval.tex:123 | never more than 0.3 ms slower on the six non-improving components | 0.3, 0.1, 0.3, 0.3, 0.0, 0.2 (L78-83) | max 0.3 |
|eval.tex:122 | "six components" | normalize + route + 4 analyze rows | = 6 |
|eval.tex:129 | 469.9 -> 265.1 is 1.77x | 469.9 / 265.1 = 1.7725 | rounds to 1.77, as story.md:92 reports |
|  eval.tex:130 | captures 72% of headroom | (469.9-265.1)/(469.9-186.2) = 204.8/283.7 = 0.7219 | rounds to 72%, as story.md:92 reports |
| threats.tex:34 | "20 requests each" | story.md:66 (20 problems) and :107 (20 questions) | ok |
| `fig_percomponent.tex` bar lengths (4.24-56.48 mm) | linear in ms | scale = 26.00 mm / 216.3 ms = 0.120203 mm/ms; e.g. 37.3 x 0.120203 = 4.48, 469.9 x 0.120203 = 56.48 | all rows check |
| `fig_react_bars.tex` bar lengths (14.91-26.00 mm) | linear in ms | scale = 26.00 mm / 66.8 ms = 0.389222 mm/ms; e.g. 39.3 x 0.389222 = 15.30 | all rows check |

## D. Third-party numbers — attributed to and verified against the cited paper

story.md is not the source for these; each was checked against the primary source
retrieved from arXiv (Atom API metadata for existence/title/authors; full text for
the in-body statistics). None is a measurement of this system.

| paper location | number | cited work | verification |
|---|---|---|---|
| related.tex:16-17 | 1.83x, 2.19x vs SGLang hier. radix cache | KVFlow, arXiv 2507.07400 | abstract, verbatim |
| related.tex:25 | 1.85x over LRU (dynamic), 1.26x over KVFlow (static) | PBKV, arXiv 2605.06472 | abstract, verbatim |
| related.tex:31-32 | 76-86% top-1 within 50 observed dispatches | CacheScout, arXiv 2608.14624 | full text: "reaches 76-86% top-1 accuracy within 50 observed dispatches"; also story.md:12 for the ~50 figure |
| related.tex:33 | 53-62% of prompt tokens, four workloads | CacheScout, arXiv 2608.14624 | full text: "accounts for 53-62% of all prompt tokens across the four multi-agent workloads"; also story.md:15 (the second occurrence, in the axes table, was removed with that table on 2026-08-22) |
| related.tex:119 | five frameworks, 5,399 agent programs | AgentFlow, arXiv 2607.01640 | abstract, verbatim |
| related.tex (SAGA para, added 2026-08-22) | 87% pattern-inference accuracy; within 1.31x of Belady's offline optimal on SWE-bench | SAGA, arXiv 2605.00528 v2 | full text: "Pattern inference achieves 87% accuracy in predicting workflow structure"; abstract: "achieving within 1.31x of Belady's optimal offline policy" (the 1.31x is a competitive ratio measured on SWE-bench, Table 5 / eval) |
| related.tex (Helium para, added 2026-08-22) | up to 1.56x over KVFlow | Helium, arXiv 2603.16104 | full text: "Helium outperforms KVFlow by up to 1.56x" (the abstract's "up to 1.56x over state-of-the-art agent serving systems" resolves to the KVFlow baseline) |

Pythia (arXiv 2604.25899) is cited with no numbers. All 16 bib entries resolve to a
real arXiv record with matching title and author list.

**Addendum 2026-08-22.** Related work extended with SAGA (arXiv 2605.00528) and
Helium (arXiv 2603.16104); Pythia paragraph rewritten to state its argument-content
prefill (strict memory pointers, execution-history injection, extra_body
annotations — verified against the arXiv HTML full text); PBKV row unchanged in
numbers ("a static workflow" reworded to "the static workflow in its evaluation",
matching the paper's singular benchmark). The two new third-party rows above are
the only numeric literals these edits introduced; one sentence with no numerals was
removed from eval.tex.

**Addendum 2026-08-22b.** The "Axes of difference" subsection (with the tab:axes
table) and the "Exploiting idle time" subsection were deleted outright. This
removes the IdleSpec statistics (55.6%, 5.1%, 9.1% — the former related.tex:107-109
row; IdleSpec is no longer cited anywhere) and the axes-table duplicate of the
CacheScout 53-62% figure. Line-number references and the occurrence counts in
sections E1/E2 predate the 2026-08-22 edits; the classification itself is
unchanged (no new unclassified literal exists).

## E. Every literal absent from story.md, classified

### E1. Rendered pass: the 49 body occurrences

| class | occurrences | examples |
|---|---|---|
| citation markers emitted by natbib | 16 | `[6, 14]`, `[13]`, `[11, 12]`, `[9]`, `[2]`, `[8]` |
| section numbers emitted by LaTeX | 16 | 2.1, 2.2, 3.1-3.6, 4.1-4.3, 6.1-6.5 |
| `Section 6` cross-references and page folios | 4 | resolved from `\ref{sec:related}` |
| `Qwen3-4B` with the hyphen dropped by pdftotext | 2 | rendered as `Qwen34B`; traces to story.md:66 |
| derived quantities (section C above) | 3 | 2.0; 2.3-5.4 |
| third-party results (section D above) | 11 | 1.83, 2.19, 1.85, 1.26, 76, 86, 55.6, 2.5, 5.1, 9.1, 5,399 |
| bibliography text that pdftotext orders ahead of the `References` heading | 2 | `arXiv:2505.09388`, `2025` |

Total 54 class assignments over 49 occurrences (a few occurrences belong to two
classes, e.g. `6.1` counted once as a section number and once inside `Section 6`).
Note that `50` (observed dispatches) and `53`/`62` do not appear in this list: both
are present in story.md (lines 12 and 15) as well as in the CacheScout paper.

### E2. Source pass: the 224 authored literals absent from story.md

| class | occurrences | what they are |
|---|---|---|
| bar lengths in `fig_percomponent.tex`, `fig_react_bars.tex` | 45 | derived from the ms values by the single scale factor in section C |
| LaTeX typographic constants in those two figures | 114 | bar height `1.5mm`, row skips `0.35mm`/`1.4mm`/`0.5mm`, `\midrule[0.4pt]`, legend swatch `3mm` |
| document/layout constants in `main.tex` | 17 | `10pt`, `margin=0.75in`, `columnsep=0.28in`, float fractions, `\definecolor` RGB triples, `black!55` |
| table geometry in `related.tex` | 5 | `\arraystretch{1.15}`, `p{0.125\textwidth}` and siblings |
| years inside citation keys | 29 | `\cite{kwon2023pagedattention}`, `\cite{zheng2026pbkv}`, ... --- key text, not data |
| derived quantities (section C) | 3 | 2.0; 2.3; 5.4 |
| third-party results (section D) | 11 | as listed above |

The three pure-geometry figures (`fig_motivation.tex`, `fig_topology.tex`,
`fig_react.tex`) contain only `\put`/`\framebox` coordinates in mm and were excluded
from the source pass by file; their only rendered numbers are `2--5 turns` and
`0.3\,s`, both traced in section B.

## F. Corrections applied in this pass

1. `sections/eval.tex:27-31` claimed "All reported values are median TTFT over 20
   cold-start requests", which is true of case study 1 (story.md:94) but not of case
   study 2, whose per-call-site medians are over 74/84/58/71/20 calls (story.md:113-115)
   issued by 20 questions. Rewritten to state each case study's basis separately.
2. `sections/eval.tex:100` said 37.3 ms is "within 2 ms" of the 35.3 ms Oracle; the
   difference is exactly 2.0 ms. Changed to "is 2.0 ms above".

No number was changed, and no claim was deleted: every data number already traced.
