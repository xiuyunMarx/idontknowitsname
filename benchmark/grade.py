"""Grade finished sessions for accuracy. A session is correct when the verdict its
application printed equals the expected label. Two kinds are left out of the fraction
(numerator and denominator) but still count toward throughput, JCT and TTFT:
  - a reply hit max_tokens (listed in summary.json "excluded")
  - verdict UNGRADED (BFCL question without a gold answer)
  - a BFCL session that gave up ("the information is not available", or no answer at
    all) on a question whose gold answer occurs nowhere in the frozen web cache: no
    sequence of tool calls could have answered it, so giving up is the right outcome
Inputs listed in bench_data/dirty_inputs.json are dropped as well, with the list as it
is now, so arms run with an older list are graded on the same inputs.

    python -m benchmark.grade                      # every arm under benchmark/results
    python -m benchmark.grade ours vanilla --levels
    python -m benchmark.grade --results benchmark/results_split/sweeps

Prints accuracy per arm and application pooled over every level, then the same per
prompt layout (original: the arm serves the applications' own prompts; relayout: the
arm serves the re-laid prompts), and with --levels per level. Writes the same tables to
<results>/accuracy/by_arm.csv, by_layout.csv and by_level.csv.

benchmark.workload calls verdict, max_tokens_hits and accuracy for summary.json.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BFCL_DIR = os.path.join(HERE, "applications", "bench_data", "BFCL")
BFCL_ANSWERS = os.path.join(BFCL_DIR, "BFCL_v4_web_search_answers.json")
BFCL_CACHE = os.environ.get("BF_CACHE_DIR") or os.path.join(BFCL_DIR, "web_cache")
VERDICT_RE = re.compile(r"^Verdict: (\S+)")
ANSWER_RE = re.compile(r"^Answer: (.*)")
GAVE_UP_RE = re.compile(
    r"not (available|specified|found|determined|possible|provided|mentioned|known|publicly|explicitly|"
    r"identified|resolved|accessible|present|included|listed|stated|disclosed|given|clear)"
    r"|(cannot|can't|could not|couldn't|unable to) (be )?(determine|identify|find|answer|resolve|establish|locate|verif)"
    r"|cannot be|could not be|unknown|no (information|data|result|record)|insufficient|not yet occurred|has not yet"
    r"|^\s*$|^0{1,2}/0{1,2}$|^n/?a$|^none$", re.I)
UNGRADED = "UNGRADED"
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
PROGRAMS = ["fact_check", "coding_agent", "doc_analysis", "BFCL_agent"]


def read_verdict(session_log: str) -> str:
    """The last `Verdict: X` line an application printed, or "" when it printed none."""
    verdict = ""
    with open(session_log, errors="replace") as f:
        for line in f:
            m = VERDICT_RE.match(line)
            if m:
                verdict = m.group(1)
    return verdict


def read_answer(session_log: str) -> str:
    """The `Answer:` line a BFCL session printed, or ""."""
    with open(session_log, errors="replace") as f:
        for line in f:
            m = ANSWER_RE.match(line)
            if m:
                return m.group(1).strip()
    return ""


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def date_forms(gold: str) -> List[str]:
    """Spellings a page may use for a gold date: 2024-10-25 -> October 25, 2024 / 25 October 2024 ..."""
    for pat, order in ((r"^(\d{4})-(\d{1,2})-(\d{1,2})$", "ymd"), (r"^(\d{4})/(\d{1,2})/(\d{1,2})$", "ymd"),
                       (r"^(\d{1,2})/(\d{1,2})/(\d{4})$", "mdy"), (r"^(\d{1,2})/(\d{1,2})$", "md")):
        m = re.match(pat, gold)
        if not m:
            continue
        g = m.groups()
        y, mo, d = (g if order == "ymd" else (g[2], g[0], g[1]) if order == "mdy" else ("", g[0], g[1]))
        month = MONTHS[int(mo) - 1]
        day = str(int(d))
        forms = [f"{month} {day}", f"{day} {month}"]
        if y:
            forms = [f"{month} {day} {y}", f"{day} {month} {y}", f"{y}{int(mo):02d}{int(d):02d}"]
        return forms
    return []


_REACHABLE: Dict[str, bool] = {}


def bfcl_reachable() -> Dict[str, bool]:
    """Per BFCL question id: does any gold answer (or a spelling of a gold date) occur in
    any search result or page of the frozen web cache? If not, no tool call sequence
    could have answered the question."""
    if _REACHABLE or not os.path.exists(BFCL_ANSWERS):
        return _REACHABLE
    text = []
    if os.path.isdir(BFCL_CACHE):
        for name in os.listdir(BFCL_CACHE):
            with open(os.path.join(BFCL_CACHE, name), errors="replace") as f:
                text.append(f.read())
    corpus = normalize(" ".join(text))
    with open(BFCL_ANSWERS) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            forms = []
            for g in row["ground_truth"]:
                forms += [normalize(str(g))] + date_forms(str(g))
            _REACHABLE[row["id"]] = any(x and x in corpus for x in forms)
    return _REACHABLE


def gave_up(answer: str) -> bool:
    return bool(GAVE_UP_RE.search(answer))


def verdict(program: str, session_log: str, task_input: str) -> str:
    """The session's grade: the application's own verdict, except that a BFCL session
    that gave up on a question the web cache cannot answer is UNGRADED."""
    v = read_verdict(session_log)
    if program == "BFCL_agent" and v == "FAILED" and not bfcl_reachable().get(task_input, True) \
            and gave_up(read_answer(session_log)):
        return UNGRADED
    return v


def max_tokens_hits(request_log: str) -> "set[int]":
    """Session pids with a request that generated right up to its max_new_tokens: the
    reply is a truncated JSON document, the session's outcome is not the agent's."""
    limit: Dict[str, int] = {}
    pids: "set[int]" = set()
    if not os.path.exists(request_log):
        return pids
    with open(request_log) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("event") == "request_start":
                limit[d["request_id"]] = int((d.get("sampling_params") or {}).get("max_new_tokens") or 0)
            elif d.get("event") == "request_end":
                cap = limit.get(d["request_id"], 0)
                if cap and int(d.get("output_tokens") or 0) >= cap:
                    pids.add(int(d["request_id"].split("-", 1)[0]))
    return pids


def graded(records: Iterable[dict], excluded: "set[int]" = frozenset()) -> List[dict]:  # type: ignore
    """The session records that count toward accuracy: not in `excluded` (pids that ran
    into max_tokens) and not UNGRADED."""
    return [r for r in records if r["pid"] not in excluded and r["verdict"] != UNGRADED]


def correct(record: dict) -> bool:
    return record["verdict"] == record["expected"]


def accuracy(records: Iterable[dict], excluded: "set[int]" = frozenset()) -> str:  # type: ignore
    """`correct/graded` for the given session records."""
    g = graded(records, excluded)
    return f"{sum(correct(r) for r in g)}/{len(g)}"


# ---------------------------------------------------------------- grading a results tree

def layout(arm: str) -> str:
    return "relayout" if "relayout" in arm or arm.startswith("ours") else "original"


def level_dirs(results: str, arm: str) -> List[str]:
    """Finished level directories of an arm, in both layouts: flat (<arm>/<level>) and
    nested (<arm>/per-agent/<agent>/<level>, <arm>/mixed/<level>)."""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(results, arm, "**", "summary.json"),
                                                        recursive=True))


def level_key(level: str) -> Tuple[int, int]:
    m = re.search(r"_c(\d+)$", level)
    if m:
        return (0, int(m.group(1)))
    m = re.search(r"L(\d+)$", level)
    return (1, int(m.group(1))) if m else (2, 0)


def grade_level(level_dir: str, dirty: Dict[str, "set[int]"]) -> Dict[str, Tuple[int, int]]:
    """Per program: (correct, graded) of the level's measured sessions."""
    with open(os.path.join(level_dir, "sessions.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    hits = max_tokens_hits(os.path.join(level_dir, "requests.jsonl"))
    for r in records:   # regrade from the session log, so old levels follow the current rules
        log = os.path.join(level_dir, "logs", f"{r['program']}-{r['phase']}-{r['idx']:03d}.log")
        if r["phase"] == "measured" and os.path.exists(log):
            r["verdict"] = verdict(r["program"], log, r["input"])
    out: Dict[str, Tuple[int, int]] = {}
    for p in sorted({r["program"] for r in records}):
        rs = graded([r for r in records if r["program"] == p and r["phase"] == "measured" and r["rc"] == 0
                     and r["input_idx"] not in dirty.get(p, ())], hits)
        out[p] = (sum(correct(r) for r in rs), len(rs))
    return out


def pct(c: int, n: int) -> str:
    return f"{100 * c / n:5.1f}% ({c}/{n})" if n else "-"


def print_table(title: str, rows: List[Tuple[str, Dict[str, Tuple[int, int]]]]) -> None:
    print(f"\n{title}")
    width = max(len(name) for name, _ in rows) + 2
    print(" " * width + "".join(f"{p:>22s}" for p in PROGRAMS))
    for name, cells in rows:
        print(f"{name:{width}s}" + "".join(f"{pct(*cells.get(p, (0, 0))):>22s}" for p in PROGRAMS))


def write_csv(path: str, head: List[str], rows: List[Tuple[str, Dict[str, Tuple[int, int]]]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head + [x for p in PROGRAMS for x in (f"{p}_correct", f"{p}_graded", f"{p}_accuracy")])
        for name, cells in rows:
            w.writerow(name.split("\t") + [x for p in PROGRAMS for c, n in [cells.get(p, (0, 0))]
                                           for x in (c, n, round(c / n, 4) if n else "")])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="*", help="arm directories under --results (default: all)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--levels", action="store_true", help="also print accuracy per level")
    a = ap.parse_args()
    from benchmark.workload import dirty_inputs   # workload imports this module; bind late
    dirty = dirty_inputs()
    arms = a.arms or sorted(d for d in os.listdir(a.results)
                            if level_dirs(a.results, d) and d != "mixed_workload_metrics")
    if not arms:
        print(f"no finished levels under {a.results}", file=sys.stderr)
        return 1
    reach = bfcl_reachable()
    dead = sorted(int(k.rsplit("_", 1)[1]) for k, ok in reach.items() if not ok)
    print(f"BFCL: {len(dead)}/{len(reach)} questions have no gold answer anywhere in {BFCL_CACHE}; "
          f"sessions that gave up on them are UNGRADED: {dead}")

    per_arm: Dict[str, Dict[str, List[int]]] = {}
    per_layout: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_level: List[Tuple[str, Dict[str, Tuple[int, int]]]] = []
    for arm in arms:
        tot: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for d in sorted(level_dirs(a.results, arm), key=lambda d: level_key(os.path.basename(d))):
            cells = grade_level(d, dirty)
            per_level.append((f"{arm}\t{os.path.basename(d)}", cells))
            for p, (c, n) in cells.items():
                tot[p][0] += c
                tot[p][1] += n
                per_layout[layout(arm)][p][0] += c
                per_layout[layout(arm)][p][1] += n
        per_arm[arm] = tot

    arm_rows = [(arm, {p: tuple(v) for p, v in tot.items()}) for arm, tot in per_arm.items()]
    layout_rows = [(lay, {p: tuple(v) for p, v in per_layout[lay].items()}) for lay in ("original", "relayout")
                   if lay in per_layout]
    print_table("accuracy per arm (all levels pooled; max_tokens, UNGRADED and dirty inputs dropped)", arm_rows)
    print_table("accuracy per prompt layout", layout_rows)
    if a.levels:
        print_table("accuracy per level", [(name.replace("\t", "/"), cells) for name, cells in per_level])

    out = os.path.join(a.results, "accuracy")
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "by_arm.csv"), ["arm"], arm_rows)
    write_csv(os.path.join(out, "by_layout.csv"), ["layout"], layout_rows)
    write_csv(os.path.join(out, "by_level.csv"), ["arm", "level"], per_level)
    print(f"\nwritten to {out}/by_arm.csv, by_layout.csv, by_level.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
