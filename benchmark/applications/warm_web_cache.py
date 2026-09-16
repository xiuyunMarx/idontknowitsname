"""Re-fetch every failed tool result in a program's web cache, one request per second.

    python -m benchmark.applications.warm_web_cache [--program BFCL_agent|fact_check] [--interval 1.0]

A failed entry starts with "Search failed for '<query>'", "Fetch failed for
'<title>'" or "Wikipedia search failed for '<query>'"; the query is taken from
that text, fetched again through the program's own tool function, and the entry
is overwritten once it succeeds.
"""
import argparse
import hashlib
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILED = re.compile(r"^(Search|Fetch|Wikipedia search) failed for '(.*)': ", re.S)
RETRIES = 6
DATA = os.path.join(REPO, "benchmark", "applications", "bench_data")
CACHES = {"BFCL_agent": os.path.join(DATA, "BFCL", "web_cache"), "fact_check": os.path.join(DATA, "HoVer", "wiki_cache")}


def fetcher(program: str):
    sys.path.insert(0, os.path.join(REPO, "static_analysis"))
    from jaclang import JacRuntime as Jac
    (an,) = Jac.jac_import("analyzer", base_path=os.path.join(REPO, "static_analysis"))
    _, mod = an.compile_guarded(os.path.join(REPO, "benchmark", "applications", f"{program}.jac"))
    if program == "BFCL_agent":
        t = mod.AgentTools()
        return lambda kind, key: t.wiki_search(key) if kind == "search" else t.wiki_page(key)
    scout = mod.EvidenceScout(slot=0)
    return lambda kind, key: scout.fetch(key)


def cache_name(program: str, kind: str, key: str) -> str:
    raw = f"{kind}:{key}" if program == "BFCL_agent" else key
    return hashlib.md5(raw.encode()).hexdigest() + ".txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="BFCL_agent", choices=list(CACHES))
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    cache = CACHES[a.program]
    todo = []
    for name in sorted(os.listdir(cache)):
        with open(os.path.join(cache, name)) as f:
            m = FAILED.match(f.read())
        if m:
            kind = "fetch" if m.group(1) == "Fetch" else "search"
            key = m.group(2)
            assert cache_name(a.program, kind, key) == name, name
            todo.append((name, kind, key))
    print(f"{len(todo)} failed entries to refetch", flush=True)
    fetch = fetcher(a.program)
    fixed = 0
    for i, (name, kind, key) in enumerate(todo):
        for attempt in range(RETRIES):
            text = fetch(kind, key)
            if not FAILED.match(text):
                with open(os.path.join(cache, name), "w") as f:
                    f.write(text)
                fixed += 1
                break
            time.sleep(a.interval * 2 ** (attempt + 1))
        print(f"[{i + 1}/{len(todo)}] {kind} {key[:60]!r}: {'ok' if not FAILED.match(text) else 'still failing'}", flush=True)
        time.sleep(a.interval)
    print(f"{fixed}/{len(todo)} refetched", flush=True)
    return 0 if fixed == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
