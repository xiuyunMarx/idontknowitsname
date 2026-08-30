"""Shared helpers for the multi-model workflow survey: paths, cached HTTP, canonical hashing."""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # survey/
RAW = ROOT / "data" / "raw"
PARSED = ROOT / "data" / "parsed"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"

UA = "multi-model-workflow-survey/0.1 (research; contact via repo)"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def get(url, retries=5, timeout=60, sleep=0.0, headers=None):
    """GET with retry/backoff. Returns bytes. Raises on final failure."""
    hdrs = {"User-Agent": UA, "Accept": "application/json, text/plain, */*"}
    if headers:
        hdrs.update(headers)
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        hdrs["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if sleep:
                time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and "github" in url:
                reset = e.headers.get("X-RateLimit-Reset")
                wait = max(5, int(reset) - int(time.time()) + 1) if reset else 60
                log(f"rate limited on {url}; sleeping {wait}s")
                time.sleep(min(wait, 3600))
                continue
            if e.code == 404:
                raise
            log(f"HTTP {e.code} on {url} (attempt {attempt + 1})")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"error on {url}: {e} (attempt {attempt + 1})")
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError(f"failed after {retries} attempts: {url}")


def get_json(url, **kw):
    return json.loads(get(url, **kw).decode("utf-8"))


def github_tree(repo, branch="main"):
    """All paths in a repo (one API call)."""
    d = get_json(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")
    return [t["path"] for t in d.get("tree", []) if t.get("type") == "blob"]


def github_raw(repo, path, branch="main"):
    return get(f"https://raw.githubusercontent.com/{repo}/{branch}/{urllib.parse.quote(path)}")


def write_if_missing(path: Path, fetch):
    """fetch() -> bytes; only called when path does not exist. Returns True if fetched."""
    if path.exists() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fetch())
    return True


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def graph_hash(node_types, edges):
    """Canonical structural hash: sorted multiset of node types + sorted typed edges.
    node_types: list of (type, version) ; edges: list of (src_type, dst_type, kind)."""
    a = sorted(f"{t}@{v}" for t, v in node_types)
    b = sorted(f"{s}->{d}:{k}" for s, d, k in edges)
    return sha1("|".join(a) + "||" + "|".join(b))


def longest_site_path(node_ids, edges, site_ids):
    """Longest path (counting only nodes in site_ids) over directed edges; back edges ignored."""
    adj = {}
    for s, d in edges:
        if s is None or d is None:
            continue
        adj.setdefault(s, []).append(d)
    memo, onstack = {}, set()
    sys.setrecursionlimit(10000)

    def best(u):
        if u in memo:
            return memo[u]
        if u in onstack:
            return 0
        onstack.add(u)
        m = 0
        for v in adj.get(u, []):
            m = max(m, best(v))
        onstack.discard(u)
        memo[u] = m + (1 if u in site_ids else 0)
        return memo[u]

    return max((best(u) for u in node_ids if u is not None), default=0)
