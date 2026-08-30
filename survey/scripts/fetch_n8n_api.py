"""Fetch the n8n public template library (category AI) via api.n8n.io.

  --index    : page through the search index -> data/raw/n8n_api/index.jsonl
  --details  : fetch full workflow JSON for every index entry that has a langchain node
               -> data/raw/n8n_api/workflows.jsonl (append, resumable)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, get_json, log, read_jsonl  # noqa: E402

OUT = RAW / "n8n_api"
INDEX = OUT / "index.jsonl"
DETAILS = OUT / "workflows.jsonl"
SEARCH = "https://api.n8n.io/api/templates/search?page={page}&rows=250&category=AI"
DETAIL = "https://api.n8n.io/api/templates/workflows/{id}"


def fetch_index():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    page = 1
    total = None
    while True:
        d = get_json(SEARCH.format(page=page), sleep=0.3)
        total = d.get("totalWorkflows", total)
        wfs = d.get("workflows", [])
        if not wfs:
            break
        for w in wfs:
            if w["id"] in seen:
                continue
            seen.add(w["id"])
            rows.append({
                "id": w["id"], "name": w.get("name"), "totalViews": w.get("totalViews"),
                "createdAt": w.get("createdAt"), "price": w.get("price"),
                "nodes": [n.get("name") for n in w.get("nodes", [])],
                "user": (w.get("user") or {}).get("username"),
            })
        log(f"index page {page}: {len(wfs)} rows, {len(rows)} total (server says {total})")
        page += 1
        if total and len(rows) >= total:
            break
    with open(INDEX, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {len(rows)} index rows to {INDEX}")


def wants_detail(row):
    return any((n or "").startswith("@n8n/n8n-nodes-langchain.") for n in row["nodes"])


def detail_files():
    return sorted(OUT.glob("workflows*.jsonl"))


def fetch_details(sleep=0.3, limit=None, shard=0, nshards=1):
    have = set()
    for f in detail_files():
        for r in read_jsonl(f):
            have.add(r["id"])
    todo = [r for r in read_jsonl(INDEX) if wants_detail(r) and r["id"] not in have]
    todo = [r for i, r in enumerate(todo) if i % nshards == shard]
    if limit:
        todo = todo[:limit]
    log(f"shard {shard}/{nshards}: {len(have)} already fetched; {len(todo)} to fetch")
    n_ok = n_err = 0
    out = DETAILS if nshards == 1 else OUT / f"workflows.shard{shard}.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            try:
                d = get_json(DETAIL.format(id=r["id"]), sleep=sleep)
                wf = (d.get("workflow") or {})
                inner = wf.get("workflow") or {}
                f.write(json.dumps({
                    "id": r["id"], "name": wf.get("name") or r["name"],
                    "totalViews": r.get("totalViews"), "createdAt": r.get("createdAt"),
                    "nodes": inner.get("nodes", []), "connections": inner.get("connections", {}),
                    "meta": inner.get("meta", {}),
                }, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_err += 1
                log(f"detail {r['id']} failed: {e}")
                f.write(json.dumps({"id": r["id"], "name": r["name"], "error": str(e)}) + "\n")
                f.flush()
            if i % 100 == 0:
                log(f"details {i}/{len(todo)} ok={n_ok} err={n_err}")
    log(f"details done ok={n_ok} err={n_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.index:
        fetch_index()
    if a.details:
        fetch_details(sleep=a.sleep, limit=a.limit, shard=a.shard, nshards=a.nshards)
