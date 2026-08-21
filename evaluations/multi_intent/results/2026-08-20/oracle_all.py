#!/usr/bin/env python3
"""Oracle (100% KV hit) TTFT for every call site: size a prompt to match the real
workload, serve it once to populate the cache, then serve it again and read that."""
import json, sys, pathlib
sys.path.insert(0, "/home/xiaoyu/proactive_prefill")
import requests
from transformers import AutoTokenizer
from runtime.side_runtime import SideRuntime

U = "http://localhost:8964"
rt = SideRuntime("/home/xiaoyu/proactive_prefill/evaluations/multi_intent/main.jac", type_check=False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
n = lambda s: len(tok.encode(s or "", add_special_tokens=False))

# real binding sizes in this workload, from each call's declared max_tokens
SIZE = {"request": 60, "profile": 96, "ticket": 96, "question": 96, "problem": 96,
        "analysis": 420, "solution": 560, "implementation": 560, "drafted": 560, "findings": 560,
        "snippets": 100, "reference": 100, "passages": 100, "style": 100,
        "review_refs": 100, "audit": 100, "recheck": 100, "audit_refs": 100}
base = " ".join(f"the quick brown fox number {i} jumps over the lazy dog" for i in range(500))
def words(ntok):
    lo, hi = 0, len(base)
    while lo < hi:
        m = (lo + hi) // 2
        if n(base[:m]) < ntok: lo = m + 1
        else: hi = m
    return base[:lo]

requests.post(f"{U}/reset", timeout=120).raise_for_status()
rows = []
for key, fn in rt.byllm_callsites.items():
    if fn.decl.kind == "visit":
        p = {"here": words(120), "candidates": words(200), "walker": words(120)}
    else:
        p = {q["name"]: words(SIZE.get(q["name"], 100)) for q in fn.decl.params}
    inv = n(fn.invariant_system) + n(fn.invariant_user_prefix)
    ptok = inv + sum(n(repr(v)) for v in p.values())
    body = {"key": key, "params": {k: repr(v) for k, v in p.items()}}
    t = [requests.post(f"{U}/call", json=body, timeout=900).json()["ttft"] * 1000 for _ in range(3)]
    rows.append({"key": key, "prompt_tokens": ptok, "cold": t[0], "hit1": t[1], "hit2": t[2]})
    print(f"{key:<24}{ptok:>6} tok   cold {t[0]:7.1f}ms   hit {t[1]:6.1f} / {t[2]:6.1f}ms", flush=True)
pathlib.Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
print("DONE")
