#!/usr/bin/env python3
"""Paired permutation (sign-flip) test over inputs: original-prompt arms vs re-laid arms.

For each workload and each input idx, take the mean of the metric across the five
original-prompt arms and across the five re-laid arms, form the paired difference,
and sign-flip test the differences. Mirrors the Table 2 (tab:quality) methodology.
"""
import json, os, sys, random, collections

BASE = "/home/xiaoyu/idontknowitsname/benchmark/mixed_results/mix4x4_h10"
ORIG = ["vanilla", "continuum", "cachescout", "kvflow", "kvonly"]
RELAID = ["relayout", "continuum_relayout", "cachescout_relayout", "kvflow_relayout", "ours"]
PROGS = {"fact_check": "Fact check", "coding_agent": "Coding", "BFCL_agent": "BFCL", "doc_analysis": "Finance"}

def load_arm(arm):
    sess = {}
    for line in open(os.path.join(BASE, arm, "sessions.jsonl")):
        r = json.loads(line)
        if r.get("phase") != "measured":
            continue
        key = (r["program"], r["idx"])
        sess[key] = {"correct": 100.0 * (r["verdict"] == r["expected"]),
                     "rounds": float(r["rounds"]),
                     "pid": str(r["pid"]), "rc": r.get("rc", 0)}
    out_tok = collections.Counter()
    for line in open(os.path.join(BASE, arm, "requests.jsonl")):
        r = json.loads(line)
        if r["event"] == "request_end":
            sid = r.get("session_id") or r["request_id"].split("-")[0]
            out_tok[sid] += r.get("output_tokens", 0) or 0
    for key, s in sess.items():
        s["output_tokens"] = float(out_tok.get(s["pid"], 0))
    return sess

arms = {a: load_arm(a) for a in ORIG + RELAID}

def sign_flip_p(diffs, n=100000, seed=0):
    obs = sum(diffs)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s) >= abs(obs) - 1e-12:
            hits += 1
    return hits / n

print(f"{'Workload':<11} {'metric':<14} {'orig':>8} {'relaid':>8} {'delta':>8} {'p':>7}  n")
for prog, label in PROGS.items():
    keys = set.intersection(*(set(k for k in arms[a] if k[0] == prog) for a in ORIG + RELAID))
    keys = sorted(keys)
    for metric in ["correct", "rounds", "output_tokens"]:
        mo = [sum(arms[a][k][metric] for a in ORIG) / len(ORIG) for k in keys]
        mr = [sum(arms[a][k][metric] for a in RELAID) / len(RELAID) for k in keys]
        diffs = [r - o for o, r in zip(mo, mr)]
        p = sign_flip_p(diffs)
        avg_o, avg_r = sum(mo) / len(mo), sum(mr) / len(mr)
        print(f"{label:<11} {metric:<14} {avg_o:>8.2f} {avg_r:>8.2f} {avg_r-avg_o:>+8.2f} {p:>7.2f}  {len(keys)}")
