"""Structural estimate, under an infinite KV cache, of how many prompt tokens are
identical to content served earlier but cannot be reused by prefix matching
because an earlier component (site header or a changed binding) differs.

Prompt model (byLLM, declaration order):  S + H_site + bindings...  (no tail).
Per call:  P (logged), LCP = longest prefix shared with an earlier prompt,
           Blocked = identical-but-not-prefix tokens, New = P - LCP - Blocked.
Lengths come from the vanilla server logs ([serve]/[decode]/[call]) and from
tokenizing the session inputs (claim / HumanEval spec / FinanceBench question).
"""
import glob, json, re, sys, os
from collections import defaultdict
from transformers import AutoTokenizer

B = "/home/xiaoyu/idontknowitsname/benchmark"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
T = lambda s: len(tok(s)["input_ids"])

SERVE = re.compile(r"^\[serve\] (\d+)-(\d+)t(\d+)-\S+ prompt_tokens=(\d+) cached_tokens=(\d+)")
DECODE = re.compile(r"^\[decode\] (\d+)-(\d+)t(\d+)-\S+ decode_ms=[\d.]+ output_tokens=(\d+)")
CALL = re.compile(r"^\[call\] (\d+) #(\d+) (\S+) ctx=\(([^)]*)\)")

# S + H_site (+ the "name = '" opener of the first binding): plateau of vanilla hits
C = {"Agent.step": 344, "Extractor.extract": 143, "Analyst.compute": 216, "Auditor.check": 266,
     "Writer.answer": 303, "ClaimAnalyzer.decompose_claim": 428, "EvidenceScout.assess_evidence": 418,
     "EvidenceAnalyzer.verify_claim": 367, "Planner.plan": 255, "Coder.implement": 344,
     "Analyzer.diagnose": 255, "Committer.commit": 214}

humaneval = {json.loads(l)["task_id"]: json.loads(l)["prompt"]
             for l in open(f"{B}/applications/bench_data/HumanEval/HumanEval.jsonl")}
finq = {json.loads(l)["financebench_id"]: json.loads(l)["question"]
        for l in open(f"{B}/applications/bench_data/finicial_bench/financebench_open_source.jsonl")}


def parse(log):
    serve, out, call = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for line in open(log, errors="replace"):
        m = SERVE.match(line)
        if m:
            serve[int(m[1])][(int(m[2]), int(m[3]))] = (int(m[4]), int(m[5])); continue
        m = DECODE.match(line)
        if m:
            out[int(m[1])][(int(m[2]), int(m[3]))] = int(m[4]); continue
        m = CALL.match(line)
        if m:
            ctx = tuple(int(x) for x in m[4].replace(" ", "").split(",") if x)
            call[int(m[1])][int(m[2]) + 1] = (m[3].split(":")[0], ctx)
    sessions = {}
    for pid, calls in call.items():
        seq = []
        for c in sorted(calls):
            if (c, 0) not in serve[pid]:
                continue
            P, H = serve[pid][(c, 0)]
            turns = [t for (cc, t) in out[pid] if cc == c]
            o = out[pid][(c, max(turns))] if turns else 0
            seq.append(dict(site=calls[c][0], ctx=calls[c][1], P=P, hit=H, out=o))
        sessions[pid] = seq
    return sessions


def rec(site, P, lcp, blocked):
    lcp = min(lcp, P)
    blocked = max(0, min(blocked, P - lcp))
    return dict(site=site, P=P, lcp=lcp, blocked=blocked, new=P - lcp - blocked)


def sim_doc(seq, question):
    sites = [c["site"] for c in seq]
    if sites != ["Extractor.extract", "Analyst.compute", "Auditor.check", "Writer.answer"]:
        return None
    P = [c["P"] for c in seq]
    Q = T(question)
    D = P[0] - C["Extractor.extract"] - Q
    return [rec(sites[0], P[0], C[sites[0]], 0),
            rec(sites[1], P[1], C[sites[1]], P[0] - C["Extractor.extract"]),          # Q + D
            rec(sites[2], P[2], C[sites[2]], P[1] - C["Analyst.compute"]),            # Q + D + F
            rec(sites[3], P[3], C[sites[3]], P[2] - C["Auditor.check"] - D)]          # Q + F + A


def sim_bfcl(seq):
    if any(c["site"] != "Agent.step" for c in seq):
        return None
    P = [c["P"] for c in seq]; O = [c["out"] for c in seq]; hit = [c["hit"] for c in seq]
    res = [rec("Agent.step", P[0], 411, 0)]                     # S+H+obs''+tools+"question = '"
    for k in range(1, len(P)):
        lcp = max(C["Agent.step"], hit[k])
        blocked = P[0] - C["Agent.step"] if k == 1 else P[k - 2] + O[k - 2] - C["Agent.step"]
        res.append(rec("Agent.step", P[k], lcp, blocked))
    return res


def sim_fact(seq, claim):
    Cl = T(claim)
    res, P_dc, P_vf = [], {}, {}
    for c in seq:
        s, ctx, P, hit = c["site"], c["ctx"], c["P"], c["hit"]
        r = ctx[0] if ctx else 1
        if s == "ClaimAnalyzer.decompose_claim":
            if r == 1:
                res.append(rec(s, P, C[s], 0))
            else:
                prev = P_dc.get(r - 1, P)
                res.append(rec(s, P, prev, P - prev))                 # new findings, seen in verify_{r-1}
            P_dc[r] = P
        elif s == "EvidenceScout.assess_evidence":
            ident = Cl + 5 if r == 1 else P_vf.get(r - 1, C["EvidenceAnalyzer.verify_claim"]) - C["EvidenceAnalyzer.verify_claim"]
            res.append(rec(s, P, max(C[s], hit), ident))             # claim + evidences
        elif s == "EvidenceAnalyzer.verify_claim":
            lcp = C[s] if r == 1 else P_vf.get(r - 1, C[s])
            res.append(rec(s, P, lcp, 0))                            # new findings only
            P_vf[r] = P
        else:
            return None
    return res


def sim_code(seq, task_ids):
    specs = {i + 1: T(humaneval[t]) for i, t in enumerate(task_ids)}
    res, P_pl, P_im, P_dg, O_dg, last_i = [], {}, {}, {}, {}, 0
    for c in seq:
        s, ctx, P, O = c["site"], c["ctx"], c["P"], c["out"]
        if s == "Planner.plan":
            i = ctx[0]; last_i = i; P_pl[i] = P
            blocked = 0 if i == 1 or (i - 1) not in P_pl else P_pl[i - 1] - C[s] - specs.get(i - 1, 0) - 5
            res.append(rec(s, P, C[s], blocked))                      # module_{i-1}
        elif s == "Coder.implement":
            i, t = ctx; P_im[(i, t)] = P
            if t == 1:
                blocked = P_pl.get(i, C["Planner.plan"]) - C["Planner.plan"] - 5           # spec + module
            else:
                blocked = P_im.get((i, t - 1), C[s]) - C[s] - (O_dg.get((i, t - 2), 0) if t >= 3 else 0)
            res.append(rec(s, P, C[s], blocked))
        elif s == "Analyzer.diagnose":
            i, t = ctx; P_dg[(i, t)] = P; O_dg[(i, t)] = O
            blocked = (P_pl.get(i, C["Planner.plan"]) - C["Planner.plan"] - 5) if t == 1 else P_dg.get((i, t - 1), C[s]) - C[s]
            res.append(rec(s, P, C[s], blocked))
        elif s == "Committer.commit":
            blocked = P_pl.get(last_i, C["Planner.plan"]) - C["Planner.plan"] - specs.get(last_i, 0) - 5   # module_n
            res.append(rec(s, P, C[s], blocked))
        else:
            return None
    return res


def main(runs):
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))   # prog -> site -> [n, P, lcp, blocked, new]
    nsess = defaultdict(int)
    for run in runs:
        sessions = parse(f"{run}/server.log")
        meta = [json.loads(l) for l in open(f"{run}/sessions.jsonl")]
        seen = set()
        for s in meta:
            key = (s["program"], s["input"])
            if key in seen or s["pid"] not in sessions or s["rc"] != 0:
                continue
            seen.add(key)
            seq = sessions[s["pid"]]
            prog = s["program"]
            if prog == "doc_analysis":
                r = sim_doc(seq, finq[s["input"]])
            elif prog == "BFCL_agent":
                r = sim_bfcl(seq)
            elif prog == "fact_check":
                r = sim_fact(seq, s["input"])
            elif prog == "coding_agent":
                r = sim_code(seq, s["input"].split(","))
            else:
                r = None
            if not r:
                continue
            nsess[prog] += 1
            for c in r:
                a = agg[prog][c["site"]]
                a[0] += 1; a[1] += c["P"]; a[2] += c["lcp"]; a[3] += c["blocked"]; a[4] += c["new"]
    print(f"{'program / site':38s} {'calls':>5s} {'P/call':>7s} {'LCP':>6s} {'blocked':>8s} {'new':>6s} | blocked/P  blocked/(P-LCP)")
    for prog in ["doc_analysis", "BFCL_agent", "fact_check", "coding_agent"]:
        tot = [0] * 5
        for site, a in agg[prog].items():
            for i in range(5): tot[i] += a[i]
            n = a[0]
            print(f"  {site:36s} {n:5d} {a[1]/n:7.0f} {a[2]/n:6.0f} {a[3]/n:8.0f} {a[4]/n:6.0f} | {100*a[3]/a[1]:6.1f}%   {100*a[3]/(a[1]-a[2]):6.1f}%")
        n = tot[0]
        print(f"{prog:38s} {n:5d} {tot[1]/n:7.0f} {tot[2]/n:6.0f} {tot[3]/n:8.0f} {tot[4]/n:6.0f} | {100*tot[3]/tot[1]:6.1f}%   {100*tot[3]/(tot[1]-tot[2]):6.1f}%   (sessions={nsess[prog]})")
        print()


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(glob.glob(f"{B}/mixed_results/mix4x*_h10/vanilla"))
    print("runs:", [os.path.relpath(r, B) for r in runs])
    main(runs)
