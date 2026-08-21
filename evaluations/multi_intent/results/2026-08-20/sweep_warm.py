#!/usr/bin/env python3
"""Oracle ceiling sweep: reset -> pass1 (cold, fills KV) -> pass2 (full KV hit, measured).
Stats are not cleared between passes, so calls[:5] is pass1 and calls[-5:] is pass2."""
import json, os, pathlib, subprocess, sys, time
import requests

APP = pathlib.Path("/home/xiaoyu/proactive_prefill/evaluations/multi_intent")
URL = "http://localhost:8964"
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cases = [json.loads(l) for l in (APP / "data/sessions.jsonl").read_text().splitlines() if l.strip()][:n]

def run(req):
    env = {**os.environ, "REQUEST": req, "JAC_ROUTE_CACHE_LAYOUT": "1"}
    return subprocess.run([sys.executable, "-m", "jaclang", "run", "main.jac"], cwd=APP, env=env,
                          capture_output=True, text=True, timeout=900)

for i, c in enumerate(cases):
    requests.post(f"{URL}/reset", timeout=120).raise_for_status()
    t0 = time.time()
    p1 = run(c["request"])           # cold pass: fills the KV cache
    p2 = run(c["request"])           # warm pass: whole prompt already cached
    st = requests.get(f"{URL}/stats", timeout=120).json()
    calls = st["stats"]["calls"]
    half = len(calls) // 2
    cold, warm = calls[:half], calls[half:]
    (out / f"{c['sid']}.json").write_text(json.dumps(
        {"case": c, "wall": time.time() - t0, "rc": [p1.returncode, p2.returncode],
         "n_calls": len(calls), "cold": cold, "warm": warm,
         "stdout1": p1.stdout[-4000:], "stdout2": p2.stdout[-4000:], "stats": st}))
    sc = sum(x["ttft"] or 0 for x in cold) * 1000
    sw = sum(x["ttft"] or 0 for x in warm) * 1000
    print(f"[{i+1}/{len(cases)}] {c['sid']} {c['desk']:<5} rc={p1.returncode}{p2.returncode} n={len(calls)} "
          f"cold SumTTFT={sc:7.1f}ms  WARM SumTTFT={sw:7.1f}ms  cached(warm)={[x['cached_tokens'] for x in warm]}", flush=True)
print("DONE", out)
