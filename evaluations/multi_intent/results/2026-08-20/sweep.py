#!/usr/bin/env python3
"""Cold-start TTFT sweep: /reset -> one jac run -> capture /stats, per case."""
import json, os, pathlib, subprocess, sys, time
import requests

APP = pathlib.Path("/home/xiaoyu/proactive_prefill/evaluations/multi_intent")
URL = "http://localhost:8964"
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cases = [json.loads(l) for l in (APP / "data/sessions.jsonl").read_text().splitlines() if l.strip()][:n]

for i, c in enumerate(cases):
    r = requests.post(f"{URL}/reset", timeout=120); r.raise_for_status()
    env = {**os.environ, "REQUEST": c["request"], "JAC_ROUTE_CACHE_LAYOUT": "1"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "jaclang", "run", "main.jac"], cwd=APP, env=env,
                       capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    st = requests.get(f"{URL}/stats", timeout=120).json()
    (out / f"{c['sid']}.json").write_text(json.dumps(
        {"case": c, "wall": wall, "rc": p.returncode, "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:], "stats": st}))
    calls = st["stats"]["calls"]
    s = sum(x["ttft"] or 0 for x in calls) * 1000
    warms = [w for w in st["stats"]["warms"] if "duration" in w]
    print(f"[{i+1}/{len(cases)}] {c['sid']} {c['desk']:<5} rc={p.returncode} wall={wall:5.1f}s "
          f"calls={len(calls)} SumTTFT={s:7.1f}ms warms={len(warms)} feeds={len(st['stats']['feeds'])}", flush=True)
print("DONE", out)
