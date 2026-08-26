"""probe vs no-probe on benchmark/mechanisms/big_fan_out.jac, one client, N rounds x 8 requests."""
import json, os, re, signal, subprocess, sys, time, glob
REPO = "/home/xiaoyu/idontknowitsname"
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
OUT = sys.argv[1]; ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CONDS = {
    "probe":   ["--spec-features", "const,ret,toolturn,probe"],
    "noprobe": ["--spec-features", "const,ret,toolturn"],
    "off":     ["--no-prefill"],
}
def start(extra, log):
    cmd = [sys.executable, f"{REPO}/start_server.py", "--model", MODEL,
           "--program", "big_fan_out:"+os.environ.get("JAC_FILE","benchmark/mechanisms/big_fan_out.jac")+":1145"] + extra
    return subprocess.Popen(cmd, cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
def stop(p):
    try: os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=30)
    except Exception:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    time.sleep(5)
def wait_ready(log, p, t=900):
    d = time.time() + t
    while time.time() < d:
        if p.poll() is not None: raise RuntimeError("server died: " + log)
        if os.path.exists(log) and "listening on" in open(log).read(): return
        time.sleep(1)
    raise RuntimeError("timeout")
for cond in (sys.argv[3].split(",") if len(sys.argv) > 3 else CONDS):
    d = f"{OUT}/{cond}"; os.makedirs(d, exist_ok=True)
    for db in glob.glob(f"{REPO}/.jac/data/*.db"): os.remove(db)
    p = start(CONDS[cond], f"{d}/server.log")
    try:
        wait_ready(f"{d}/server.log", p)
        env = {**os.environ, "MODEL": MODEL, "JAC_ROUTE_CACHE_LAYOUT": "1",
               "LD_LIBRARY_PATH": "/home/xiaoyu/miniconda3/envs/jaseci/lib"}
        with open(f"{d}/client.log", "w") as cl:
            for r in range(ROUNDS):
                t0 = time.time()
                rc = subprocess.run(["jac", "run", os.environ.get("JAC_FILE","benchmark/mechanisms/big_fan_out.jac")], cwd=REPO,
                                    env=env, stdout=cl, stderr=subprocess.STDOUT, timeout=900).returncode
                print(f"[{cond}] round {r} rc={rc} wall={time.time()-t0:.1f}s", flush=True)
    finally:
        stop(p)
