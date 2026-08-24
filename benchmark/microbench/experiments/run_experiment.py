"""Multi-tenant validation sweep for global speculation + token accounting.

Four conditions, run back to back with identical (seeded) load:
    off           --no-prefill
    newest-idle   --spec-policy newest --no-budget   (pre-multi-tenant behavior)
    global-idle   --no-budget                        (Fix 1 alone)
    global-budget (loads profile-<model>.json)       (Fix 1 + Fix 2)

Each condition starts the guard server, then runs every tenant program in its
own loop of --rounds sequential `jac run` processes with rotating inputs and a
seeded stagger, identical across conditions. Server stdout goes to
<out>/<condition>/server.log, per-run client results to clients.jsonl.

Integrity: a client that cannot reach the server silently falls back to a local
model and still "succeeds", so afterwards the server log must show exactly
tenants x rounds `Registered` lines — anything less voids the trial.

Usage: python experiments/run_experiment.py [--rounds 10] [--conditions a,b] [--out DIR]
"""

import argparse
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 20260823

CONDITIONS = {
    "off": ["--no-prefill"],
    "newest-idle": ["--spec-policy", "newest", "--no-budget"],
    "global-idle": ["--no-budget"],
    "global-budget": [],
}

# (program_name, jac file, port, input env var, cases)
TENANTS = [
    ("research", "research_agent.jac", 8964, "QUESTION", [
        "How long are audit records retained?",
        "How are customer exports encrypted?",
        "What does production access require?",
        "Is MFA required for production access?",
        "What encryption protects exports at rest?",
    ]),
    ("operations", "operations_agent.jac", 8965, "REQUEST", [
        "Calculate the error rate and report p95 latency.",
        "Check whether error rate breaches the alert threshold.",
        "Report p95 latency and whether it is acceptable.",
        "Summarize service health from error rate and latency.",
        "Is the current error rate above the paging threshold?",
    ]),
    ("route", "route_wire.jac", 8966, "TICKET", [
        "I was charged twice for last month's invoice.",
        "Our export job fails with a 500 after the last release.",
        "Someone logged into my account from another country.",
        "Please refund the duplicate charge on my card.",
        "The dashboard shows an internal error since this morning.",
    ]),
    ("support_email", "support_email_agent.jac", 8967, "EMAIL", [
        "My welcome kit still hasn't arrived and I ordered a month ago. Order 8841. If it's lost I want my money back.",
        "I want to cancel my annual plan and get whatever refund I'm owed. Order 8841.",
        "Where is my order 8841? The tracking page hasn't moved in a week.",
        "You billed me for two seats but we only use one. Order 8841, please fix.",
        "The welcome kit arrived damaged. Order 8841. Replace it or refund it.",
    ]),
    ("pipeline", "data_pipeline_agent.jac", 8968, "RECORD", [
        "Customer Meyer GmbH reports that 4 units of the RackServe X2 arrived dented on 2026-08-15, production line down, wants replacement this week.",
        "Acme Corp asks whether the StoragePod S8 supports encryption at rest, no urgency, sales inquiry dated 2026-08-18.",
        "Customer Larsen A/S orders 12 units of NetSwitch N4 for delivery in September, PO attached 2026-08-19.",
        "Kwan Ltd reports intermittent reboots of 2 EdgeBox E1 units since 2026-08-10, severity high, production impact.",
        "Customer Ortiz SL complains invoice 5512 double-charged shipping on 2026-08-17, requests correction.",
    ]),
    ("moderation", "moderation_agent.jac", 8969, "POST", [
        "Buy cheap followers now at follow-blast dot com, DM @growthguru, limited offer!!!",
        "I know where you live and you will regret posting that.",
        "You people are all idiots and @dev_kate is the dumbest of them all, uninstall your brain.",
        "Honestly this update made the app slower on my phone, anyone else?",
        "Win a free phone!! just click bit.ly/freefone and enter your card details!!",
    ]),
    ("dispatch", "dispatch_router.jac", 8970, "ISSUE", [
        "Checkout returns 502 for EU users since 14:00, replication lag on the orders database is climbing.",
        "Users report login loops after the SSO certificate rotation this morning.",
        "Card charges are being declined for 3D-secure payments since the processor maintenance window.",
        "The landing page renders a blank screen on Safari after yesterday's release.",
        "Packet loss between us-east and eu-west spiked to 8% and API timeouts follow.",
    ]),
]


def start_server(extra: list, log_path: str, model: str = MODEL) -> subprocess.Popen:
    cmd = [sys.executable, "start_server.py", "--model", model, "--workers", "8"]
    for name, f, port, _, _ in TENANTS:
        cmd += ["--program", f"{name}:jac_programs/{f}:{port}"]
    cmd += extra
    log = open(log_path, "w")
    # New session so shutdown can signal the whole group — a bare terminate()
    # orphans the VLLM::EngineCore child, which keeps the GPU memory.
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)


def stop_server(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=30)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(5)  # let the driver release GPU memory before the next condition


def wait_ready(log_path: str, proc: subprocess.Popen, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log_path}")
        try:
            with open(log_path) as f:
                if f.read().count("listening on") >= len(TENANTS):
                    return
        except OSError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server not ready within {timeout}s, see {log_path}")


def tenant_loop(name: str, jac_file: str, env_var: str, cases: list, rounds: int,
                stagger: float, out_dir: str, results: list, lock: threading.Lock) -> None:
    time.sleep(stagger)
    env = {**os.environ, "MODEL": MODEL, "JAC_ROUTE_CACHE_LAYOUT": "1",
           "LD_LIBRARY_PATH": "/home/xiaoyu/miniconda3/envs/jaseci/lib"}
    with open(os.path.join(out_dir, f"client-{name}.log"), "w") as log:
        for r in range(rounds):
            env[env_var] = cases[r % len(cases)]
            started = time.time()
            proc = subprocess.run(["jac", "run", f"jac_programs/{jac_file}"], cwd=ROOT,
                                  env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            with lock:
                results.append({"tenant": name, "round": r, "wall": time.time() - started,
                                "rc": proc.returncode})


def run_condition(cond: str, rounds: int, out_root: str) -> None:
    out_dir = os.path.join(out_root, cond)
    os.makedirs(out_dir, exist_ok=True)
    for db in glob.glob(os.path.join(ROOT, ".jac", "data", "*.db")):
        os.remove(db)  # route graphs accumulate across runs and skew repeated trials

    log_path = os.path.join(out_dir, "server.log")
    server = start_server(CONDITIONS[cond], log_path)
    try:
        wait_ready(log_path, server)
        print(f"[{cond}] server ready, launching {len(TENANTS)} tenants x {rounds} rounds")
        rng = random.Random(SEED)  # same stagger schedule for every condition
        results: list = []
        lock = threading.Lock()
        threads = [
            threading.Thread(target=tenant_loop, args=(
                name, f, var, cases, rounds, rng.uniform(0.0, 3.0), out_dir, results, lock))
            for name, f, _, var, cases in TENANTS
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(os.path.join(out_dir, "clients.jsonl"), "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
    finally:
        stop_server(server)

    with open(log_path) as f:
        registered = len(re.findall(r"Registered", f.read()))
    expected = len(TENANTS) * rounds
    ok = all(r["rc"] == 0 for r in results)
    print(f"[{cond}] done: registered={registered}/{expected} client_rc_ok={ok}")
    if registered != expected:
        print(f"[{cond}] WARNING: registration count mismatch — silent-fallback runs, trial is void")


def ensure_profile(out_root: str) -> None:
    path = os.path.join(ROOT, f"profile-{MODEL.rsplit('/', 1)[-1]}.json")
    if os.path.exists(path):
        return
    print(f"[profile] {path} missing, running the interference sweep once")
    log_path = os.path.join(out_root, "profile.log")
    server = start_server(["--profile"], log_path)
    try:
        wait_ready(log_path, server, timeout=3600)  # listening => sweep finished and saved
    finally:
        stop_server(server)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", time.strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    os.makedirs(args.out, exist_ok=True)
    if "global-budget" in conds:
        ensure_profile(args.out)
    for cond in conds:
        run_condition(cond, args.rounds, args.out)
    print(f"results in {args.out}; aggregate with: python experiments/aggregate.py {args.out}")


if __name__ == "__main__":
    main()
