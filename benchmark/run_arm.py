"""Every experiment of one arm on one server (plan_on_greatlake.md), resumable.

    python -m benchmark.run_arm <arm> [--part all|sweeps|mixed] [--only sweep_coding_agent_c20,mix_L8]
                                      [--results benchmark/results] [--timeout 7200] [--smoke] [--dry-run]

One SLURM job runs this once per arm: the arm's server is started once (start_server.bash,
one model load), then

  1. the four per-agent concurrency sweeps, workloads back to back, no flush (the cache and
     the learned statistics carry over, as in the September/October runs);
  2. one KV flush (POST /v1/cache/flush: device and host tiers, the planner's ledger), then
     the five mixed-workload lane levels, a KV flush between consecutive levels. Learned
     statistics are not reset by a flush.

ours_noprefetch runs the mixed part only (its default --part).

Output: <results>/<arm>/<level>/ per level (see benchmark.workload: logs/, sessions.jsonl,
session_events.jsonl, server.log slice, requests.jsonl slice, summary.json); the server's own
files at <results>/<arm>/server.<n>.log (n = start count), server.pid, requests.jsonl (all
levels); progress.log one line per event; provenance in git_head.txt / working_tree.patch /
plan.json. A level whose summary.json exists is skipped, so re-running the same command
after a job died or hit its wall time continues where it stopped (an unfinished level is
wiped and redone). Before every level the server is health-checked and restarted if it
died (logged as SERVER-RESTART: the cache state resets there).

SIGTERM / SIGINT / SIGUSR1 (SLURM's time-limit warning, forwarded by the batch script)
kill the running sessions and the server and exit 130; the level in progress is redone
on resume.
"""
import argparse
import fnmatch
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from benchmark import workload

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
START_SERVER = os.path.join(REPO, "start_server.bash")
PORT = int(os.environ.get("GUARD_PORT", "8964"))

ARMS = ["vanilla", "adapter", "kvonly", "relayout", "ours", "continuum", "cachescout", "kvflow",
        "continuum_relayout", "cachescout_relayout", "kvflow_relayout", "ours_noprefetch"]
MIX_ONLY = {"ours_noprefetch"}

# per-agent sweeps: (program, sessions per lane k, concurrency levels c)
SWEEPS = [
    ("fact_check", 4, [4, 8, 12, 16, 24]),
    ("coding_agent", 3, [8, 16, 32, 48, 64, 80]),
    ("BFCL_agent", 3, [4, 8, 12, 16, 20, 24]),
    ("doc_analysis", 4, [4, 6, 8, 12, 16, 20]),
]
# mixed workload: lanes per program; coding fixed at 6 sessions per lane, the other three follow
MIX_LANES = [2, 3, 4, 5, 6, 8]
MIX_PROGRAMS = ["fact_check", "BFCL_agent", "coding_agent", "doc_analysis"]
MIX_FIXED = {"coding_agent": 6}


@dataclass
class Level:
    name: str                   # directory name under <results>/<arm>/
    kind: str                   # "sweep" | "mix" | "smoke"
    lanes: Dict[str, int]
    sessions: Dict[str, int]
    warmup: int
    wrap: bool
    flush_before: bool = False


def plan_levels(arm: str, part: str, smoke: bool) -> List[Level]:
    if smoke:
        return [Level("smoke_fact_check", "smoke", {"fact_check": 2}, {"fact_check": 1}, warmup=1, wrap=False,
                      flush_before=False)]
    levels: List[Level] = []
    if part in ("all", "sweeps"):
        for program, k, cs in SWEEPS:
            for i, c in enumerate(cs):
                levels.append(Level(f"sweep_{program}_c{c}", "sweep", {program: c}, {program: k},
                                    warmup=1 if i == 0 else 0, wrap=True))
    if part in ("all", "mixed"):
        for lanes in MIX_LANES:
            levels.append(Level(f"mix_L{lanes}", "mix", {p: lanes for p in MIX_PROGRAMS},
                                {p: MIX_FIXED.get(p, 0) for p in MIX_PROGRAMS}, warmup=1, wrap=False,
                                flush_before=True))
    return levels


class Arm:
    def __init__(self, arm: str, results: str, timeout_s: float, python: str):
        self.arm = arm
        self.out = os.path.join(results, arm)
        os.makedirs(self.out, exist_ok=True)
        self.timeout_s = timeout_s
        self.python = python
        self.progress = open(os.path.join(self.out, "progress.log"), "a", buffering=1)
        self.pidfile = os.path.join(self.out, "server.pid")
        self.request_log = os.path.join(self.out, "requests.jsonl")
        self.server_log: Optional[str] = None
        self.starts = len([f for f in os.listdir(self.out) if f.startswith("server.") and f.endswith(".log")])
        self.stopping = False

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%m-%d %H:%M:%S')} {self.arm} {msg}"
        print(line, flush=True)
        self.progress.write(line + "\n")

    # ------------------------------------------------------------------ server
    def server_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({"PIDFILE": self.pidfile, "REQUEST_LOG": self.request_log, "LOG": self.server_log or ""})
        if self.python:
            env["PYTHON"] = self.python
        return env

    def start_server(self) -> None:
        self.server_log = os.path.join(self.out, f"server.{self.starts}.log")
        self.starts += 1
        self.log(f"server start #{self.starts} log={os.path.relpath(self.server_log, REPO)}")
        t0 = time.time()
        rc = subprocess.run(["bash", START_SERVER, self.arm, "--engine-log", "info"], cwd=REPO,
                            env=self.server_env()).returncode
        if rc != 0:
            raise RuntimeError(f"start_server.bash {self.arm} failed with rc={rc}")
        self.log(f"server ready in {time.time() - t0:.0f}s")

    def stop_server(self) -> None:
        subprocess.run(["bash", START_SERVER, "stop"], cwd=REPO, env=self.server_env())

    @staticmethod
    def alive() -> bool:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def ensure_server(self, before: str) -> None:
        if self.alive():
            return
        if self.server_log is None:
            self.start_server()
            return
        self.log(f"SERVER-RESTART: not answering before {before} (cache state resets)")
        self.stop_server()
        time.sleep(3)
        self.start_server()

    def flush(self, before: str, tries: int = 40, wait_s: float = 5.0) -> bool:
        """POST /v1/cache/flush until the engine reports it flushed (it refuses while busy)."""
        for i in range(tries):
            try:
                req = urllib.request.Request(f"http://localhost:{PORT}/v1/cache/flush", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    reply = json.loads(r.read().decode() or "{}")
                if reply.get("flushed"):
                    self.log(f"flush before {before}: ok (attempt {i + 1})")
                    return True
                self.log(f"flush before {before}: engine busy, retry in {wait_s:.0f}s")
            except Exception as e:
                self.log(f"flush before {before}: {type(e).__name__}: {e}; retry in {wait_s:.0f}s")
            time.sleep(wait_s)
        self.log(f"FLUSH-FAILED before {before}: continuing with a warm cache")
        return False

    # ------------------------------------------------------------------ levels
    def run(self, levels: List[Level]) -> int:
        todo = [lv for lv in levels if not os.path.exists(os.path.join(self.out, lv.name, "summary.json"))]
        for lv in levels:
            if lv not in todo:
                self.log(f"{lv.name} done already, skipped")
        if not todo:
            self.log("nothing to do")
            return 0
        failed = 0
        for n, lv in enumerate(todo):
            if self.stopping:
                break
            self.ensure_server(lv.name)
            if lv.flush_before:
                self.flush(lv.name)
            spec = " ".join(f"{p}={lv.lanes[p]}x{lv.sessions.get(p, 0) or 'follow'}" for p in lv.lanes)
            self.log(f"{lv.name} start ({n + 1}/{len(todo)}) {spec} warmup={lv.warmup} wrap={lv.wrap}")
            t0 = time.time()
            try:
                s = workload.run_level(os.path.join(self.out, lv.name), lv.lanes, lv.sessions, warmup=lv.warmup,
                                       wrap_fixed=lv.wrap, server=f"localhost:{PORT}", server_log=self.server_log or "",
                                       request_log=self.request_log, timeout_s=self.timeout_s,
                                       tag=f"{self.arm}/{lv.name}")
            except Exception as e:
                failed += 1
                self.log(f"{lv.name} FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {e}")
                continue
            a = s["all"]
            progs = s["programs"]
            sessions = sum(r["sessions"] for r in progs.values())
            bad = sum(r["failed"] for r in progs.values())
            host = [r["host_pct"] for r in progs.values() if "host_pct" in r]
            miss = [r["miss_pct"] for r in progs.values() if "miss_pct" in r]
            self.log(f"{lv.name} done in {time.time() - t0:.0f}s: sessions={sessions} failed={bad} "
                     f"full_load={s['full_load_s']}s sessions/min={a['sessions_per_min_full_load']} "
                     f"prompt_ktok/min={a['prompt_ktokens_per_min']} recompute_ktok/min={a['recompute_ktokens_per_min']} "
                     f"host%={host} miss%={miss}")
            if lv.kind == "smoke":   # the smoke test also exercises the flush path of this arm's controller
                if not self.flush(f"{lv.name} end"):
                    failed += 1
        return 1 if failed else 0


def provenance(out: str, levels: List[Level], args: argparse.Namespace) -> None:
    def git(*cmd: str) -> str:
        try:
            return subprocess.check_output(["git", *cmd], cwd=REPO, stderr=subprocess.DEVNULL).decode()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
    with open(os.path.join(out, "git_head.txt"), "a") as f:
        f.write(f"{time.strftime('%F %T')} {git('rev-parse', 'HEAD').strip()} {' '.join(sys.argv[1:])}\n")
    tracked = ["server", "model", "static_analysis", "start_server.bash", "benchmark/workload.py",
               "benchmark/run_arm.py", "benchmark/applications"]
    patch = git("diff", "HEAD", "--", *tracked)
    if patch and not os.path.exists(os.path.join(out, "working_tree.patch")):
        with open(os.path.join(out, "working_tree.patch"), "w") as f:
            f.write(patch)
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump({"arguments": vars(args), "levels": [lv.__dict__ for lv in levels],
                   "server_defaults": {k: os.environ.get(k) for k in ("MODEL", "HOST", "KV", "CLIP", "SCHED")},
                   "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
                   "host": os.uname().nodename}, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("--part", choices=["all", "sweeps", "mixed"], default=None,
                    help="default: all (mixed for ours_noprefetch)")
    ap.add_argument("--only", default="", help="comma-separated level name globs, e.g. 'sweep_coding_*,mix_L8'")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--timeout", type=float, default=7200.0, help="per session (s)")
    ap.add_argument("--python", default=os.environ.get("PYTHON", ""), help="server interpreter (start_server.bash PYTHON)")
    ap.add_argument("--smoke", action="store_true", help="one short fact_check level (c=2, k=1) and a flush, then stop")
    ap.add_argument("--dry-run", action="store_true", help="print the levels and what would be skipped")
    a = ap.parse_args()
    part = a.part or ("mixed" if a.arm in MIX_ONLY else "all")
    levels = plan_levels(a.arm, part, a.smoke)
    if a.only:
        pats = [p.strip() for p in a.only.split(",") if p.strip()]
        levels = [lv for lv in levels if any(fnmatch.fnmatch(lv.name, p) for p in pats)]
    if a.dry_run:
        for lv in levels:
            done = os.path.exists(os.path.join(a.results, a.arm, lv.name, "summary.json"))
            print(f"{'skip' if done else 'run '} {a.arm}/{lv.name}: lanes={lv.lanes} sessions={lv.sessions} "
                  f"warmup={lv.warmup} wrap={lv.wrap} flush_before={lv.flush_before}")
        return 0

    # The tool caches (git-ignored, carried by greatlakes/sync_to_greatlakes.sh) stand in for live web
    # search: without them fact_check and BFCL hit Wikipedia from the compute node and get rate-limited.
    for var in ("FC_CACHE_DIR", "BF_CACHE_DIR"):
        d = os.environ.get(var) or workload.ENV_DEFAULTS[var]
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        if n == 0:
            print(f"{var}={d} is empty or missing: sync the search caches first (see greatlakes/README.md)",
                  file=sys.stderr)
            return 2
        print(f"[cache] {var}={os.path.relpath(d, REPO)}: {n} files", flush=True)

    arm = Arm(a.arm, a.results, a.timeout, a.python)
    provenance(arm.out, levels, a)

    def on_signal(signum, _frame):
        arm.stopping = True
        arm.log(f"signal {signal.Signals(signum).name}: killing {workload.kill_active_sessions()} sessions and the server")
        arm.stop_server()
        arm.log("ARM-INTERRUPTED (re-run the same command to resume)")
        os._exit(130)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, on_signal)

    arm.log(f"ARM-START part={part} levels={len(levels)} results={os.path.relpath(arm.out, REPO)}")
    try:
        rc = arm.run(levels)
    finally:
        arm.stop_server()
    arm.log("ARM-DONE" if rc == 0 else "ARM-DONE-WITH-FAILURES (re-run the same command to redo them)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
