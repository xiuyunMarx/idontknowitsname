"""Mixed workload: the four applications as interleaved sessions on one server.

    python -m benchmark.mixed_workload --tag ours --server-log ours.log \
        --lanes fact_check=4,coding_agent=8,doc_analysis=4,BFCL_agent=4 --sessions 5

Every session is one `jac run` of an application with its input in the program's
environment variable. Lanes of one program run their sessions back to back; a
program with 0 sessions per lane "follows": it keeps taking inputs until every
fixed lane has finished. Sessions completed within the same fixed windows are
reported so arms run separately can be compared.

Results go to benchmark/mixed_results/<tag>/: one log per session, sessions.jsonl,
summary.json, and a row per program appended to benchmark/mixed_results/summary.csv.
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
APPS = os.path.join(HERE, "applications")
DATA = os.path.join(APPS, "bench_data")
RESULTS = os.path.join(HERE, "mixed_results")
JAC = os.environ.get("JAC") or (os.path.join(os.path.dirname(sys.executable), "jac")
                               if os.path.exists(os.path.join(os.path.dirname(sys.executable), "jac")) else "jac")

ENV_DEFAULTS = {
    "FC_MIN_ROUNDS": "2",
    "FC_TOOL_DELAY_S": "2",
    "CA_TOOL_DELAY_S": "2",
    "BF_TOOL_DELAY_S": "2",
    "FC_CACHE_DIR": os.path.join(DATA, "HoVer", "wiki_cache"),
    "BF_CACHE_DIR": os.path.join(DATA, "BFCL", "web_cache"),
}

SERVE_RE = re.compile(r"^\[serve\] (\d+)-\d+t(\d+)-\w+ (.*)$")
CALL_RE = re.compile(r"^\[call\] (\d+) #(\d+) (\S+?)(?: predicted=(hit|miss))?(?: ctx=\([^)]*\))?$")
VERDICT_RE = re.compile(r"^Verdict: (\S+)")
ROUNDS_RE = re.compile(r"^Rounds: (\d+), (?:findings|notes|attempts): (\d+)")


def read_tsv(path: str, label_col: int, input_col: int) -> List[Dict[str, str]]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            out.append({"input": cols[input_col], "expected": cols[label_col]})
    return out


def read_jsonl_ids(path: str, key: str) -> List[Dict[str, str]]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append({"input": str(json.loads(line)[key]), "expected": "PASSED"})
    return out


PROGRAMS = {
    "fact_check": {
        "env": "FC_CLAIM",
        "no_header_last": True,
        "tasks": lambda: read_tsv(os.path.join(DATA, "HoVer", "hover_claims_120.tsv"), 0, 2),
    },
    "coding_agent": {
        "env": "CA_TASKS",
        "no_header_last": False,
        "tasks": lambda: read_tsv(os.path.join(DATA, "HumanEval", "tasks.txt"), 0, 1),
    },
    "doc_analysis": {
        "env": "DA_TASK",
        "no_header_last": False,
        "tasks": lambda: read_jsonl_ids(os.path.join(DATA, "finicial_bench", "financebench_open_source.jsonl"), "financebench_id"),
    },
    "BFCL_agent": {
        "env": "BF_TASK",
        "no_header_last": False,
        "tasks": lambda: read_jsonl_ids(os.path.join(DATA, "BFCL", "BFCL_v4_web_search.json"), "id"),
    },
}


class Runner:
    def __init__(self, tag: str, timeout_s: float):
        self.out_dir = os.path.join(RESULTS, tag)
        self.log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.timeout_s = timeout_s
        self.records: List[dict] = []
        self.lock = threading.Lock()
        self.next_idx: Dict[str, int] = {}
        self.stop = threading.Event()
        self.env = dict(os.environ)
        for k, v in ENV_DEFAULTS.items():
            self.env.setdefault(k, v)

    def take(self, program: str, tasks: List[dict], wrap: bool) -> Optional[dict]:
        with self.lock:
            i = self.next_idx.get(program, 0)
            if i >= len(tasks) and not wrap:
                return None
            self.next_idx[program] = i + 1
            task = dict(tasks[i % len(tasks)])
            task["idx"] = i
            return task

    def run_session(self, program: str, task: dict, phase: str, lane: str = "warmup", fixed: bool = False) -> dict:
        env = dict(self.env)
        env[PROGRAMS[program]["env"]] = task["input"]
        log = os.path.join(self.log_dir, f"{program}-{phase}-{task['idx']:03d}.log")
        t0 = time.time()
        with open(log, "w") as out:
            proc = subprocess.Popen([JAC, "run", os.path.join(APPS, f"{program}.jac")],
                                    cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -9
        t1 = time.time()
        verdict, rounds, calls = "", 0, 0
        with open(log) as f:
            for line in f:
                m = VERDICT_RE.match(line)
                if m:
                    verdict = m.group(1)
                m = ROUNDS_RE.match(line)
                if m:
                    rounds, calls = int(m.group(1)), int(m.group(2))
        rec = {"program": program, "phase": phase, "idx": task["idx"], "pid": proc.pid, "input": task["input"],
               "expected": task["expected"], "verdict": verdict, "rounds": rounds, "notes": calls,
               "rc": rc, "t0": t0, "t1": t1, "wall": round(t1 - t0, 3), "lane": lane, "fixed": fixed}
        with self.lock:
            self.records.append(rec)
        print(f"[{program}] {phase} {task['idx']} rc={rc} wall={rec['wall']}s rounds={rounds} "
              f"verdict={verdict}/{task['expected']}", flush=True)
        return rec

    def lane(self, program: str, tasks: List[dict], sessions: int) -> None:
        done = 0
        while sessions == 0 or done < sessions:
            if sessions == 0 and self.stop.is_set():
                break
            task = self.take(program, tasks, wrap=(sessions == 0))
            if task is None:
                break
            self.run_session(program, task, "measured", threading.current_thread().name, sessions > 0)
            done += 1


def register_programs(server: str, programs: List[str]) -> None:
    sys.path.insert(0, REPO)
    from static_analysis.agent_launcher import analyze_program, register
    for p in programs:
        payload = analyze_program(os.path.join(APPS, f"{p}.jac"))
        reply = register(server, payload, no_header_last=PROGRAMS[p]["no_header_last"])
        print(f"[register] {p}: {reply}", flush=True)


def parse_server_log(path: str, pids: Dict[int, str]) -> Dict[str, dict]:
    """Per program: [serve] rows and [call] prediction hits of the measured sessions."""
    per: Dict[str, dict] = {p: {"serves": [], "hit": 0, "pred": 0} for p in set(pids.values())}
    with open(path) as f:
        for line in f:
            m = SERVE_RE.match(line)
            if m and int(m.group(1)) in pids:
                fields = dict(kv.split("=", 1) for kv in m.group(3).split())
                per[pids[int(m.group(1))]]["serves"].append({k: float(v) for k, v in fields.items()})
                continue
            m = CALL_RE.match(line)
            if m and int(m.group(1)) in pids and m.group(4):
                d = per[pids[int(m.group(1))]]
                d["pred"] += 1
                d["hit"] += m.group(4) == "hit"
    return per


def q(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def summarize(records: List[dict], server: Dict[str, dict], windows: List[int]) -> dict:
    measured = [r for r in records if r["phase"] == "measured"]
    start = min(r["t0"] for r in measured)
    fixed = [r for r in measured if r["fixed"]]
    full_load = min(max(r["t1"] for r in fixed if r["lane"] == lane) for lane in {r["lane"] for r in fixed}) - start
    out = {"start": start, "full_load_s": round(full_load, 1), "programs": {}}
    for p in sorted({r["program"] for r in measured}):
        rs = [r for r in measured if r["program"] == p]
        ok = [r for r in rs if r["rc"] == 0]
        walls = [r["wall"] for r in ok]
        row = {
            "sessions": len(rs), "failed": len(rs) - len(ok),
            "jct_p50_s": round(q(walls, 0.5), 1), "jct_p95_s": round(q(walls, 0.95), 1),
            "jct_mean_s": round(statistics.mean(walls), 1) if walls else 0.0,
            "rounds_mean": round(statistics.mean(r["rounds"] for r in ok), 2) if ok else 0.0,
            "accuracy": f"{sum(r['verdict'] == r['expected'] for r in ok)}/{len(ok)}",
            "done_full_load": sum(r["t1"] - start <= full_load for r in rs),
        }
        for w in windows:
            row[f"done_{w}s"] = sum(r["t1"] - start <= w for r in rs)
        s = server.get(p, {}).get("serves", [])
        if s:
            prompt = sum(x["prompt_tokens"] for x in s)
            row.update({
                "calls": len(s),
                "prompt_tokens_per_call": round(prompt / len(s)),
                "ttft_p50_ms": round(q([x["ttft_ms"] for x in s], 0.5)),
                "ttft_p95_ms": round(q([x["ttft_ms"] for x in s], 0.95)),
                "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in s)),
                "device_pct": round(100 * sum(x["cached_device"] for x in s) / prompt, 1),
                "host_pct": round(100 * sum(x["cached_host"] for x in s) / prompt, 1),
                "miss_pct": round(100 * (1 - sum(x["cached_tokens"] for x in s) / prompt), 1),
            })
            pred = server[p]["pred"]
            row["predicted_hit"] = round(server[p]["hit"] / pred, 2) if pred else None
        out["programs"][p] = row
    tot = {f"done_{w}s": sum(v[f"done_{w}s"] for v in out["programs"].values()) for w in windows}
    tot["done_full_load"] = sum(v["done_full_load"] for v in out["programs"].values())
    tot["sessions_per_min_full_load"] = round(tot["done_full_load"] / (full_load / 60), 2) if full_load else 0.0 #type: ignore
    out["all"] = tot
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lanes", default="fact_check=4,coding_agent=8,doc_analysis=4,BFCL_agent=4",
                    help="program=lanes,...; the four applications by default")
    ap.add_argument("--sessions", default="5", help="per lane: N, or program=N,... (0 = follow)")
    ap.add_argument("--warmup", type=int, default=1, help="sequential sessions per program before measuring")
    ap.add_argument("--server", default="localhost:8964")
    ap.add_argument("--server-log", default="", help="the server's stdout, for cache and prediction stats")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--windows", default="600,900,1200", help="fixed windows (s) for completion counts")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per session (s)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lanes = {k: int(v) for k, v in (kv.split("=") for kv in a.lanes.split(","))}
    if "=" in a.sessions:
        sessions = {k: int(v) for k, v in (kv.split("=") for kv in a.sessions.split(","))}
    else:
        sessions = {p: int(a.sessions) for p in lanes}
    windows = [int(w) for w in a.windows.split(",")]
    tasks = {p: PROGRAMS[p]["tasks"]() for p in lanes}
    for p in lanes:
        print(f"[plan] {p}: {lanes[p]} lanes x {sessions.get(p, 0) or 'follow'} sessions, {len(tasks[p])} inputs", flush=True)
    if a.dry_run:
        return 0

    if not a.no_register:
        register_programs(a.server, list(lanes))
    runner = Runner(a.tag, a.timeout)

    for p in lanes:
        for _ in range(a.warmup):
            task = runner.take(p, tasks[p], wrap=True)
            runner.run_session(p, task, "warmup")  #type: ignore

    threads = []
    for p, n in lanes.items():
        for i in range(n):
            t = threading.Thread(target=runner.lane, args=(p, tasks[p], sessions.get(p, 0)), name=f"{p}-{i}")
            threads.append(t)
    t_start = time.time()
    for t in threads:
        t.start()
    fixed = [t for t in threads if sessions.get(t.name.rsplit("-", 1)[0], 0) > 0]
    for t in fixed:
        t.join()
    runner.stop.set()
    for t in threads:
        t.join()

    with open(os.path.join(runner.out_dir, "sessions.jsonl"), "w") as f:
        for r in runner.records:
            f.write(json.dumps(r) + "\n")

    pids = {r["pid"]: r["program"] for r in runner.records if r["phase"] == "measured"}
    server = parse_server_log(a.server_log, pids) if a.server_log else {}
    summary = summarize(runner.records, server, windows)
    summary["tag"], summary["lanes"], summary["sessions"], summary["elapsed_s"] = a.tag, lanes, sessions, round(time.time() - t_start, 1)
    with open(os.path.join(runner.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{a.tag}] full-load window {summary['full_load_s']} s, "
          f"{summary['all']['sessions_per_min_full_load']} sessions/min; "
          + ", ".join(f"{k}={v}" for k, v in summary["all"].items() if k.startswith("done_")), flush=True)
    for p, row in summary["programs"].items():
        print(f"[{a.tag}/{p}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    csv_path = os.path.join(RESULTS, "summary.csv")
    rows = [dict(tag=a.tag, program=p, **row) for p, row in summary["programs"].items()]
    fields = ["tag", "program"] + sorted({k for r in rows for k in r} - {"tag", "program"})
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
