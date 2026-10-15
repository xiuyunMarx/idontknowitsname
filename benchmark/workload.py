"""One benchmark level: the applications as interleaved sessions on a running server.

    python -m benchmark.workload --out-dir benchmark/results/ours/sweep_fact_check_c4 \
        --lanes fact_check=4 --sessions 4 --wrap --server-log benchmark/results/ours/server.0.log

Every session is one `jac run` of an application with its input in the program's
environment variable. Lanes of one program run their sessions back to back. A lane
with a session count is "fixed": with --wrap it keeps taking inputs modulo the dataset
(a repeat comes >= 100 sessions later, long after the pools turned over); without it
the lane stops when the dataset is exhausted. A program with 0 sessions per lane
"follows": it keeps taking inputs until every fixed lane has finished.

The level's raw data goes to --out-dir: logs/<program>-<phase>-<idx>.log per session,
sessions.jsonl, session_events.jsonl, run_config.json, summary.json, and the slice of
the server's log (and of its request trace, if given) that belongs to this level:
server.log and requests.jsonl. Throughput is reported as prompt ktokens/min (every
request's full prompt), recompute ktokens/min (the uncached part, the prefill the GPU
did) and output ktokens/min, over the program's elapsed time and over the full-load
window (first measured start to the first fixed lane's end).

benchmark.run_arm drives the levels of one arm; this module is also usable alone.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
APPS = os.path.join(HERE, "applications")
DATA = os.path.join(APPS, "bench_data")
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

# request ids are <client pid>-<call>t<turn>-<hex>; the pid is the session (OpenAI `user` field)
SERVE_RE = re.compile(r"^\[serve\] (\d+)-\d+t(\d+)-\w+ (.*)$")
DECODE_RE = re.compile(r"^\[decode\] (\d+)-\d+t\d+-\w+ decode_ms=([\d.]+) output_tokens=(\d+)")
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
        "env": "FC_CLAIM", "no_header_last": True,
        "tasks": lambda: read_tsv(os.path.join(DATA, "HoVer", "hover_claims_120.tsv"), 0, 2),
    },
    "coding_agent": {
        "env": "CA_TASKS", "no_header_last": False,
        "tasks": lambda: read_tsv(os.path.join(DATA, "HumanEval", "tasks.txt"), 0, 1),
    },
    "doc_analysis": {
        "env": "DA_TASK", "no_header_last": False,
        "tasks": lambda: read_jsonl_ids(os.path.join(DATA, "finicial_bench", "financebench_open_source.jsonl"),
                                        "financebench_id"),
    },
    "BFCL_agent": {
        "env": "BF_TASK", "no_header_last": False,
        "tasks": lambda: read_jsonl_ids(os.path.join(DATA, "BFCL", "BFCL_v4_web_search.json"), "id"),
    },
}


ACTIVE_PROCS: "set[subprocess.Popen]" = set()   # running `jac run` sessions, for a driver's signal handler
_ACTIVE_LOCK = threading.Lock()


def kill_active_sessions() -> int:
    with _ACTIVE_LOCK:
        procs = list(ACTIVE_PROCS)
    for p in procs:
        try:
            p.kill()
        except OSError:
            pass
    return len(procs)


class Runner:
    def __init__(self, out_dir: str, timeout_s: float):
        self.out_dir = out_dir
        self.log_dir = os.path.join(out_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.timeout_s = timeout_s
        self.records: List[dict] = []
        self.lock = threading.Lock()
        self.next_idx: Dict[str, int] = {}
        self.stop = threading.Event()
        self.env = dict(os.environ)
        for k, v in ENV_DEFAULTS.items():
            self.env.setdefault(k, v)
        self.records_file = open(os.path.join(out_dir, "sessions.jsonl"), "w", buffering=1)
        self.events_file = open(os.path.join(out_dir, "session_events.jsonl"), "w", buffering=1)

    def close(self) -> None:
        self.records_file.close()
        self.events_file.close()

    def event(self, event: str, **fields) -> None:
        with self.lock:
            self.events_file.write(json.dumps({"event": event, "time_ns": time.time_ns(),
                                               "monotonic_ns": time.monotonic_ns(), **fields}) + "\n")

    def take(self, program: str, tasks: List[dict], wrap: bool) -> Optional[dict]:
        with self.lock:
            i = self.next_idx.get(program, 0)
            if i >= len(tasks) and not wrap:
                return None
            self.next_idx[program] = i + 1
            task = dict(tasks[i % len(tasks)])
            task["idx"] = i                      # running number within this level
            task["input_idx"] = i % len(tasks)   # position in the dataset (repeats when wrapping)
            return task

    def run_session(self, program: str, task: dict, phase: str, lane: str = "warmup", fixed: bool = False) -> dict:
        env = dict(self.env)
        env[PROGRAMS[program]["env"]] = task["input"]
        log = os.path.join(self.log_dir, f"{program}-{phase}-{task['idx']:03d}.log")
        t0 = time.time()
        with open(log, "w") as out:
            proc = subprocess.Popen([JAC, "run", os.path.join(APPS, f"{program}.jac")],
                                    cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
            with _ACTIVE_LOCK:
                ACTIVE_PROCS.add(proc)
            self.event("session_start", program=program, phase=phase, idx=task["idx"], input_idx=task["input_idx"],
                       pid=proc.pid, lane=lane, fixed=fixed, t0=t0, input=task["input"], expected=task["expected"])
            try:
                rc = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                rc = -9
            finally:
                with _ACTIVE_LOCK:
                    ACTIVE_PROCS.discard(proc)
        t1 = time.time()
        verdict, rounds, calls = "", 0, 0
        with open(log, errors="replace") as f:
            for line in f:
                m = VERDICT_RE.match(line)
                if m:
                    verdict = m.group(1)
                m = ROUNDS_RE.match(line)
                if m:
                    rounds, calls = int(m.group(1)), int(m.group(2))
        rec = {"program": program, "phase": phase, "idx": task["idx"], "input_idx": task["input_idx"],
               "pid": proc.pid, "input": task["input"], "expected": task["expected"], "verdict": verdict,
               "rounds": rounds, "notes": calls, "rc": rc, "t0": t0, "t1": t1, "wall": round(t1 - t0, 3),
               "lane": lane, "fixed": fixed}
        with self.lock:
            self.records.append(rec)
            self.records_file.write(json.dumps(rec) + "\n")
        self.event("session_end", **rec)
        print(f"[{program}] {phase} {task['idx']} rc={rc} wall={rec['wall']}s rounds={rounds} "
              f"verdict={verdict}/{task['expected']}", flush=True)
        return rec

    def lane(self, program: str, tasks: List[dict], sessions: int, wrap_fixed: bool) -> None:
        done = 0
        while sessions == 0 or done < sessions:
            if sessions == 0 and self.stop.is_set():
                break
            task = self.take(program, tasks, wrap=(sessions == 0 or wrap_fixed))
            if task is None:
                break
            self.run_session(program, task, "measured", threading.current_thread().name, sessions > 0)
            done += 1


def analyze(agent: str) -> dict:
    """The agent's registration payload: from static_analysis/.cache when it is warm (keyed by the
    agent's absolute path and content), else by running the launcher under the jac binary, whose
    runtime has jaclang (the serving env does not)."""
    from static_analysis.agent_launcher import analyze_program
    try:
        return analyze_program(agent)
    except ImportError:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out = tmp.name
        try:
            subprocess.run([JAC, "-m", "static_analysis.agent_launcher", "--no-register", "--dry-run",
                            "--out", out, agent], cwd=REPO, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            with open(out) as f:
                return json.load(f)
        finally:
            os.unlink(out)


def register_programs(server: str, programs: List[str], wait_s: float = 1800.0) -> None:
    """Register the programs, waiting for the controller: right after "[warmup] engine ready" the
    guard controller may still be running the device profiler (first start of a model on a GPU,
    minutes for a 32B model), during which every request waits in its pool."""
    sys.path.insert(0, REPO)
    from static_analysis.agent_launcher import register
    for p in programs:
        payload = analyze(os.path.join(APPS, f"{p}.jac"))
        t0 = time.time()
        while True:
            try:
                reply = register(server, payload, no_header_last=PROGRAMS[p]["no_header_last"])
                break
            except Exception as e:
                if time.time() - t0 > wait_s:
                    raise
                print(f"[register] {p}: {type(e).__name__}: {e}; controller busy? retry in 15 s", flush=True)
                time.sleep(15)
        print(f"[register] {p}: {reply}", flush=True)


# ---------------------------------------------------------------------------- server-side data

def copy_slice(src: str, dst: str, start: int, end: Optional[int] = None) -> int:
    """Copy src[start:end] (bytes) to dst; returns the number of bytes copied."""
    n = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        f.seek(start)
        remaining = None if end is None else max(0, end - start)
        while remaining is None or remaining > 0:
            chunk = f.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            out.write(chunk)
            n += len(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return n


def copy_trace_window(src: str, dst: str, t_lo_ns: int, t_hi_ns: int) -> int:
    """The request-trace events (BENCH_REQUEST_LOG) with time_ns in [t_lo_ns, t_hi_ns]."""
    n = 0
    with open(src, errors="replace") as f, open(dst, "w") as out:
        for line in f:
            try:
                t = json.loads(line)["time_ns"]
            except (ValueError, KeyError):
                continue
            if t_lo_ns <= t <= t_hi_ns:
                out.write(line)
                n += 1
    return n


def parse_server_log(path: str, pids: Dict[int, str]) -> Dict[str, dict]:
    """Per program: [serve] rows, [decode] rows and [call] prediction hits of the measured sessions."""
    per: Dict[str, dict] = {p: {"serves": [], "decodes": [], "hit": 0, "pred": 0} for p in set(pids.values())}
    with open(path, errors="replace") as f:
        for line in f:
            m = SERVE_RE.match(line)
            if m and int(m.group(1)) in pids:
                fields = dict(kv.split("=", 1) for kv in m.group(3).split() if "=" in kv)
                per[pids[int(m.group(1))]]["serves"].append({k: float(v) for k, v in fields.items()})
                continue
            m = DECODE_RE.match(line)
            if m and int(m.group(1)) in pids:
                per[pids[int(m.group(1))]]["decodes"].append({"decode_ms": float(m.group(2)),
                                                              "output_tokens": int(m.group(3))})
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


def _rate(x: float, seconds: float) -> float:
    return round(x / (seconds / 60), 2) if seconds > 0 else 0.0


def summarize(records: List[dict], server: Dict[str, dict], windows: List[int]) -> dict:
    measured = [r for r in records if r["phase"] == "measured"]
    start = min(r["t0"] for r in measured)
    end = max(r["t1"] for r in measured)
    fixed = [r for r in measured if r["fixed"]]
    lanes = {r["lane"] for r in fixed}
    full_load = (min(max(r["t1"] for r in fixed if r["lane"] == lane) for lane in lanes) - start) if lanes else end - start
    out = {"start": start, "end": end, "elapsed_s": round(end - start, 1), "full_load_s": round(full_load, 1),
           "programs": {}}
    tot_prompt = tot_recompute = tot_output = tot_calls = 0.0
    for p in sorted({r["program"] for r in measured}):
        rs = [r for r in measured if r["program"] == p]
        ok = [r for r in rs if r["rc"] == 0]
        walls = [r["wall"] for r in ok]
        elapsed = max(r["t1"] for r in rs) - start   # first measured start to this program's last finish
        row = {
            "sessions": len(rs), "failed": len(rs) - len(ok),
            "timed_out": sum(r["rc"] == -9 for r in rs),
            "elapsed_s": round(elapsed, 1),
            "sessions_per_min": _rate(len(ok), elapsed),
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
            cached = sum(x["cached_tokens"] for x in s)
            dec = server[p].get("decodes", [])
            output = sum(d["output_tokens"] for d in dec)
            row.update({
                "calls": len(s),
                "calls_per_min": _rate(len(s), elapsed),
                "prompt_tokens_per_call": round(prompt / len(s)),
                "prompt_ktokens_per_min": _rate(prompt / 1000, elapsed),
                "recompute_ktokens_per_min": _rate((prompt - cached) / 1000, elapsed),
                "output_ktokens_per_min": _rate(output / 1000, elapsed),
                "output_tokens_per_call": round(output / len(dec)) if dec else 0,
                "decode_ms_mean": round(statistics.mean(d["decode_ms"] for d in dec)) if dec else 0,
                "ttft_p50_ms": round(q([x["ttft_ms"] for x in s], 0.5)),
                "ttft_p95_ms": round(q([x["ttft_ms"] for x in s], 0.95)),
                "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in s)),
                "device_pct": round(100 * sum(x["cached_device"] for x in s) / prompt, 1) if prompt else 0.0,
                "host_pct": round(100 * sum(x["cached_host"] for x in s) / prompt, 1) if prompt else 0.0,
                "miss_pct": round(100 * (1 - cached / prompt), 1) if prompt else 0.0,
            })
            pred = server[p]["pred"]
            row["predicted_hit"] = round(server[p]["hit"] / pred, 2) if pred else None
            tot_prompt += prompt
            tot_recompute += prompt - cached
            tot_output += output
            tot_calls += len(s)
        out["programs"][p] = row
    tot = {f"done_{w}s": sum(v[f"done_{w}s"] for v in out["programs"].values()) for w in windows}
    tot["done_full_load"] = sum(v["done_full_load"] for v in out["programs"].values())
    tot["sessions_per_min_full_load"] = _rate(tot["done_full_load"], full_load)
    whole = end - start
    tot["calls_per_min"] = _rate(tot_calls, whole)
    tot["prompt_ktokens_per_min"] = _rate(tot_prompt / 1000, whole)
    tot["recompute_ktokens_per_min"] = _rate(tot_recompute / 1000, whole)
    tot["output_ktokens_per_min"] = _rate(tot_output / 1000, whole)
    out["all"] = tot
    return out


# ---------------------------------------------------------------------------- one level

def run_level(out_dir: str, lanes: Dict[str, int], sessions: Dict[str, int], *, warmup: int = 1,
              wrap_fixed: bool = False, server: str = "localhost:8964", server_log: str = "",
              request_log: str = "", timeout_s: float = 3600.0, windows: Tuple[int, ...] = (600, 900, 1200),
              register: bool = True, tag: str = "") -> dict:
    """Run one level into a fresh out_dir and return its summary (also written to summary.json)."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    tag = tag or os.path.basename(out_dir.rstrip("/"))
    tasks = {p: PROGRAMS[p]["tasks"]() for p in lanes}
    for p in lanes:
        print(f"[plan] {tag} {p}: {lanes[p]} lanes x {sessions.get(p, 0) or 'follow'} sessions, "
              f"{len(tasks[p])} inputs, wrap={wrap_fixed}", flush=True)
    if register:
        register_programs(server, list(lanes))

    log_start = os.path.getsize(server_log) if server_log and os.path.exists(server_log) else 0
    t_level_start_ns = time.time_ns()
    runner = Runner(out_dir, timeout_s)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"tag": tag, "lanes": lanes, "sessions": sessions, "warmup": warmup, "wrap_fixed": wrap_fixed,
                   "timeout_s": timeout_s, "windows": list(windows), "server": server, "server_log": server_log,
                   "server_log_offset": log_start, "request_log": request_log,
                   "task_counts": {p: len(t) for p, t in tasks.items()}, "jac": JAC,
                   "created_time_ns": t_level_start_ns,
                   "environment": {k: v for k, v in runner.env.items() if k.startswith(("FC_", "BF_", "CA_", "DA_"))}},
                  f, indent=2)

    for p in lanes:
        for _ in range(warmup):
            task = runner.take(p, tasks[p], wrap=True)
            runner.run_session(p, task, "warmup")  # type: ignore[arg-type]
    runner.next_idx.clear()   # warmup is not part of the measured set: every level measures from input 0

    threads = []
    for p, n in lanes.items():
        for i in range(n):
            t = threading.Thread(target=runner.lane, args=(p, tasks[p], sessions.get(p, 0), wrap_fixed),
                                 name=f"{p}-{i}")
            threads.append(t)
    t_start = time.time()
    runner.event("measurement_start", t_start=t_start, lanes=lanes, sessions=sessions)
    for t in threads:
        t.start()
    fixed = [t for t in threads if sessions.get(t.name.rsplit("-", 1)[0], 0) > 0]
    for t in fixed:
        t.join()
    runner.stop.set()
    for t in threads:
        t.join()
    runner.event("measurement_end")
    runner.close()
    t_level_end_ns = time.time_ns()

    server_stats: Dict[str, dict] = {}
    if server_log and os.path.exists(server_log):
        n = copy_slice(server_log, os.path.join(out_dir, "server.log"), log_start)
        print(f"[level] server.log slice: {n} bytes from offset {log_start}", flush=True)
        pids = {r["pid"]: r["program"] for r in runner.records if r["phase"] == "measured"}
        server_stats = parse_server_log(os.path.join(out_dir, "server.log"), pids)
    if request_log and os.path.exists(request_log):
        n = copy_trace_window(request_log, os.path.join(out_dir, "requests.jsonl"), t_level_start_ns, t_level_end_ns)
        print(f"[level] requests.jsonl slice: {n} events", flush=True)

    summary = summarize(runner.records, server_stats, list(windows))
    summary.update({"tag": tag, "lanes": lanes, "sessions": sessions, "wrap_fixed": wrap_fixed,
                    "level_start_ns": t_level_start_ns, "level_end_ns": t_level_end_ns,
                    "measured_elapsed_s": round(time.time() - t_start, 1)})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{tag}] full-load window {summary['full_load_s']} s, "
          f"{summary['all']['sessions_per_min_full_load']} sessions/min, "
          f"{summary['all']['prompt_ktokens_per_min']} prompt ktok/min, "
          f"{summary['all']['recompute_ktokens_per_min']} recompute ktok/min; "
          + ", ".join(f"{k}={v}" for k, v in summary["all"].items() if k.startswith("done_")), flush=True)
    for p, row in summary["programs"].items():
        print(f"[{tag}/{p}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return summary


def parse_sessions(spec: str, lanes: Dict[str, int]) -> Dict[str, int]:
    if "=" in spec:
        return {k: int(v) for k, v in (kv.split("=") for kv in spec.split(","))}
    return {p: int(spec) for p in lanes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="fresh directory for this level's raw data and summary")
    ap.add_argument("--lanes", required=True, help="program=lanes,...")
    ap.add_argument("--sessions", default="0", help="per lane: N, or program=N,... (0 = follow)")
    ap.add_argument("--wrap", action="store_true", help="fixed lanes wrap around the dataset (plan_on_greatlake.md)")
    ap.add_argument("--warmup", type=int, default=1, help="sequential sessions per program before measuring")
    ap.add_argument("--server", default="localhost:8964")
    ap.add_argument("--server-log", default="", help="the server's stdout; its slice for this level is copied")
    ap.add_argument("--request-log", default="", help="the server's BENCH_REQUEST_LOG; its window is copied")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--windows", default="600,900,1200", help="fixed windows (s) for completion counts")
    ap.add_argument("--timeout", type=float, default=3600.0, help="per session (s)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lanes = {k: int(v) for k, v in (kv.split("=") for kv in a.lanes.split(","))}
    sessions = parse_sessions(a.sessions, lanes)
    if a.dry_run:
        for p in lanes:
            print(f"[plan] {p}: {lanes[p]} lanes x {sessions.get(p, 0) or 'follow'} sessions, "
                  f"{len(PROGRAMS[p]['tasks']())} inputs, wrap={a.wrap}")
        return 0
    run_level(a.out_dir, lanes, sessions, warmup=a.warmup, wrap_fixed=a.wrap, server=a.server,
              server_log=a.server_log, request_log=a.request_log, timeout_s=a.timeout,
              windows=tuple(int(w) for w in a.windows.split(",")), register=not a.no_register)
    return 0


if __name__ == "__main__":
    sys.exit(main())
