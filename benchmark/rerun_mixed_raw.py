"""Run the eleven 4x4 arms with isolated output and recoverable raw records.

Run with the sglang Python environment. --plan prints the exact configuration.
The batch stops on an infrastructure failure; individual failed sessions remain
in the data. It never selects a publication measurement window.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
ARMS = [
    ("ours", "server.server", []),
    ("relayout", "server.server", ["--lru"]),
    ("vanilla", "server.server", ["--lru", "--no-relayout"]),
    ("kvonly", "server.server", ["--no-relayout"]),
    ("ours_noprefetch", "server.server", ["--no-promote"]),
    ("continuum", "server.continuum_server", []),
    ("cachescout", "server.cacheScout_server", []),
    ("kvflow", "server.KVFlow_server", []),
    ("continuum_relayout", "server.continuum_server", ["--relayout"]),
    ("cachescout_relayout", "server.cacheScout_server", ["--relayout"]),
    ("kvflow_relayout", "server.KVFlow_server", ["--relayout"]),
]
LANES = "fact_check=4,BFCL_agent=4,coding_agent=4,doc_analysis=4"
SESSIONS = "coding_agent=6,fact_check=0,BFCL_agent=0,doc_analysis=0"
SETTINGS = {"MODEL": "Qwen/Qwen3-14B-AWQ", "HOST": "10", "KV": "31830",
            "SCHED": "fcfs", "GRAMMAR_BACKEND": "xgrammar",
            "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "4096",
            "FC_MIN_ROUNDS": "2", "FC_TOOL_DELAY_S": "2", "BF_TOOL_DELAY_S": "2",
            "CA_TOOL_DELAY_S": "2", "DA_DOC_CHARS": "40000", "DA_PAGE_WINDOW": "4"}


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sources():
    paths = set()
    for directory in ["server", "model", "static_analysis"]:
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update((ROOT / "benchmark").glob("*.py"))
    paths.update((ROOT / "benchmark/applications").glob("*.jac"))
    return sorted(p for p in paths if p.is_file())


def hashes(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(3)
        # Children may still be alive even if the parent has exited.
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def stop_existing():
    modules = {module for _, module, _ in ARMS} | {"benchmark.mixed_workload"}
    selected = {}
    for proc in psutil.process_iter(["pid", "cmdline", "uids"]):
        args = proc.info["cmdline"] or []
        if proc.info["uids"].real != os.getuid() or "-m" not in args:
            continue
        index = args.index("-m")
        if index + 1 >= len(args) or args[index + 1] not in modules:
            continue
        try:
            for child in proc.children(recursive=True) + [proc]:
                selected[child.pid] = child
        except psutil.NoSuchProcess:
            pass
    print("Stopping existing benchmark/server processes:", sorted(selected), flush=True)
    for proc in selected.values():
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(selected.values()), timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def archive(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
        tar.add(source, arcname=source.name)
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "  " + destination.name + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--backup-dir", type=Path, required=True)
    ap.add_argument("--stop-existing", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    batch = args.batch_dir.resolve()
    backup = args.backup_dir.resolve()
    config = {"arms": ARMS, "lanes": LANES, "sessions": SESSIONS,
              "warmup_sessions_per_program": 1, "session_timeout_s": 1800,
              "environment": SETTINGS, "python": sys.executable,
              "batch_dir": str(batch), "backup_dir": str(backup),
              "analysis_window": None, "fresh_server_per_arm": True,
              "tool_cache_policy": "independent copies of the same initial caches"}
    if args.plan:
        config["validated_source_files"] = len(hashes(sources()))
        print(json.dumps(config, indent=2))
        return 0
    batch.mkdir(parents=True, exist_ok=False)
    backup.mkdir(parents=True, exist_ok=False)
    save(batch / "config.json", config)
    state = {"pid": os.getpid(), "status": "preparing", "started_time_ns": time.time_ns(), "arms": {}}
    save(batch / "status.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        paths = sources()
        baseline = hashes(paths)
        snapshot = batch / "source_snapshot"
        for src in paths:
            dst = snapshot / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        save(snapshot / "sha256.json", baseline)
        (snapshot / "git_head.txt").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT))
        (snapshot / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--", "server", "model", "static_analysis", "benchmark/*.py", "benchmark/applications/*.jac"], cwd=ROOT))
        (snapshot / "packages.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=ROOT))
        data = ROOT / "benchmark/applications/bench_data"
        for relative in ["HoVer/hover_claims_120.tsv", "HumanEval/tasks.txt", "BFCL/BFCL_v4_web_search.json", "finicial_bench/financebench_open_source.jsonl"]:
            dst = snapshot / "inputs" / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data / relative, dst)
        caches = {"FC_CACHE_DIR": data / "HoVer/wiki_cache", "BF_CACHE_DIR": data / "BFCL/web_cache"}
        for key, src in caches.items():
            shutil.copytree(src, batch / "initial_tool_cache" / key)
        archive(snapshot, backup / "source_snapshot.tar.gz")
        archive(batch / "initial_tool_cache", backup / "initial_tool_cache.tar.gz")
        shutil.copy2(batch / "config.json", backup / "config.json")
        if args.stop_existing:
            stop_existing()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), stopped_time_ns=time.time_ns())
        save(batch / "status.json", state)
        save(backup / "status.json", state)
        raise

    server = driver = None
    current = None
    try:
        for arm, module, flags in ARMS:
            if hashes(sources()) != baseline:
                raise RuntimeError("Source files changed during the batch; refusing to mix code versions")
            current = batch / arm
            current.mkdir()
            env = dict(os.environ)
            env.update(SETTINGS)
            env["PATH"] = str(Path(sys.executable).parent) + ":" + env.get("PATH", "")
            env["BENCH_REQUEST_LOG"] = str(current / "requests.jsonl")
            # Explicitly pin input paths so inherited shell settings cannot change the workload.
            env["DA_DATA"] = str(data / "finicial_bench/financebench_open_source.jsonl")
            env["DA_PAGES_DIR"] = str(data / "finicial_bench/pages")
            for key in caches:
                dst = current / "tool_cache" / key
                shutil.copytree(batch / "initial_tool_cache" / key, dst)
                env[key] = str(dst)
            slog = current / "server.log"
            scmd = [sys.executable, "-u", "-m", module, SETTINGS["MODEL"], *flags,
                    "--sched", "fcfs", "--host", "10", "--kv", "31830", "--engine-log", "info"]
            dcmd = [sys.executable, "-u", "-m", "benchmark.mixed_workload",
                    "--tag", batch.name + "_" + arm, "--lanes", LANES, "--sessions", SESSIONS,
                    "--warmup", "1", "--timeout", "1800", "--server-log", str(slog),
                    "--out-dir", str(current / "driver")]
            save(current / "launch.json", {"server": scmd, "driver": dcmd,
                                           "environment": {k: env[k] for k in [*SETTINGS, *caches, "BENCH_REQUEST_LOG", "DA_DATA", "DA_PAGES_DIR"]}})
            state.update(status="running", current_arm=arm)
            state["arms"][arm] = {"status": "starting_server", "started_time_ns": time.time_ns()}
            save(batch / "status.json", state)
            print(time.strftime("%F %T"), arm, "starting server", flush=True)
            with slog.open("x") as out:
                server = subprocess.Popen(scmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while "[warmup] engine ready" not in slog.read_text(errors="replace"):
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"{arm} server failed to become ready; see {slog}")
                time.sleep(3)
            with (current / "driver.out").open("x") as out:
                driver = subprocess.Popen(dcmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            state["arms"][arm].update(status="running", server_pid=server.pid, driver_pid=driver.pid)
            save(batch / "status.json", state)
            print(time.strftime("%F %T"), arm, "driver started", driver.pid, flush=True)
            deadline = time.monotonic() + 8 * 3600
            while driver.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"{arm} server exited while workload was running")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{arm} exceeded the eight-hour run limit")
                time.sleep(5)
            if driver.returncode:
                raise RuntimeError(f"{arm} driver exited {driver.returncode}; see driver.out")
            stop_group(driver)
            driver = None
            stop_group(server)
            server = None
            summary = json.loads((current / "driver/summary.json").read_text())
            if not (current / "requests.jsonl").stat().st_size:
                raise RuntimeError(f"{arm} has no raw request events")
            failed = sum(p["failed"] for p in summary["programs"].values())
            state["arms"][arm].update(status="complete_with_session_failures" if failed else "complete",
                                      failed_sessions=failed, finished_time_ns=time.time_ns())
            archive(current, backup / (arm + ".tar.gz"))
            save(batch / "status.json", state)
            save(backup / "status.json", state)
            print(time.strftime("%F %T"), arm, "complete; backed up; failed_sessions=", failed, flush=True)
        state.update(status="complete", finished_time_ns=time.time_ns())
        print("MIX4X4-RAW-DONE", flush=True)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=repr(exc), stopped_time_ns=time.time_ns())
        raise
    finally:
        stop_group(driver)
        stop_group(server)
        save(batch / "status.json", state)
        save(backup / "status.json", state)
        if current is not None and state["status"] != "complete":
            archive(current, backup / (current.name + "_partial.tar.gz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
