"""Replay synthetic programs against the server and report per-call wall times.

    python -m benchmark.run highload [--workers 16] [--sessions 6] [--stagger 0.15]
    python -m benchmark.run gappy    [--sessions 4] [--gap 4.0]

highload — the KV-pressure regime: W concurrent closed-loop workers, each its
own `flow` program with a distinct ~2.2k-token static prefix; together the
prefixes overflow the device KV pool, and with no think-time sleeps the only
planner windows are the micro-gaps between a reply and the next request.
Arrival pattern: one thread per worker, started `--stagger` seconds apart,
then each worker runs its sessions back-to-back (the next request leaves only
when the previous reply is in).

gappy — the idle-window regime: sequential sessions with real tool-gap sleeps,
one flow worker plus one visit worker, so full-prompt prefill and the routing
probe get a stage.

Client wall times print per (session, call); the interesting server metrics
(ttft_ms, cached_device/cached_host, [prefill]/[probe]/[steer], drift) are in
the server log. Compare a run against `--no-spec` on a fresh server each time —
programs are learned online and in memory only.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

from benchmark.programs import (ROUTE_SYS, SYS, VISIT_USER, act_user, report_user,
                                scout_user, service_system, step_user)


def post(base: str, path: str, body: dict):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def chat(base: str, model: str, sid: str, system: str, user: str):
    t0 = time.time()
    out = post(base, "/v1/chat/completions", {
        "model": model, "user": sid, "temperature": 0, "max_tokens": 32,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]})
    return out["choices"][0]["message"]["content"] or "", time.time() - t0


def flow_session(base: str, model: str, sid: str, w: int, s: int, gap: float):
    """step -> (gap) -> report carrying step's reply; returns the two wall times."""
    system = service_system(w)
    r1, t1 = chat(base, model, sid, system, step_user(w, s))
    if gap:
        time.sleep(gap)
    _, t2 = chat(base, model, sid, system, report_user(w, r1))
    post(base, "/v1/sessions/close", {"user": sid})
    return [t1, t2]


def visit_session(base: str, model: str, sid: str, gap: float):
    """announce -> (gap) -> route -> (gap) -> act; returns the three wall times."""
    _, t1 = chat(base, model, sid, SYS, scout_user())
    if gap:
        time.sleep(gap)
    _, t2 = chat(base, model, sid, ROUTE_SYS, VISIT_USER)
    if gap:
        time.sleep(gap)
    _, t3 = chat(base, model, sid, SYS, act_user())
    post(base, "/v1/sessions/close", {"user": sid})
    return [t1, t2, t3]


def report(tag: str, results, wall: float) -> None:
    print(f"[{tag}] total_wall={wall:.1f}s calls={len(results)}")
    for s in sorted({s for s, _, _ in results}):
        for c in sorted({c for s_, c, _ in results if s_ == s}):
            xs = sorted(t for s_, c_, t in results if s_ == s and c_ == c)
            print(f"[{tag}] session={s} call={c} mean={statistics.mean(xs):.3f}s "
                  f"p95={xs[int(0.95 * (len(xs) - 1))]:.3f}s n={len(xs)}")


def highload(args) -> None:
    results = []   # (session index, call index, wall_s)
    lock = threading.Lock()

    def worker(w: int) -> None:
        for s in range(1, args.sessions + 1):
            walls = flow_session(args.base, args.model, f"hl-{w}-{s}", w, s, gap=0.0)
            with lock:
                results.extend((s, c + 1, t) for c, t in enumerate(walls))

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
    for t in threads:
        t.start()
        time.sleep(args.stagger)   # staggered arrivals, like real traffic
    for t in threads:
        t.join()
    report(args.tag, results, time.time() - t0)


def gappy(args) -> None:
    results = []
    t0 = time.time()
    for s in range(1, args.sessions + 1):
        walls = flow_session(args.base, args.model, f"drv-{s}", 0, s, gap=args.gap)
        results.extend((s, c + 1, t) for c, t in enumerate(walls))
        walls = visit_session(args.base, args.model, f"vis-{s}", gap=args.gap)
        results.extend((s, c + 1, t) for c, t in enumerate(walls))
    report(args.tag, results, time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["highload", "gappy"])
    ap.add_argument("--base", default="http://127.0.0.1:8964")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--stagger", type=float, default=0.15)
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.tag is None:
        args.tag = args.mode
    (highload if args.mode == "highload" else gappy)(args)
