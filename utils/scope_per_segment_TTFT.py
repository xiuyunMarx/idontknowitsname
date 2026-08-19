#!/usr/bin/env python3
"""Per-call-site TTFT scoping from GuardServer's /stats.

Every served completion is appended to stats.calls (guard_server.py:387 for the
/call path, :476 for the /generate wire byllm actually uses) carrying the
wall-clock TTFT measured in _generate (guard_server.py:172 — request submitted
until its first token, queueing included) and the engine's cached_tokens, which
is the number that says WHY a TTFT moved: a warm that landed shows up as cached
tokens, not as a guess. One row per served completion, so a ReAct ability
contributes one row per turn, not one per `def`.

Two units matter here, in this order:
  per-call TTFT  — the honest A/B unit (matched call sites), because vLLM
                   numerics differ across batch compositions and greedy
                   trajectories diverge between conditions;
  SIGMA TTFT     — the sum over the flow's calls, the headline figure in
                   docs/architecture.md. E2E span lives in scope_overall_TTFT.py.

usage:
  python3 utils/scope_per_segment_TTFT.py                        # live server
  python3 utils/scope_per_segment_TTFT.py run.json               # saved dump
  python3 utils/scope_per_segment_TTFT.py --save run.json        # fetch and keep
  python3 utils/scope_per_segment_TTFT.py base.json spec.json    # A/B, matched calls
  python3 utils/scope_per_segment_TTFT.py --group                # aggregate per site
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_URL = "http://localhost:8964/stats"


def load(src: str) -> Dict[str, Any]:
    """A live /stats URL or a saved JSON dump of one."""
    if src.startswith("http://") or src.startswith("https://"):
        try:
            with urllib.request.urlopen(src, timeout=30) as r:
                return json.load(r)
        except OSError as e:
            raise SystemExit(f"cannot reach {src} ({e}) — is guard_server.py up on that port?")
    with open(src) as f:
        return json.load(f)


def calls_of(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((data.get("stats") or {}).get("calls") or [])


def ms(x: Optional[float]) -> Optional[float]:
    return None if x is None else x * 1000.0


def fmt(x: Optional[float], width: int = 9, prec: int = 1) -> str:
    return f"{'-':>{width}}" if x is None else f"{x:>{width}.{prec}f}"


def occurrence_keys(calls: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """(key, nth call of that key) — the identity used to match A/B runs, so a
    multi-turn ReAct site lines up turn by turn instead of collapsing."""
    seen: Dict[str, int] = {}
    out: List[Tuple[str, int]] = []
    for c in calls:
        k = str(c.get("key", ""))
        seen[k] = seen.get(k, 0) + 1
        out.append((k, seen[k]))
    return out


# --------------------------------------------------------------- single run
def report_single(data: Dict[str, Any], group: bool) -> Dict[str, Any]:
    calls = calls_of(data)
    if not calls:
        print("no calls recorded (is the client run finished? is this the right server?)")
        return {"calls": 0}
    width = max([len(str(c.get("key", ""))) for c in calls] + [len("call site")])
    total = 0.0
    rows: List[Dict[str, Any]] = []
    if group:
        agg: Dict[str, List[Dict[str, Any]]] = {}
        for c in calls:
            agg.setdefault(str(c.get("key", "")), []).append(c)
        print(f"{'call site':<{width}} {'n':>3} {'mean TTFT':>10} {'min':>8} {'max':>8} {'mean cached':>12}")
        for key, cs in agg.items():
            ts = [ms(c.get("ttft")) for c in cs if c.get("ttft") is not None]
            cached = [c.get("cached_tokens") for c in cs if c.get("cached_tokens") is not None]
            mean = sum(ts) / len(ts) if ts else None
            mean_c = sum(cached) / len(cached) if cached else None
            total += sum(ts or [])
            print(f"{key:<{width}} {len(cs):>3} {fmt(mean, 10)} {fmt(min(ts) if ts else None, 8)} {fmt(max(ts) if ts else None, 8)} {fmt(mean_c, 12, 0)}")
            rows.append({"key": key, "n": len(cs), "mean_ttft_ms": mean, "mean_cached": mean_c})
    else:
        print(f"{'#':>3}  {'call site':<{width}} {'TTFT ms':>9} {'cached':>7} {'decode ms':>10} {'total ms':>9}")
        for i, c in enumerate(calls):
            t = ms(c.get("ttft"))
            d = ms(c.get("duration"))
            decode = None if (t is None or d is None) else d - t
            total += t or 0.0
            print(f"{i:>3}  {str(c.get('key', '')):<{width}} {fmt(t)} {str(c.get('cached_tokens', '-')):>7} {fmt(decode, 10)} {fmt(d, 9)}")
            rows.append({"i": i, "key": c.get("key"), "ttft_ms": t, "cached_tokens": c.get("cached_tokens"), "duration_ms": d})
    n = len(calls)
    print(f"\nSIGMA TTFT {total:.1f} ms over {n} call(s); mean {total / n:.1f} ms")
    print(speculation_line(data))
    return {"calls": n, "sigma_ttft_ms": total, "mean_ttft_ms": total / n, "rows": rows}


def speculation_line(data: Dict[str, Any]) -> str:
    """One line of context: did speculation actually run, and did it cost anything?"""
    st = data.get("stats") or {}
    warms = st.get("warms") or []
    done = [w for w in warms if "duration" in w]
    dup = [w for w in warms if w.get("skipped") == "dup"]
    unreach = [w for w in warms if w.get("skipped") == "unreachable"]
    fed = sum(int(w.get("fed") or 0) for w in done)
    sched = data.get("scheduler") or {}
    exec_ms = sum(ms(w.get("duration")) or 0.0 for w in done)
    park_ms = sum(ms(w.get("parked")) or 0.0 for w in done)
    return (f"warms: {len(done)} executed ({exec_ms:.0f} ms engine, {park_ms:.0f} ms parked), "
            f"{len(dup)} dup-skipped, {len(unreach)} unreachable, {fed} binding(s) carried; "
            f"feeds recorded {len(st.get('feeds') or [])}; reorders {len(st.get('reorders') or [])}; "
            f"busy_budget {sched.get('busy_budget')}, preempt_signals {sched.get('preempt_signals')}")


# ------------------------------------------------------------------- A / B
def report_ab(base: Dict[str, Any], test: Dict[str, Any], base_name: str, test_name: str) -> Dict[str, Any]:
    """Matched-call comparison: same call site, same turn index. Calls present in
    only one run are listed separately — trajectory divergence is expected, and
    silently pairing across it is how a speedup gets faked."""
    b_calls, t_calls = calls_of(base), calls_of(test)
    b_idx = {k: c for k, c in zip(occurrence_keys(b_calls), b_calls)}
    t_idx = {k: c for k, c in zip(occurrence_keys(t_calls), t_calls)}
    matched = [k for k in b_idx if k in t_idx]
    width = max([len(k[0]) for k in b_idx] + [len(k[0]) for k in t_idx] + [len("call site")])
    print(f"baseline = {base_name}   test = {test_name}\n")
    print(f"{'call site':<{width}} {'turn':>4} {'base ms':>9} {'test ms':>9} {'speedup':>8} {'base cch':>9} {'test cch':>9}")
    rows: List[Dict[str, Any]] = []
    sb = st = 0.0
    for k in matched:
        tb, tt = ms(b_idx[k].get("ttft")), ms(t_idx[k].get("ttft"))
        sb += tb or 0.0
        st += tt or 0.0
        sp = (tb / tt) if (tb and tt) else None
        print(f"{k[0]:<{width}} {k[1]:>4} {fmt(tb)} {fmt(tt)} {(fmt(sp, 7, 2) + 'x') if sp else fmt(None, 8)} "
              f"{str(b_idx[k].get('cached_tokens', '-')):>9} {str(t_idx[k].get('cached_tokens', '-')):>9}")
        rows.append({"key": k[0], "turn": k[1], "base_ttft_ms": tb, "test_ttft_ms": tt, "speedup": sp,
                     "base_cached": b_idx[k].get("cached_tokens"), "test_cached": t_idx[k].get("cached_tokens")})
    if matched:
        sp = (sb / st) if st else None
        print(f"\n{'SIGMA (matched)':<{width}} {len(matched):>4} {fmt(sb)} {fmt(st)} {(fmt(sp, 7, 2) + 'x') if sp else ''}")
    only_b = [k for k in b_idx if k not in t_idx]
    only_t = [k for k in t_idx if k not in b_idx]
    if only_b or only_t:
        print(f"\nunmatched (trajectory divergence — excluded from the comparison):")
        for k in only_b:
            print(f"  baseline only: {k[0]} turn {k[1]}  {fmt(ms(b_idx[k].get('ttft')), 8)} ms")
        for k in only_t:
            print(f"  test only    : {k[0]} turn {k[1]}  {fmt(ms(t_idx[k].get('ttft')), 8)} ms")
    print(f"\nbaseline  {speculation_line(base)}")
    print(f"test      {speculation_line(test)}")
    return {"matched": rows, "sigma_base_ms": sb, "sigma_test_ms": st,
            "sigma_speedup": (sb / st) if st else None,
            "unmatched_base": [list(k) for k in only_b], "unmatched_test": [list(k) for k in only_t]}


def main() -> None:
    ap = argparse.ArgumentParser(description="per-call-site TTFT from GuardServer /stats (one run, or a matched-call A/B of two)")
    ap.add_argument("sources", nargs="*", help="0 args = live server; 1 = one dump/URL; 2 = baseline then test")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"live /stats endpoint (default {DEFAULT_URL})")
    ap.add_argument("--save", metavar="PATH", help="write the fetched payload to PATH (keep a run for later A/B)")
    ap.add_argument("--group", action="store_true", help="aggregate rows per call site instead of per call")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON on stdout")
    args = ap.parse_args()

    srcs = args.sources or [args.url]
    if len(srcs) > 2:
        ap.error("at most two sources (baseline, test)")
    data = [load(s) for s in srcs]
    if args.save:
        with open(args.save, "w") as f:
            json.dump(data[-1], f)
        print(f"[saved {args.save}]", file=sys.stderr)
    out = report_ab(data[0], data[1], srcs[0], srcs[1]) if len(data) == 2 else report_single(data[0], args.group)
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
