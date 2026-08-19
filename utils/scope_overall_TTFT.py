#!/usr/bin/env python3
"""End-to-end TTFT scoping from GuardServer's /stats event log.

stats.calls carries no timestamps, so the user-visible figure — program start
until the first token the user actually sees — is reconstructed from the Monitor
event log (guard_server.py:50, whose `t` /stats already normalizes to the first
event):

    first_token(call) = t[call_end] - duration + ttft
    E2E TTFT          = first_token(target call) - t[first call_start]

That is the definition evaluations/demo/support_agent.jac prescribes. The target
defaults to the last call in the log; pin it with --last-key when the flow ends
in calls the user never sees.

The gaps between one call_end and the next call_start are client turnaround and
tool execution: they count in E2E, appear in no call's ttft, and are exactly the
idle windows the scheduler runs speculation in — so each gap is annotated with
the warms that landed inside it.

Caveat carried from docs/architecture.md: E2E is decode-dominated and diverges
between conditions (outputs differ in length), so treat the span as a scoping
figure and A/B on matched calls (scope_per_segment_TTFT.py).

usage:
  python3 utils/scope_overall_TTFT.py                              # live server
  python3 utils/scope_overall_TTFT.py run.json --last-key resolve_case
  python3 utils/scope_overall_TTFT.py base.json spec.json          # A/B of the spans
  python3 utils/scope_overall_TTFT.py --verbose                    # full event timeline
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_URL = "http://localhost:8964/stats"


def load(src: str) -> Dict[str, Any]:
    if src.startswith("http://") or src.startswith("https://"):
        try:
            with urllib.request.urlopen(src, timeout=30) as r:
                return json.load(r)
        except OSError as e:
            raise SystemExit(f"cannot reach {src} ({e}) — is guard_server.py up on that port?")
    with open(src) as f:
        return json.load(f)


def ms(x: Optional[float]) -> Optional[float]:
    return None if x is None else x * 1000.0


def fmt(x: Optional[float], width: int = 9, prec: int = 1) -> str:
    return f"{'-':>{width}}" if x is None else f"{x:>{width}.{prec}f}"


def build_calls(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair each call_end with its most recent unconsumed call_start of the same
    key, and place the first token inside that window. Keyless serves (the
    '(generic)' rows in stats.calls) emit no events — guard_server.py gates the
    monitor on `key` — so they are invisible here by construction."""
    pending: Dict[str, List[float]] = {}
    out: List[Dict[str, Any]] = []
    for e in events:
        kind, key = e.get("kind"), str(e.get("key", ""))
        if kind == "call_start":
            pending.setdefault(key, []).append(float(e["t"]))
        elif kind == "call_end":
            t_end = float(e["t"])
            starts = pending.get(key) or []
            t_start = starts.pop() if starts else None
            ttft, dur = e.get("ttft"), e.get("duration")
            t_ft = (t_end - dur + ttft) if (dur is not None and ttft is not None) else None
            out.append({"key": key, "t_start": t_start, "t_ft": t_ft, "t_end": t_end,
                        "ttft": ttft, "duration": dur})
    return out


def pick_target(calls: List[Dict[str, Any]], last_key: str, nth: Optional[int]) -> int:
    if nth is not None:
        return nth if nth >= 0 else len(calls) + nth
    if last_key:
        hits = [i for i, c in enumerate(calls) if c["key"] == last_key]
        if not hits:
            raise SystemExit(f"no call_end for key {last_key!r}; keys seen: {sorted({c['key'] for c in calls})}")
        return hits[-1]
    return len(calls) - 1


def warms_in(events: List[Dict[str, Any]], t0: float, t1: float) -> Tuple[int, int, float]:
    """Speculative work that executed inside a window: (done, preempted, engine ms)."""
    done = [e for e in events if e.get("kind") == "warm_done" and t0 <= float(e["t"]) <= t1]
    pre = [e for e in events if e.get("kind") == "warm_preempted" and t0 <= float(e["t"]) <= t1]
    return len(done), len(pre), sum(ms(e.get("duration")) or 0.0 for e in done)


def report_single(data: Dict[str, Any], last_key: str, nth: Optional[int], verbose: bool) -> Dict[str, Any]:
    events = data.get("events") or []
    calls = build_calls(events)
    if not calls:
        print("no keyed calls in the event log (only '(generic)' serves, or the run never started)")
        return {"calls": 0}
    ti = pick_target(calls, last_key, nth)
    target = calls[ti]
    t_origin = calls[0]["t_start"] if calls[0]["t_start"] is not None else calls[0]["t_end"]
    width = max([len(c["key"]) for c in calls] + [len("call site")])

    print(f"{'#':>3}  {'call site':<{width}} {'start s':>8} {'1st tok':>8} {'end s':>8} {'TTFT ms':>9} {'decode ms':>10} {'gap after':>10}  idle-window warms")
    gaps = 0.0
    sigma_ttft = 0.0
    decode_before = 0.0
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(calls):
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        gap = None
        if nxt is not None and nxt["t_start"] is not None:
            gap = nxt["t_start"] - c["t_end"]
        note = ""
        if gap is not None and gap > 0:
            d, p, eng = warms_in(events, c["t_end"], nxt["t_start"])
            if d or p:
                note = f"{d} warm(s), {eng:.0f} ms engine" + (f", {p} preempted" if p else "")
        if i <= ti:
            sigma_ttft += ms(c["ttft"]) or 0.0
            if i < ti:
                decode_before += (ms(c["duration"]) or 0.0) - (ms(c["ttft"]) or 0.0)
                gaps += (gap or 0.0) * 1000.0
        mark = " <- E2E ends here" if i == ti else ""
        print(f"{i:>3}  {c['key']:<{width}} {fmt(c['t_start'], 8, 3)} {fmt(c['t_ft'], 8, 3)} {fmt(c['t_end'], 8, 3)} "
              f"{fmt(ms(c['ttft']))} {fmt((ms(c['duration']) or 0) - (ms(c['ttft']) or 0), 10)} {fmt((gap or 0) * 1000 if gap is not None else None, 10)}  {note}{mark}")
        rows.append({"i": i, "key": c["key"], "t_start": c["t_start"], "t_first_token": c["t_ft"],
                     "t_end": c["t_end"], "ttft_ms": ms(c["ttft"]), "gap_after_ms": (gap or 0) * 1000 if gap is not None else None})

    if target["t_ft"] is None or t_origin is None:
        print("\ntarget call has no usable timestamps (missing ttft/duration)")
        return {"calls": len(calls)}
    span = (target["t_ft"] - t_origin) * 1000.0
    print(f"\nfirst call_start        {t_origin:8.3f} s   ({calls[0]['key']})")
    print(f"target first token      {target['t_ft']:8.3f} s   ({target['key']}, call #{ti})")
    print(f"OVERALL (E2E) TTFT      {span:8.1f} ms")
    print(f"\n  decomposition (sums to the span):")
    print(f"    prefill / TTFT of calls 0..{ti}   {sigma_ttft:9.1f} ms   ({sigma_ttft / span * 100:4.1f}%)  <- what proactive prefill attacks")
    print(f"    decode of calls 0..{ti - 1}          {decode_before:9.1f} ms   ({decode_before / span * 100:4.1f}%)")
    print(f"    gaps (client + tool time)      {gaps:9.1f} ms   ({gaps / span * 100:4.1f}%)  <- the idle windows speculation runs in")
    resid = span - sigma_ttft - decode_before - gaps
    if abs(resid) > 1.0:
        print(f"    unaccounted                    {resid:9.1f} ms   (overlapping calls or a missing call_start)")
    d, p, eng = warms_in(events, t_origin, target["t_ft"])
    print(f"\n  inside the span: {d} warm(s) executed ({eng:.0f} ms engine), {p} preempted by a real arrival")
    if verbose:
        print("\n  event timeline:")
        for e in events:
            extra = {k: v for k, v in e.items() if k not in ("t", "kind", "key")}
            print(f"    [{float(e['t']):8.3f}s] {str(e.get('kind')):<15} {e.get('key')}  {extra if extra else ''}")
    return {"calls": len(calls), "target": target["key"], "target_index": ti, "e2e_ttft_ms": span,
            "sigma_ttft_ms": sigma_ttft, "decode_ms": decode_before, "gap_ms": gaps, "rows": rows}


def report_ab(base: Dict[str, Any], test: Dict[str, Any], names: List[str], last_key: str, nth: Optional[int]) -> Dict[str, Any]:
    out = []
    for data, name in zip((base, test), names):
        print(f"=== {name} " + "=" * max(0, 60 - len(name)))
        out.append(report_single(data, last_key, nth, verbose=False))
        print()
    b, t = out[0].get("e2e_ttft_ms"), out[1].get("e2e_ttft_ms")
    if b and t:
        print(f"E2E TTFT  {b:.1f} ms -> {t:.1f} ms   ({b / t:.2f}x)")
        sb, st_ = out[0].get("sigma_ttft_ms"), out[1].get("sigma_ttft_ms")
        if sb and st_:
            print(f"  of which prefill: {sb:.1f} -> {st_:.1f} ms ({sb / st_:.2f}x)")
        print("  note: E2E is decode-dominated and trajectory-divergent — matched-call TTFT (scope_per_segment_TTFT.py) is the honest A/B unit.")
    return {"base": out[0], "test": out[1], "speedup": (b / t) if (b and t) else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="end-to-end TTFT from GuardServer /stats (program start -> the first token the user sees)")
    ap.add_argument("sources", nargs="*", help="0 args = live server; 1 = one dump/URL; 2 = baseline then test")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"live /stats endpoint (default {DEFAULT_URL})")
    ap.add_argument("--last-key", default="", help="call site whose first token ends the span (default: the last call in the log)")
    ap.add_argument("--nth", type=int, help="index into the call list instead of a key (negative counts from the end)")
    ap.add_argument("--save", metavar="PATH", help="write the fetched payload to PATH")
    ap.add_argument("--verbose", action="store_true", help="also print the raw event timeline (speculate / warm_done / ret_record)")
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
    out = (report_ab(data[0], data[1], srcs, args.last_key, args.nth) if len(data) == 2
           else report_single(data[0], args.last_key, args.nth, args.verbose))
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
