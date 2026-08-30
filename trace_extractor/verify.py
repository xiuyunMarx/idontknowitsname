"""Check a recovered workflow against measured ground truth.

    python -m trace_extractor.verify results/trace.jsonl benchmark/workload/profiles.json

profiles.json (benchmark/workload/profile.py) holds, per case, the callsite chain the old
compiler-keyed server recorded (`Owner.method@LINE:COL`, `__visit@file:line`, kinds
call/tool_turn/generate). The trace holds what the history-based server saw. Both are
reduced to per-session label sequences with ReAct turns folded and compared program by
program, session by session (profile.py runs cases sequentially, so the i-th session of a
program in the trace is the i-th case in profiles.json)."""
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List

from .parser import build_programs, load_trace
from .primitives import CallsiteType, Program

_AT = re.compile(r"@.*$")


def truth_sequences(profiles: dict) -> Dict[str, List[List[str]]]:
    """program -> [folded label sequence per case, in case order]."""
    out: Dict[str, List[List[str]]] = defaultdict(list)
    cases = sorted(profiles["cases"].items(), key=lambda kv: (kv[1]["program"], kv[1]["case_index"]))
    for _, case in cases:
        if case.get("rc", 0) != 0:
            continue
        seq: List[str] = []
        for step in case["chain"]:
            if step["kind"] == "tool_turn" and seq:
                continue
            label = step["callsite"]
            label = "visit" if label.startswith("__visit@") else _AT.sub("", label)
            seq.append(label)
        name = case["program"].rsplit("/", 1)[-1].replace(".jac", "")
        out[name].append(seq)
    return out


def recovered_sequences(programs: Dict[str, Program]) -> Dict[str, List[List[str]]]:
    out: Dict[str, List[List[str]]] = {}
    for name, p in programs.items():
        sessions = sorted(p.sessions.values(), key=lambda s: s.calls[0].t_arrive if s.calls else 0)
        out[name] = [["visit" if p.callsites[c.key].kind is CallsiteType.VISITBY else p.label(c.key) for c in s.calls]
                     for s in sessions]
    return out


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    programs = build_programs(load_trace(argv[0]))
    with open(argv[1]) as f:
        truth = truth_sequences(json.load(f))
    got = recovered_sequences(programs)
    bad = 0
    for name in sorted(set(truth) | set(got)):
        t, g = truth.get(name, []), got.get(name, [])
        print(f"== {name}: {len(g)} recovered session(s), {len(t)} ground-truth case(s), "
              f"{len(programs[name].callsites) if name in programs else 0} callsite(s)")
        for i, seq in enumerate(g):
            ref = t[i] if i < len(t) else None
            ok = ref == seq
            bad += not ok
            mark = "ok  " if ok else ("??  " if ref is None else "DIFF")
            print(f"  {mark} #{i}: " + " -> ".join(seq))
            if ref is not None and not ok:
                print(f"       truth: " + " -> ".join(ref))
        if name in programs:
            p = programs[name]
            for key, turns in p.turns.items():
                if max(turns) > 1:
                    print(f"  turns {p.label(key)}: {turns}")
            for (a, b), c in p.succ2.items():
                if a == b or any(k == a for k in c):
                    print(f"  loop  {p.label(a)} -> {p.label(b)} -> {dict((p.label(k), n) for k, n in c.items())}")
    print(f"\n{bad} session(s) differ from ground truth")
    for p in programs.values():
        print()
        print(p.describe())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
