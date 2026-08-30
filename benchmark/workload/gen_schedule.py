#!/usr/bin/env python
"""Seeded, prefix-stable workload schedules for run_schedule.py.

A schedule is a list of `jac run` launches with a start offset each:

    python benchmark/workload/gen_schedule.py --pattern burst --burst-size 3 --burst-gap 25 --count 40 \
        --out benchmark/workload/schedules/burst-k3-g25.json
    python benchmark/workload/gen_schedule.py --pattern poisson --rate 0.5 --count 40 \
        --out benchmark/workload/schedules/poisson-0.5.json
    python benchmark/workload/gen_schedule.py --pattern adversarial --profiles benchmark/workload/profiles.json \
        --burst-size 3 --burst-gap 25 --count 40 --out benchmark/workload/schedules/adv-k3-g25.json

`adversarial` builds bursts on which lookahead pays: for every burst it draws `--candidates`
random program/case combinations, replays each on the offline simulator (serve/sim.py) with
the per-case chains and measured load times of profiles.json (profile.py) under FIFO, greedy
and DP, and keeps the combination with the largest greedy − DP waiting gap. The carry-over
state (which models the previous burst left on host) seeds the next burst's simulation.

Reproducible and extensible: three independent random streams (arrival gaps, program
choice, case index) are seeded from `--seed`, and every entry consumes exactly one draw
from each, so the same seed with a larger `--count` yields the smaller schedule as an
exact prefix. Arrivals are open-loop (offsets do not depend on how fast the server is)."""
import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from send_requests import PROGRAMS  # noqa: E402  program path -> (env var, number of inputs)
from serve.cost import CostModel  # noqa: E402
from serve.planner import choose, dp_order, fifo_order, greedy_order  # noqa: E402
from serve.predict import predicted_chain  # noqa: E402
from serve.sim import Scenario, simulate  # noqa: E402

EXCLUDE = {"cascade.jac"}   # no entry point since 857ae80


def default_weights() -> dict:
    return {os.path.basename(p): 1 for p in PROGRAMS
            if os.path.basename(p) not in EXCLUDE and os.path.exists(os.path.join(ROOT, p))}


class Profiles:
    """profiles.json as simulator inputs: chains per (program, case), a load table and, unless
    `oracle`, the planner's view of each chain: the branch predictor's path from the current
    callsite (a Program per .jac, trained on the profiled callsite sequences the way the
    server's ProgramInstance trains it) with per-callsite mean exec estimates."""

    def __init__(self, path: str, oracle: bool = False, exclude=()):
        with open(path) as f:
            self.data = json.load(f)
        for name in exclude:                      # cases to keep out of every schedule (e.g. flaky ones)
            self.data["cases"].pop(name, None)
        self.sha1 = hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]
        self.cost = CostModel()
        for key, secs in self.data["cost"].get("load_s", {}).items():
            model, tier = key.rsplit("|", 1)
            self.cost.observe_load(model, tier, secs)
        self.chains, self.steps = {}, {}
        for name, case in self.data["cases"].items():
            if case["rc"] == 0 and case["chain"]:
                self.chains[name] = [(c["model"], max(0.05, c["exec_s"])) for c in case["chain"]]
                self.steps[name] = [(c["callsite"], c["model"], max(0.05, c["exec_s"])) for c in case["chain"]]
        models = {m for chain in self.chains.values() for m, _ in chain}
        self.load = {m: {"host": self.cost.load_s(m, "host"), "ssd": self.cost.load_s(m, "ssd")} for m in models}
        self.programs = {} if oracle else self._train_programs()

    def _train_programs(self) -> dict:
        from static_pass.primitives import Program
        programs, call_exec = {}, {}
        for name, case in self.data["cases"].items():
            path = case["program"]
            if path not in programs:
                programs[path] = Program(name=os.path.splitext(os.path.basename(path))[0]).build_program(os.path.join(ROOT, path))
            calls = []   # (callsite, exec of the whole call), consecutive turns of one call merged
            for key, _, ex in self.steps.get(name, []):
                if calls and calls[-1][0] == key:
                    calls[-1][1] += ex
                else:
                    calls.append([key, ex])
            for (a, _), (b, _) in zip(calls, calls[1:]):
                try:
                    programs[path].update_branch_probs(a, b)
                except ValueError:
                    pass
            for key, ex in calls:
                call_exec.setdefault(key, []).append(ex)
        self.call_exec = {k: sum(v) / len(v) for k, v in call_exec.items()}
        return programs

    def lookahead(self, name: str, j: int):
        """The planner's chain for case `name` at turn `j`: the head's callsite estimate, then the
        predicted path's callsites with their estimates (as Controller.plan_input builds it)."""
        key, model, _ = self.steps[name][j]
        prog = self.programs[self.data["cases"][name]["program"]]
        est = lambda k, m: self.call_exec.get(k, self.cost.exec_s(k, m))
        out = [(model, est(key, model))]
        for k, m in predicted_chain(prog, key, model, 12):
            out.append((m, est(k, m)))
        return out

    def cases_of(self, program_path: str):
        return [k for k, c in self.data["cases"].items() if c["program"] == program_path and k in self.chains]


def _planners():
    fifo = lambda pi: fifo_order([[s.model for s in c.steps] for c in pi.chains])
    return {"fifo": fifo,
            "greedy": lambda pi: choose(pi, fifo(pi), greedy_order(pi))[0],
            "dp": lambda pi: choose(pi, fifo(pi), dp_order(pi))[0]}


def score_burst(prof: Profiles, cases, offsets, tier: dict, gpu_slots: int = 1):
    """Simulated total waiting of one burst under each planner, and the tier map DP leaves behind."""
    scn = Scenario(prof.load, [prof.chains[c] for c in cases], arrivals=list(offsets), tier=dict(tier), gpu_slots=gpu_slots,
                   lookahead=None if not prof.programs else (lambda i, j: prof.lookahead(cases[i], j)))
    waits, after = {}, None
    for name, planner in _planners().items():
        r = simulate(scn, planner)
        waits[name] = round(r.wait, 2)
        if name == "dp":
            after = r
    return waits, after


def adversarial_bursts(prof: Profiles, seed: int, count: int, weights: dict, burst_size: int, burst_gap: float,
                       jitter: float, candidates: int, host_slots: int = 3, gpu_slots: int = 1):
    by_name = {os.path.basename(p): p for p in PROGRAMS}
    progs = [by_name[n] for n in weights if prof.cases_of(by_name[n])]
    w = [weights[os.path.basename(p)] for p in progs]
    rs = {k: random.Random(f"{seed}:{k}") for k in ("gap", "program", "case")}
    entries, bursts = [], []
    tier = {m: "ssd" for m in prof.load}                       # cold start, as run_schedule.py resets
    recent: list = []                                          # models on host, most recent last
    for b in range((count + burst_size - 1) // burst_size):
        k = min(burst_size, count - b * burst_size)
        cands = []
        for _ in range(candidates):                            # fixed draws per burst: prefix-stable
            ps = [rs["program"].choices(progs, weights=w)[0] for _ in range(k)]
            cs = [rs["case"].choice(prof.cases_of(p)) for p in ps]
            offs = sorted(rs["gap"].uniform(0.0, jitter) for _ in range(k))
            cands.append((cs, offs))
        best = None
        for cs, offs in cands:
            waits, after = score_burst(prof, cs, offs, tier, gpu_slots)
            gap = waits["greedy"] - waits["dp"]
            key = (gap, waits["fifo"] - waits["dp"])
            if best is None or key > best[0]:
                best = (key, cs, offs, waits, after)
        _, cs, offs, waits, after = best
        base = b * burst_gap
        for c, off in zip(cs, offs):
            case = prof.data["cases"][c]
            entries.append({"order": len(entries), "program": case["program"], "env": case["env"],
                            "case_index": case["case_index"], "start_offset_s": round(base + off, 3)})
        bursts.append({"burst": b, "cases": cs, "sim_wait": waits})
        # carry-over: what DP's replay used last stays on host (up to host_slots), the rest is on SSD
        for m in after.actions:
            recent = [x for x in recent if x != m] + [m]
        recent = recent[-host_slots:]
        tier = {m: ("host" if m in recent else "ssd") for m in prof.load}
    return entries, bursts


def generate(seed: int, pattern: str, count: int, weights: dict, rate: float = 0.5,
             burst_size: int = 3, burst_gap: float = 25.0, jitter: float = 1.0,
             profiles: str = "", candidates: int = 200, host_slots: int = 3, gpu_slots: int = 1,
             oracle: bool = False, exclude=()) -> dict:
    meta = {"seed": seed, "pattern": pattern, "count": count, "weights": weights, "generator": "gen_schedule.py v3"}
    if pattern == "adversarial":
        prof = Profiles(profiles, oracle=oracle, exclude=exclude)
        entries, bursts = adversarial_bursts(prof, seed, count, weights, burst_size, burst_gap, jitter, candidates,
                                             host_slots, gpu_slots)
        meta.update(burst_size=burst_size, burst_gap=burst_gap, jitter=jitter, candidates=candidates, oracle=oracle,
                    exclude=list(exclude),
                    profiles=os.path.relpath(profiles, ROOT), profiles_sha1=prof.sha1, bursts=bursts,
                    sim_wait_total={k: round(sum(b["sim_wait"][k] for b in bursts), 1) for k in ("fifo", "greedy", "dp")})
        return {"meta": meta, "entries": entries}
    by_name = {os.path.basename(p): p for p in PROGRAMS}
    progs = [by_name[n] for n in weights]
    w = [weights[n] for n in weights]
    rs = {k: random.Random(f"{seed}:{k}") for k in ("gap", "program", "case")}
    entries, t = [], 0.0
    for i in range(count):
        if pattern == "poisson":
            t += rs["gap"].expovariate(rate) if i else 0.0
        elif pattern == "burst":
            t = (i // burst_size) * burst_gap + rs["gap"].uniform(0.0, jitter)
        else:
            raise ValueError(pattern)
        p = rs["program"].choices(progs, weights=w)[0]
        entries.append({"order": i, "program": p, "env": PROGRAMS[p][0],
                        "case_index": rs["case"].randrange(PROGRAMS[p][1]), "start_offset_s": round(t, 3)})
    meta.update({"rate": rate} if pattern == "poisson" else {"burst_size": burst_size, "burst_gap": burst_gap, "jitter": jitter})
    return {"meta": meta, "entries": entries}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--pattern", choices=["poisson", "burst", "adversarial"], default="poisson")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--rate", type=float, default=0.5, help="poisson: mean arrivals per second")
    ap.add_argument("--burst-size", type=int, default=3, help="burst: arrivals per burst")
    ap.add_argument("--burst-gap", type=float, default=25.0, help="burst: seconds between bursts")
    ap.add_argument("--jitter", type=float, default=1.0, help="burst: arrivals spread uniformly over this many seconds")
    ap.add_argument("--weights", default="", help="tenant mix, e.g. hover.jac=2,triage.jac=1 (default: every program once)")
    ap.add_argument("--profiles", default=os.path.join(ROOT, "benchmark/workload/profiles.json"), help="adversarial: profile.py output")
    ap.add_argument("--candidates", type=int, default=200, help="adversarial: combinations tried per burst")
    ap.add_argument("--host-slots", type=int, default=3, help="adversarial: server host_slots, for the carry-over state")
    ap.add_argument("--gpu-slots", type=int, default=1)
    ap.add_argument("--oracle", action="store_true", help="adversarial: score with the true chains instead of the planner's predicted view")
    ap.add_argument("--exclude", default="", help="adversarial: profiled cases to leave out, e.g. deep_research.jac:0,hover.jac:2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = default_weights()
    if args.weights:
        weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
        unknown = set(weights) - {os.path.basename(p) for p in PROGRAMS}
        if unknown:
            raise SystemExit(f"unknown programs: {sorted(unknown)}")
    sched = generate(args.seed, args.pattern, args.count, weights, args.rate, args.burst_size, args.burst_gap, args.jitter,
                     args.profiles, args.candidates, args.host_slots, args.gpu_slots, args.oracle,
                     [x for x in args.exclude.split(",") if x])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sched, f, indent=1)
    last = sched["entries"][-1]["start_offset_s"] if sched["entries"] else 0.0
    meta = {k: v for k, v in sched["meta"].items() if k != "bursts"}
    print(f"[schedule] {args.out}: {len(sched['entries'])} entries over {last:.1f}s, {meta}")
    for b in sched["meta"].get("bursts", []):
        print(f"   burst {b['burst']}: {b['cases']} sim_wait={b['sim_wait']}")


if __name__ == "__main__":
    main()
