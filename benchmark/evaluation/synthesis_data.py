"""Build a multi-tenant cold-start blend from the three evaluation packs.

A blend is a schedule, not a copy of the data: the tenant programs load their own
pack and select a case by index, so an entry only has to say which tenant runs, which
case index it takes, and when it starts. That keeps the blend small and keeps one
source of truth for the case content in `datasets/`.

    python benchmark/evaluation/synthesis_data.py --blend-ratio 0.5,0.3,0.2 \
        --size 12 --trials 5

`--blend-ratio` is read in `--tenants` order (default hover, intercode_sql,
deep_search) and gives each tenant its share of the `--size` workflows in one trial.
Cases are drawn from a seeded per-tenant permutation and consumed across trials, so a
case repeats only after its pack is exhausted. Everything is a pure function of
`--seed` and the packs.

The output feeds a runner shaped like `microbench/experiments/run_experiment.py`:

    server.programs   ->  start_server.py --program <name>:<path>:<port>
    entries[].env     ->  the environment for one `jac run <jac_file>`
    entries[].start_offset_s -> when to launch it inside the trial
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "datasets")
PROGRAMS = os.path.join(HERE, "programs")
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SEED = 20260824

HOVER_PACK = "hover_claims_30.json"
SQL_PACK = "intercode_sql_30.json"
DEEP_PACK = "deep_search_30.json"


def coma_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


# ------------------------------------------------------------------ pack loaders
# Each loader returns the pack's cases in file order. `index` is what the tenant's
# selector environment variable takes; `strata` is the metadata a blend reports on
# and a future mixture policy can select by.

def load_hover_datasets() -> List[dict]:
    claims = json.load(open(os.path.join(DATASETS, HOVER_PACK)))
    return [{"index": i, "id": c["uid"],
             "strata": {"hops": c["num_hops"], "label": c["label"]}}
            for i, c in enumerate(claims)]


def load_SQL_dataset() -> List[dict]:
    tasks = json.load(open(os.path.join(DATASETS, SQL_PACK)))["tasks"]
    return [{"index": i, "id": t["id"],
             "strata": {"hardness": t["hardness"], "db": t["db"]}}
            for i, t in enumerate(tasks)]


def load_deep_research_datasets() -> List[dict]:
    tasks = json.load(open(os.path.join(DATASETS, DEEP_PACK)))["tasks"]
    return [{"index": i, "id": t["id"],
             "strata": {"kind": t["kind"], "cluster": t["cluster"]}}
            for i, t in enumerate(tasks)]


TENANTS: Dict[str, dict] = {
    "hover": {"jac": "HoVer.jac", "selector": "HOVER_CLAIM_INDEX",
              "loader": load_hover_datasets, "pack": HOVER_PACK},
    "intercode_sql": {"jac": "intercode_sql.jac", "selector": "SQL_TASK_INDEX",
                      "loader": load_SQL_dataset, "pack": SQL_PACK},
    "deep_search": {"jac": "deep_search.jac", "selector": "DEEP_TASK_INDEX",
                    "loader": load_deep_research_datasets, "pack": DEEP_PACK},
}


def tenant_spec(tenant: str) -> dict:
    """Name and port straight out of the .jac, so a blend cannot drift from the
    program it schedules."""
    path = os.path.join(PROGRAMS, TENANTS[tenant]["jac"])
    source = open(path).read()
    port = re.search(r"comm_port\s*=\s*(\d+)", source)
    name = re.search(r'program_name\s*=\s*"([^"]+)"', source)
    if not (port and name):
        raise ValueError(f"{path}: no comm_port / program_name to read")
    return {"tenant": tenant,
            "program_name": name.group(1),
            "port": int(port.group(1)),
            "jac_file": os.path.relpath(path, REPO),
            "selector": TENANTS[tenant]["selector"]}


def pack_fingerprint(tenant: str, cases: int) -> dict:
    name = TENANTS[tenant]["pack"]
    digest = hashlib.sha1(open(os.path.join(DATASETS, name), "rb").read()).hexdigest()
    return {"file": f"datasets/{name}", "cases": cases, "sha1": digest[:12]}


# --------------------------------------------------------------------- sampling

def allocate(size: int, ratios: List[float]) -> List[int]:
    """Split `size` workflows over the tenants by largest remainder, so the counts
    sum to exactly `size` instead of drifting with rounding."""
    exact = [size * r for r in ratios]
    counts = [int(x) for x in exact]
    for position in sorted(range(len(ratios)), key=lambda i: exact[i] - counts[i],
                           reverse=True)[: size - sum(counts)]:
        counts[position] += 1
    return counts


class CaseStream:
    """A tenant's case order: one seeded permutation, consumed across trials and
    reshuffled when the pack runs out, so nothing repeats until everything ran."""

    def __init__(self, cases: List[dict], seed: int, tenant: str):
        self.cases = cases
        self.rng = random.Random(f"{seed}:{tenant}")
        self.order: List[int] = []

    def take(self, count: int) -> List[dict]:
        out = []
        for _ in range(count):
            if not self.order:
                self.order = list(range(len(self.cases)))
                self.rng.shuffle(self.order)
            out.append(self.cases[self.order.pop()])
        return out


def arrival_offsets(count: int, arrival: str, window: float, rate: float, jitter: float,
                    rng: random.Random) -> List[float]:
    """Start offsets (seconds, ascending) for `count` workflows of one multi-mode trial.

    uniform  evenly spaced over `window` (the original blend behaviour)
    burst    all inside the first `jitter` seconds
    poisson  a Poisson process of `rate` arrivals per second"""
    if count == 0:
        return []
    if arrival == "uniform":
        span = max(1, count - 1)
        return [round(window * i / span, 3) for i in range(count)]
    if arrival == "burst":
        return sorted(round(rng.uniform(0.0, jitter), 3) for _ in range(count))
    if arrival == "poisson":
        out, t = [], 0.0
        for _ in range(count):
            t += rng.expovariate(rate)
            out.append(round(t, 3))
        return out
    raise ValueError(arrival)


def build_trial(trial: int, specs: List[dict], streams: Dict[str, CaseStream],
                counts: List[int], mode: str, window: float, rng: random.Random,
                arrival: str = "uniform", rate: float = 1.0, jitter: float = 1.0) -> dict:
    entries = []
    for spec, count in zip(specs, counts):
        for case in streams[spec["tenant"]].take(count):
            entries.append({
                "tenant": spec["tenant"],
                "program_name": spec["program_name"],
                "jac_file": spec["jac_file"],
                "port": spec["port"],
                "case_index": case["index"],
                "case_id": case["id"],
                "strata": case["strata"],
                "env": {spec["selector"]: str(case["index"])},
            })
    # Interleave the tenants: grouping them would hand the engine one tenant's whole
    # workload at a time, which is not the concurrency a blend is meant to create.
    rng.shuffle(entries)
    offsets = arrival_offsets(len(entries), arrival, window, rate, jitter, rng) if mode == "multi" else []
    for position, entry in enumerate(entries):
        entry["order"] = position
        if mode == "multi":
            entry["start_offset_s"] = offsets[position]
        else:
            entry["start_offset_s"] = None  # sequential: start when the previous ends
    return {"trial": trial, "entries": entries}


def summarize(blend: dict) -> dict:
    per_tenant = collections.Counter()
    strata = collections.defaultdict(collections.Counter)
    unique = collections.defaultdict(set)
    for trial in blend["trials"]:
        for entry in trial["entries"]:
            per_tenant[entry["tenant"]] += 1
            unique[entry["tenant"]].add(entry["case_index"])
            for key, value in entry["strata"].items():
                strata[f"{entry['tenant']}.{key}"][str(value)] += 1
    return {
        "workflows": sum(per_tenant.values()),
        "per_tenant": dict(per_tenant),
        "distinct_cases": {tenant: len(seen) for tenant, seen in unique.items()},
        "strata": {key: dict(counter) for key, counter in sorted(strata.items())},
    }


def build_blend(args: argparse.Namespace) -> dict:
    specs = [tenant_spec(tenant) for tenant in args.tenants]
    counts = allocate(args.size, args.blend_ratio)
    streams = {spec["tenant"]: CaseStream(TENANTS[spec["tenant"]]["loader"](),
                                          args.seed, spec["tenant"])
               for spec in specs}
    for spec, count in zip(specs, counts):
        pack = len(streams[spec["tenant"]].cases)
        if count > pack:
            print(f"[warn] {spec['tenant']}: {count} workflows per trial over a "
                  f"{pack}-case pack, so cases repeat inside a trial")

    rng = random.Random(f"{args.seed}:order")
    trials = [build_trial(t, specs, streams, counts, args.mode, args.window, rng,
                          args.arrival, args.rate, args.jitter)
              for t in range(args.trials)]

    active = [spec for spec, count in zip(specs, counts) if count > 0]
    blend = {
        "meta": {
            "seed": args.seed,
            "size": args.size,
            "trials": args.trials,
            "mode": args.mode,
            "window_s": args.window if args.mode == "multi" else None,
            "arrival": args.arrival if args.mode == "multi" else None,
            "rate_per_s": args.rate if args.mode == "multi" and args.arrival == "poisson" else None,
            "jitter_s": args.jitter if args.mode == "multi" and args.arrival == "burst" else None,
            "tenants": args.tenants,
            "blend_ratio": args.blend_ratio,
            "per_trial_counts": dict(zip(args.tenants, counts)),
            # which pack version this blend indexes into: a case index means
            # nothing without the file it points at
            "packs": {spec["tenant"]: pack_fingerprint(spec["tenant"],
                                                       len(streams[spec["tenant"]].cases))
                      for spec in specs},
        },
        # exactly the tenants this blend uses: a cold-start trial should not register
        # a program it never runs
        "server": {"programs": [f"{s['program_name']}:{s['jac_file']}:{s['port']}"
                                for s in active]},
        "trials": trials,
    }
    blend["summary"] = summarize(blend)
    return blend


def write_jsonl(blend: dict, path: str) -> None:
    with open(path, "w") as handle:
        for trial in blend["trials"]:
            for entry in trial["entries"]:
                row = {"trial": trial["trial"], **entry}
                row.update({f"strata.{k}": v for k, v in row.pop("strata").items()})
                handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset for multi tenant")
    parser.add_argument("--blend-ratio", type=coma_list, default=[1 / 3, 1 / 3, 1 / 3],
                        help="share of each tenant's workflows, in --tenants order")
    parser.add_argument("--tenants", type=lambda s: s.split(","), default=list(TENANTS),
                        help=f"comma-separated subset of {', '.join(TENANTS)}")
    parser.add_argument("--size", type=int, default=12,
                        help="workflows in one trial (default 12)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--mode", choices=["multi", "single"], default="multi",
                        help="multi: all workflows start inside --window; "
                             "single: one after another")
    parser.add_argument("--window", type=float, default=3.0,
                        help="seconds over which a multi-mode trial starts its workflows")
    parser.add_argument("--arrival", choices=["uniform", "burst", "poisson"], default="uniform",
                        help="multi mode: uniform over --window, burst within --jitter, "
                             "or a Poisson process of --rate per second")
    parser.add_argument("--rate", type=float, default=1.0, help="poisson arrivals per second")
    parser.add_argument("--jitter", type=float, default=1.0, help="burst: max start offset in seconds")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default=os.path.join(HERE, "blends", "blend.json"))
    parser.add_argument("--jsonl", default=None, help="also write one flat row per workflow")
    args = parser.parse_args()

    unknown = [t for t in args.tenants if t not in TENANTS]
    if unknown:
        parser.error(f"unknown tenants: {unknown}; known: {list(TENANTS)}")
    if len(args.blend_ratio) != len(args.tenants):
        parser.error(f"--blend-ratio has {len(args.blend_ratio)} values for "
                     f"{len(args.tenants)} tenants ({', '.join(args.tenants)})")
    if any(r < 0 for r in args.blend_ratio):
        parser.error("blend ratios must be non-negative")
    # binary floating point: 0.1 + 0.2 + 0.7 is 0.9999999999999999, so compare with a
    # tolerance rather than against 1.0 exactly
    if abs(sum(args.blend_ratio) - 1.0) > 1e-6:
        parser.error(f"blend ratios must sum to 1.0, got {sum(args.blend_ratio)}")
    if args.size < 1 or args.trials < 1:
        parser.error("--size and --trials must be at least 1")

    blend = build_blend(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(blend, open(args.out, "w"), indent=1)
    if args.jsonl:
        write_jsonl(blend, args.jsonl)

    summary = blend["summary"]
    print(f"[blend] {args.out}")
    print(f"[blend] {args.trials} trial(s) x {args.size} workflows, mode={args.mode}"
          + ({"uniform": f", uniform over {args.window}s", "burst": f", burst within {args.jitter}s",
              "poisson": f", poisson {args.rate}/s"}[args.arrival] if args.mode == "multi" else ""))
    for tenant, count in blend["meta"]["per_trial_counts"].items():
        distinct = summary["distinct_cases"].get(tenant, 0)
        print(f"  {tenant:14s} {count:3d}/trial  {summary['per_tenant'].get(tenant, 0):4d} total"
              f"  {distinct:3d} distinct cases")
    for key, counter in summary["strata"].items():
        print(f"  {key:26s} {dict(sorted(counter.items()))}")
    print("[blend] server: " + " ".join(f"--program {p}" for p in blend["server"]["programs"]))


if __name__ == "__main__":
    main()
