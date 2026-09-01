"""Generate the large-document dataset for report_gen.jac: deterministic synthetic
operations-review documents (seeded), sized to pressure the KV pool — a 10KB doc
is ~2.5k prompt tokens, and every call of a report_gen session carries it.

    python benchmark/datasets/gen_docs.py [n_docs] [out_dir]
"""
import os
import random
import sys

SERVICES = ["ingest gateway", "billing pipeline", "search tier", "session store",
            "export worker", "auth frontend", "metrics collector", "webhook fanout"]
VERBS = ["degraded", "recovered", "saturated", "flapped", "stabilized", "regressed"]
CAUSES = ["a stale routing table", "connection pool exhaustion", "a misconfigured retry budget",
          "clock drift on two replicas", "an oversized batch job", "a leaking file descriptor",
          "cache stampede after a deploy", "an expired internal certificate"]
ACTIONS = ["raised the circuit-breaker threshold", "rolled back the canary",
           "resharded the hot partition", "doubled the worker pool",
           "pinned the client library", "added a back-pressure valve",
           "moved the cron window", "tightened the health check"]

SECTION_THEMES = ["Availability", "Latency", "Capacity", "Incidents", "Cost",
                  "Security posture", "Dependency risk", "Rollout quality"]


def paragraph(rng, week, svc):
    lines = []
    for _ in range(rng.randint(3, 5)):
        lines.append(
            f"During week {week}, the {svc} {rng.choice(VERBS)} after {rng.choice(CAUSES)}; "
            f"the on-call {rng.choice(ACTIONS)}, bringing p99 from {rng.randint(180, 950)}ms "
            f"back to {rng.randint(40, 170)}ms while error budget burn stayed at "
            f"{rng.randint(2, 38)} percent and throughput held near {rng.randint(3, 90)}k rps.")
    return " ".join(lines)


def make_doc(rng, idx, target_kb):
    out = [f"Quarterly Operations Review — document {idx:02d}",
           "Scope: consolidated notes from the weekly service reviews.", ""]
    week = 1
    while sum(len(s) for s in out) < target_kb * 1024:
        theme = rng.choice(SECTION_THEMES)
        svc = rng.choice(SERVICES)
        out.append(f"== {theme}: {svc} ==")
        for _ in range(rng.randint(2, 3)):
            out.append(paragraph(rng, week, svc))
            week += 1
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "docs")
    os.makedirs(out_dir, exist_ok=True)
    sizes = [6, 10, 14]  # KB mix: light / medium / heavy prefill
    for i in range(1, n + 1):
        rng = random.Random(1000 + i)
        doc = make_doc(rng, i, sizes[(i - 1) % len(sizes)])
        path = os.path.join(out_dir, f"doc_{i:02d}.txt")
        with open(path, "w") as f:
            f.write(doc)
        print(f"{path}: {len(doc)} bytes")
