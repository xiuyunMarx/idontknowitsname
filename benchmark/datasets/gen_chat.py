"""Generate the group_chat dataset: the routed query pool plus one ~8 KB context
document per sub-agent (kb_manual / web_playbook / analyst_reference), the text
each sub-agent carries as the first argument of its ReAct call. Queries are
labeled with the sub-agent that should answer them (kb = KnowledgeBase, web =
WebResearcher, calc = DataAnalyst) so visit-by routing accuracy is checkable.
The kb manual repeats the search_base facts verbatim so answers stay checkable.
Everything is deterministic and apostrophe-free (the texts land verbatim inside
prompts the server field parser must survive).

    python benchmark/datasets/gen_chat.py [out_dir]
"""
import os
import random
import sys

QUERIES = [
    ("kb", "What is the API rate limit on the pro plan?"),
    ("kb", "How much does the pro plan cost per seat?"),
    ("kb", "How many projects does the free tier allow?"),
    ("kb", "Login keeps looping after SSO, how do we fix it?"),
    ("kb", "How long are backups retained and when do they run?"),
    ("kb", "Can we burst above the API quota, and by how much?"),
    ("kb", "Is there a way to restore data from three weeks ago?"),
    ("kb", "Why would auth cookies cause a redirect loop on our domain?"),
    ("web", "What did the latest MLPerf inference round report for 8B class models?"),
    ("web", "What is the current consensus on the best open embedding model this year?"),
    ("web", "Summarize the newest PCIe 6.0 adoption numbers reported this year."),
    ("web", "What figure do recent benchmarks give for NVMe random read latency?"),
    ("web", "Find the measured energy overhead of confidential computing reported recently."),
    ("web", "What do current reviews measure for USB4 enclosure throughput?"),
    ("calc", "If a job processes 1450 records per minute, how many records is that in 6.5 hours?"),
    ("calc", "A cluster of 12 nodes each stores 3.2 TB at 68 percent full; how many TB are used in total?"),
    ("calc", "Convert 250 megabits per second into gigabytes per hour."),
    ("calc", "We pay 29 dollars per seat for 47 seats monthly; what is the annual bill?"),
    ("calc", "What is 17.5 percent of 3840, minus 96?"),
    ("calc", "How many 512 token chunks fit into a 31830 token budget, and how many tokens remain?"),
]

# The four search_base entries, verbatim, so the manual and the tool agree.
KB_ENTRIES = [
    "KB-12: The pro plan is $29/seat/month; the free tier allows 3 projects.",
    "KB-31: API rate limit is 600 requests/min per key; bursts up to 1200 for 10s.",
    "KB-07: Login loops are usually stale SSO cookies; clearing the *.example.com cookies fixes it.",
    "KB-44: Backups run nightly at 02:00 UTC and are kept for 35 days.",
]

KB_AREAS = ["Plans and billing", "API and quotas", "Authentication", "Backups and restore",
            "Workspaces and projects", "Integrations", "Audit and compliance", "Support process"]
KB_SUBJECTS = ["seat", "workspace", "API key", "project", "webhook", "export job",
               "SSO connection", "audit log", "invoice", "service account"]
KB_RULES = ["is created from the admin console", "can be renamed at any time",
            "is billed on the first of the month", "expires after 90 days of inactivity",
            "is limited to 50 per workspace on the free tier", "requires the owner role",
            "is retained for 12 months after deletion", "supports at most 25 members",
            "is rotated automatically every 180 days", "must be verified by email"]


def kb_manual(rng):
    out = ["Product Knowledge Base Manual", "Edition 7, internal support reference.", ""]
    out.append("== Canonical entries ==")
    out.extend(KB_ENTRIES)
    out.append("")
    n = 100
    for area in KB_AREAS:
        out.append(f"== {area} ==")
        for _ in range(rng.randint(7, 9)):
            n += 1
            out.append(f"KB-{n}: Each {rng.choice(KB_SUBJECTS)} {rng.choice(KB_RULES)}; "
                       f"the {rng.choice(KB_SUBJECTS)} {rng.choice(KB_RULES)}.")
        out.append("")
    out.append("== Answering rules ==")
    out.append("Quote the KB entry id that supports the answer. Prefer the canonical entries "
               "when two entries overlap. Never invent a number that is not in this manual.")
    return "\n".join(out)


WEB_TOPICS = ["accelerator benchmarks", "storage latency", "interconnect adoption",
              "embedding model rankings", "confidential computing overhead",
              "peripheral throughput reviews", "open model releases", "power efficiency"]
WEB_SOURCES = ["vendor whitepapers", "peer reviewed papers", "independent lab reviews",
               "standards body press releases", "conference talks", "community forums",
               "benchmark consortium reports", "analyst notes"]
WEB_TIERS = ["tier 1 (primary measurement)", "tier 2 (reputable secondary)",
             "tier 3 (unverified, quote with caution)"]


def web_playbook(rng):
    out = ["Web Research Playbook", "How the research sub-agent searches, vets and cites.", ""]
    out.append("== Procedure ==")
    out.append("1. Search once with the query as given. 2. Read the snippets; if none states the "
               "measured figure, fetch the page whose snippet promises numbers. 3. Answer in two "
               "sentences: the figure with its unit and date, then the source tier.")
    out.append("")
    for topic in WEB_TOPICS:
        out.append(f"== Source guide: {topic} ==")
        for _ in range(rng.randint(4, 6)):
            out.append(f"Prefer {rng.choice(WEB_SOURCES)} over {rng.choice(WEB_SOURCES)} for "
                       f"{topic}; treat them as {rng.choice(WEB_TIERS)} and require a "
                       f"methodology section dated within {rng.randint(6, 24)} months.")
        out.append("")
    out.append("== Citation format ==")
    out.append("Cite as (site, year). Report ranges when sources disagree by more than 10 percent. "
               "Never present a snippet number as measured unless the fetched page confirms it.")
    return "\n".join(out)


UNITS = [("bit", 1), ("byte", 8), ("kilobyte", 8 * 1e3), ("megabyte", 8 * 1e6),
         ("gigabyte", 8 * 1e9), ("terabyte", 8 * 1e12), ("kilobit", 1e3), ("megabit", 1e6),
         ("gigabit", 1e9)]
TIMES = [("second", 1), ("minute", 60), ("hour", 3600), ("day", 86400), ("week", 604800)]


def analyst_reference(rng):
    out = ["Quantitative Analyst Reference", "Unit tables, formulas and rounding rules.", ""]
    out.append("== Data units (decimal, in bits) ==")
    for name, bits in UNITS:
        out.append(f"1 {name} = {bits:.0f} bits")
    out.append("")
    out.append("== Time units (in seconds) ==")
    for name, s in TIMES:
        out.append(f"1 {name} = {s} seconds")
    out.append("")
    out.append("== Conversion table: bits per second to bytes per hour ==")
    for name, bits in UNITS:
        for t, s in TIMES:
            out.append(f"1 {name}/second over 1 {t} = {bits * s / 8:.0f} bytes")
    out.append("")
    out.append("== Formulas ==")
    out.append("throughput_total = rate_per_unit * units_elapsed")
    out.append("used_capacity = node_count * capacity_per_node * fill_fraction")
    out.append("annual_cost = unit_price * units * 12")
    out.append("percent_of = value * percent / 100")
    out.append("chunks = floor(budget / chunk_size); remainder = budget - chunks * chunk_size")
    out.append("")
    out.append("== Worked examples ==")
    for _ in range(30):
        a, b, c = rng.randint(3, 60), rng.randint(2, 48), rng.randint(2, 9)
        out.append(f"{a} units at {b} per unit for {c} periods = {a * b * c}; "
                   f"{a} percent of {b * 100} = {a * b}; floor({a * b * c} / {c * 7}) = "
                   f"{(a * b * c) // (c * 7)} remainder {(a * b * c) % (c * 7)}.")
    out.append("")
    out.append("== Rounding rules ==")
    out.append("Keep full precision through calculate; round only the final answer to 2 decimals. "
               "State units in every answer. Use calculate for every arithmetic step.")
    return "\n".join(out)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "chat")
    os.makedirs(out_dir, exist_ok=True)
    qp = os.path.join(out_dir, "queries.tsv")
    with open(qp, "w") as f:
        f.write("\n".join(f"{agent}\t{q}" for agent, q in QUERIES) + "\n")
    print(f"{qp}: {len(QUERIES)} queries across 3 sub-agents")
    for name, gen in [("kb_manual", kb_manual), ("web_playbook", web_playbook),
                      ("analyst_reference", analyst_reference)]:
        text = gen(random.Random(name))
        assert "'" not in text, name
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w") as f:
            f.write(text)
        print(f"{path}: {len(text)} bytes")
