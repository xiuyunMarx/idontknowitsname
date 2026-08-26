import re, sys, os, statistics as st
OUT = sys.argv[1]
for cond in sorted(os.listdir(OUT)):
    sl = f"{OUT}/{cond}/server.log"; cl = f"{OUT}/{cond}/client.log"
    if not os.path.exists(sl): continue
    s = open(sl).read(); c = open(cl).read() if os.path.exists(cl) else ""
    serve = re.findall(r"\[serve\] (\S+) duration_ms=([\d.]+) cached_tokens=(\d+) prompt_tokens=(\d+)", s)
    def grp(pat):
        rows = [(float(d), int(ct), int(pt)) for rid, d, ct, pt in serve if re.search(pat, rid)]
        if not rows: return "n=0"
        return (f"n={len(rows)} ttft_ms={st.mean(r[0] for r in rows):.1f} "
                f"cached/prompt={sum(r[1] for r in rows)}/{sum(r[2] for r in rows)} "
                f"({100*sum(r[1] for r in rows)/max(1,sum(r[2] for r in rows)):.0f}%) "
                f"full_hit={sum(1 for r in rows if r[2]-r[1] <= 16)}/{len(rows)}")
    routes = re.findall(r"\[route\] .*?hit=(\d)", s)
    elapsed = [float(x) for x in re.findall(r"ELAPSED: ([\d.]+)s", c)]
    print(f"== {cond}")
    print("  router   :", grp(r":serve:|visit|route"))
    print("  resolve_*:", grp(r"resolve_"))
    print("  summarize:", grp(r"summarize"))
    print(f"  probe hit: {sum(map(int,routes))}/{len(routes)}" if routes else "  probe hit: n/a")
    print(f"  ELAPSED/request: mean={st.mean(elapsed):.2f}s n={len(elapsed)}" if elapsed else "  no client elapsed")
    print("  unrouted:", c.count("(unrouted)"))
