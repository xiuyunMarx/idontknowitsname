CacheScout rerun of 2026-09-29 with call-site agent identity (server/cacheScout_server.py
after the 09-29 change: agent = call site from the guard front, anchor = longest common
prefix of the site's prompts; both arms through the guard front; cache flushed before each run).

mix4x4_h10/cachescout           original prompt order, complete (16:01-17:58)
mix4x4_h10/cachescout_relayout  re-laid prompts, complete (14:26-16:00)
sweeps/                         coding sweep of cachescout, stopped at 18:xx by the user; partial

Result: cache statistics identical to Vanilla on the original order (anchors are 250-600 of
5-7k prompt tokens), throughput 0.93x Vanilla with TTFT +11% uniformly across programs;
on re-laid prompts equal to Relayout-only (104.7 vs 108.8 ktok/min, miss 84.3 vs 87.4).
Archived, not adopted: the user restored the 09-27/28 results (old prefix-fingerprint
CacheScout, see README_old_fingerprint_data.md) to mix4x4_h10/ and sweeps/ pending a decision.
