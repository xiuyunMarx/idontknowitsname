1. Prefetch only when available space exists

## What each arm is

All arms run the same server code (`server/server.py`) on the same SGLang engine. They differ only in which parts are switched on.

**vanilla** (`./start_server.bash vanilla`, flags `--lru --no-relayout`)
Plain SGLang. The prompt is sent to the engine exactly as the agent wrote it. The cache throws out the least recently used thing when it needs space. Nothing is predicted, nothing is protected, nothing is pulled back from host memory ahead of time. This is the "do nothing" baseline.

**relayout** (`./start_server.bash relayout`, flag `--lru`)
Same as vanilla, plus the server reorders the pieces of each prompt so the parts that stay the same across calls come first and the parts that change come last. That is the whole trick: more of the prompt matches what is already in the cache. Eviction is still plain least-recently-used. This arm measures what the compiler's knowledge of the prompt structure is worth on its own.

**kvonly** (`./start_server.bash kvonly`, flag `--no-relayout`)
The opposite of relayout: prompts stay in the agent's original order, but the planner is on (bands, protection, prefetch, see below). Used only to show that the planner without the reordering does very little.

**hold** (`./start_server.bash ours --no-promote`)
relayout plus the part of the planner that only decides what to throw away first. Once a call is served, the server marks the bytes nobody will read again (the latest tool result, a finished session's private text) as "throw these out first", and marks the prefix the next predicted call will reuse as "keep this". It never copies anything from host memory back to the GPU. This arm measures what knowing which bytes are dead is worth.

**ours** (`./start_server.bash ours`)
hold plus prefetch: when the server predicts the next call and the prefix it needs is sitting in host memory, it copies it back to the GPU before the call arrives. Since 2026-09-17 morning it only does that while running requests hold less than 80% of the GPU cache (`PROMOTE_MAX_USAGE = 0.8` in `server/kv_planner.py`); when the cache is fuller than that a copy-back would only push out something a running request is about to need, so it stays quiet and behaves like hold.

**ours_gate** (the validation run of the rule above)
Same as ours, run once at fact_check c=16 and c=12 right after the 80% rule was added, to check that it fixes the case where ours was slower than hold.

**relayout_r6400 / ours_r6400** (env `SGLANG_PREFETCH_RESERVE_TOKENS=6400`)
Same as relayout / ours, but the engine admits fewer requests so that 6400 tokens of GPU cache (20% of the pool) stay free as a staging area. In ours_r6400 a prefetch may also push out ordinary cached data to use that area. These two arms answer "is it better to fill the cache completely or to keep a staging area for prefetch". Answer from 2026-09-17: no, the extra queueing costs more than the staging area gives back; keeping the area free for plain LRU is useless.

**continuum** (`./start_server.bash continuum`)
Our re-implementation of the Continuum paper: it guesses how long a tool call will take and pins the whole request's cache on the GPU for about that long, then lets it go. Pins whole requests, not pieces.

**cachescout** (`./start_server.bash cachescout`)
Our re-implementation of CacheScout: another published policy that keeps or drops cached requests based on observed reuse. Baseline only.

## Knobs that change the environment, not the arm

- `HOST=8|16`: gigabytes of host (CPU) memory used as the second cache tier. 16 GB holds about 108k tokens.
- `CLIP=128`: the engine normally reserves room for each running request's future output when deciding how many requests to admit. Setting the estimate to 128 tokens makes it admit as many requests as fit right now, so the GPU cache fills up and every decode step has to throw something out. This is the "let the engine fill up" environment.
- `max_tokens` in the agents: with CLIP set it is only the stop condition for generation, not an admission limit.
- `BF_EXCERPT_CHARS=0`: the BFCL agent keeps full tool results in its history, so prompts grow to 8–10k tokens by step 6.
- `BF_TOOL_DELAY_S=2 BF_TOOL_DELAY_STD_S=2`: each BFCL tool call waits a random time drawn from a heavy-tailed distribution with mean 2 s (the same shape Continuum measured), the same value for the same call in every arm.
