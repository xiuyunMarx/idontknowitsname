"""Prefill speed of uncached prompt tokens per arm, TP=1 (../A40) vs TP=4 (this shelf), on the mixed workload.

Only calls prefilled with no other prefill in flight and at most one running decode are counted,
so TTFT is the prefill time of the call's uncached tokens. Pooled over all lane levels.

    python3 prefill_speed.py
"""
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["vanilla", "relayout", "ours_noprefetch", "ours"]


def prefill_speeds(shelf: str, arm: str) -> list:
    """Uncached tokens per second of every isolated call with at least 1K uncached tokens."""
    out = []
    for path in glob.glob(str(ROOT / shelf / "raw_data" / arm / "mixed" / "mix_L*" / "requests.jsonl")):
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e["event"] != "first_token" or e["inflight_prefill"] or e["inflight_decode"] > 1:
                    continue
                uncached = e["prompt_tokens"] - e["cached_tokens"]
                if uncached >= 1000:
                    out.append(uncached / e["ttft_ms"] * 1000)
    return out


if __name__ == "__main__":
    for arm in ARMS:
        tp1, tp4 = prefill_speeds("A40", arm), prefill_speeds("A40-TP4", arm)
        s1, s4 = statistics.median(tp1), statistics.median(tp4)
        print(f"{arm:16s} TP=1 {s1:5.0f} tok/s  TP=4 {s4:5.0f} tok/s  {s4 / s1:.2f}x  (n={len(tp1)}, {len(tp4)})")
