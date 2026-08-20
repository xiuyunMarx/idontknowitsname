"""Probe speculator for `visit ... by llm()` fan-out sites.

At a selection site the compiler gives us the candidate SET but not which
candidate this request will pick — blindly prefilling every candidate's
downstream byllm calls spends idle-window capacity (and cache space) N ways.
The probe recovers a ranking from the in-flight computation itself, training
free: fork the route prompt the engine has JUST prefilled (its KV is in APC, so
the fork's prefill is ~free), steer the assistant into answering immediately
("truncate the thinking"), decode a handful of tokens with logprobs, and read
the model's own probability distribution over candidate handles. The predicted
winner's byllm call sites get FAVORED in the drainer's ranking — warmed first;
the rest stay enqueued behind them (advisory over-approximation: a wrong probe
just restores today's blind order, never a wrong prompt).

The probe is decode-shaped: a few output tokens on a byte-cached prefix. It
deliberately bypasses the scheduler's spec_slot gate (which is calibrated for
warm-shaped PREFILL interference) AND runs at priority=0: at priority=1 vLLM's
priority queue holds it until the route's own decode finishes — exactly too
late. Its interference is bounded (~zero prefill via APC + a handful of decode
steps riding the route's batch); it launches at the route's FIRST TOKEN, so the
TTFT-critical window is already over. A decode-vs-decode interference
calibration is future work.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from vllm.sampling_params import SamplingParams

from runtime.incremental_feed import parse_candidate_handles, slugify_handle


PROBE_STEER = "The walker should visit:"
THINK_CLOSE = "<think>\n\n</think>\n\n" # Empty for none-thinking model


def _match_owner(handle: str, owners: List[str]) -> Optional[str]:
    """Handle -> node type: handles are _slugify(TypeName) plus an optional
    `_suffix` disambiguator, so match by slug identity/prefix (longest owner
    first, so RoomBig wins over Room for handle 'RoomBig_2')."""
    for o in sorted(owners, key=len, reverse=True):
        s = slugify_handle(o)
        if handle == s or handle.startswith(s + "_"):
            return o
    return None


class SpeculateCandidate:
    """Per-server probe runner: one in-flight probe per visit site."""

    def __init__(self, engine: Any, server: Any, max_probe_tokens: int = 8, top_logprobs: int = 20, probe_priority: int = 0):
        self.engine = engine
        self.server = server
        self.max_probe_tokens = max_probe_tokens
        self.top_logprobs = top_logprobs
        self.probe_priority = probe_priority
        self._inflight: set = set()

    # ------------------------------------------------------------------ scoring
    def _handle_tokens(self, handles: List[str]) -> Dict[str, List[int]]:
        tok = self.server.tokenizer
        return {h: tok.encode(f" {h}", add_special_tokens=False) for h in handles}

    @staticmethod
    def _score(cand_tokens: Dict[str, List[int]], greedy: List[int], steps: List[Optional[Dict[int, Any]]]) -> Dict[str, float]:
        """Cumulative logprob of each handle's token sequence along the greedy path."""
        scores: Dict[str, float] = {}
        for h, toks in cand_tokens.items():
            s = 0.0
            for i, t in enumerate(toks):
                if i >= len(steps) or steps[i] is None:
                    break
                entry = steps[i].get(t) #type: ignore
                if entry is None:
                    s += -30.0  # not in top-k at the deciding position: effectively ruled out
                    break
                s += float(getattr(entry, "logprob", entry))
                if i >= len(greedy) or greedy[i] != t:
                    break  # diverged from the sampled path; later tables are conditioned on greedy, not on h
            scores[h] = s
        return scores

    # -------------------------------------------------------------------- probe
    async def probe(self, key: str, route_prompt: str) -> Optional[str]:
        """Sample the choice distribution for one route request and favor the
        predicted candidate's downstream byllm calls in the spec queue.
        Returns the predicted handle (None if the probe could not run)."""
        if not route_prompt or key in self._inflight:
            return None
        ctx = self.server.feeds.get_visit_ctx(key)
        if not ctx:
            return None
        owners = list({k.split(".")[0] for k in self.server.side_rt.callsites_topo.get(key, [])})
        cand_types = {h: o for h in parse_candidate_handles(ctx.get("candidates", "")) if (o := _match_owner(h, owners)) is not None}
        if len(cand_types) < 2:
            return None  # nothing to rank
        self._inflight.add(key)
        t0 = time.perf_counter()
        try:
            cand_tokens = self._handle_tokens(list(cand_types))
            need = min(self.max_probe_tokens, max(len(t) for t in cand_tokens.values()))
            sp = SamplingParams(max_tokens=need, temperature=0.0, logprobs=self.top_logprobs)
            final, _ = await self.server._generate(route_prompt + THINK_CLOSE + PROBE_STEER, sp, f"probe-{uuid.uuid4().hex[:8]}", priority=self.probe_priority)
            out = final.outputs[0] if final is not None and final.outputs else None
            if out is None or not out.logprobs:
                return None
            scores = self._score(cand_tokens, list(out.token_ids), list(out.logprobs))
            ranked = sorted(scores, key=scores.get, reverse=True) #type: ignore
            predicted = ranked[0]
            dist = _softmax(scores)
            favored = self.server.favor_candidate(key, cand_types[predicted], reason=f"probe:{key}")
            self.server.stats["probes"].append({"key": key, "predicted": predicted, "type": cand_types[predicted], "dist": {h: round(p, 4) for h, p in dist.items()}, "favored": favored, "duration": time.perf_counter() - t0})
            self.server.monitor._log("probe_done", key, predicted=predicted, favored=favored)
            return predicted
        except Exception as e:  # advisory: a failed probe must never surface
            self.server.monitor._log("probe_error", key, error=repr(e))
            return None
        finally:
            self._inflight.discard(key)

    def launch(self, key: str, route_prompt: str) -> None:
        """Fire-and-forget from a sync monitor hook."""
        asyncio.get_running_loop().create_task(self.probe(key, route_prompt))


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values())
    exps = {h: math.exp(s - m) for h, s in scores.items()}
    z = sum(exps.values()) or 1.0
    return {h: v / z for h, v in exps.items()}
