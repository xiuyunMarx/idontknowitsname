"""Deadline-driven KV planner: predicted calls become promotion jobs (host -> device)
and eviction protection."""
import asyncio
import math
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Job:
    key: str                    # dedup identity: f"{sid}|{kind}|{site}"
    sid: str                    # session id
    epoch: int                  # session epoch at submission; a bump voids the job
    site: str                   # target callsite key
    kind: str                   # "promote" (host tokens to load) | "hold" (protect only: nothing to load)
    toks: List[int]             # full target token prefix
    p: float                    # probability the call happens
    value: float                # p × exposed-prefill tokens saved if this job lands
    deadline: float             # monotonic: expected arrival, safety-tightened
    t90: float                  # monotonic: pessimistic arrival, for staleness GC
    host: int                   # promotable host-tier tokens at submission
    done_upto: int = 0          # token index known to be on the device
    attempts: int = 0           # failed promotion RPCs since the last progress
    state: str = "queued"       # queued | running | done | void
    cooldown_cycles: int = 0    # planner cycles to skip after a no-room promotion refusal

class KVPlanner:
    TICK_S = 0.015               # planner heartbeat between wake events
    SAFETY_K = 0.5              # deadline tightening per unit of arrival spread
    MAX_ATTEMPTS = 3            # failed promotion RPCs before a job is dropped
    STALE_SLACK_S = 5.0         # past t90 by this much: the call never came
    MIN_HORIZON_S = 0.5         # a call due now still counts as half a second away in the density
    PROTECT_FRAC = 0.6          # share of the device pool the planned band may cover
    SCORE_SCALE = 1000          # score -> integer priority step (see model.promote.kv_priority)
    RELEASE_SLACK_S = 0.5       # release ahead of the latest start
    NO_ROOM_COOLDOWN_CYCLES = 5 # cycles every promotion job sits out after one is refused for space
    PROMOTE_MAX_USAGE = 0.8     # no load-backs while the device pool is fuller than this: they only churn

    def __init__(self, engine) -> None:
        self.engine = engine
        self._jobs: Dict[str, Dict[str, Job]] = {}   # sid -> job key -> job
        self._wake = asyncio.Event()
        self._dirty = False   # the priority map no longer mirrors the plan
        self._demote: List[Tuple[List[int], int]] = []  # served (ids, fixed_len) awaiting demotion
        self._served: Dict[str, List[Tuple[List[int], int]]] = {}  # sid -> (prompt, static retirement boundary)
        self._retire: List[Tuple[List[int], int]] = []  # ended sessions' prompts awaiting retirement

    # ------------------------------------------------------------------ intake
    def submit(self, sid: str, epoch: int, jobs: List[Job], extend: bool = False) -> None:
        """Adopt a session's fresh job set. Replace semantics: a previously queued
        job not re-submitted is void — its branch collapsed. A re-submitted key with
        longer toks keeps its progress and gets refreshed timing; `extend` merges
        instead of replacing (routing follow-ups add jobs)."""
        held = self._jobs.setdefault(sid, {})
        for j in jobs:
            prev = held.get(j.key)
            if prev is not None and prev.state in ("queued", "running") and prev.epoch == j.epoch:
                if len(j.toks) > len(prev.toks):
                    prev.toks = j.toks
                    prev.attempts = 0
                prev.p, prev.value = j.p, j.value
                prev.deadline, prev.t90 = j.deadline, j.t90
                prev.host = j.host
                continue
            if prev is not None and prev.state == "running":
                prev.state = "void"      # stale epoch still executing
            held[j.key] = j              # replaces a done job too: its prefix was evicted
        if not extend:
            keep = {j.key for j in jobs}
            for k, prev in held.items():
                if k not in keep and prev.state in ("queued", "running"):
                    prev.state = "void"
        self._dirty = True

    def note_served(self, sid: str, ids: List[int], fixed_len: int, *, static_len: int) -> None:
        """A real call landed: everything past "fixed_len" (the head the flow rules
        can rebuild) is transient. At session end only "static_len" survives
        Other live plans can still protect shared bytes when retirement lands."""
        fixed_len = max(0, min(fixed_len, len(ids)))
        static_len = max(0, min(static_len, fixed_len))
        self._demote.append((ids, fixed_len))
        self._served.setdefault(sid, []).append((ids, static_len))
        self._dirty = True

    def invalidate(self, sid: str, ids: List[int], keep_len: int) -> None:
        """A served prompt's values past `keep_len` are marked as evictable."""
        keep_len = max(0, min(keep_len, len(ids)))
        self._demote.append((ids, keep_len))
        served = self._served.get(sid, [])
        for i, (old_ids, old_keep) in enumerate(served):
            if old_ids == ids and keep_len < old_keep:
                served[i] = (old_ids, keep_len)
                # Session retirement must start at the most aggressive invalidation boundary
        self._dirty = True

    def note_arrival(self, sid: str, site: str, t_arrive: float) -> None:
        """A call arrived, update ground truth for the next prediction"""
        js = [j for j in self._jobs.get(sid, {}).values() if j.site == site]
        if not js:
            print(f"[timing] {sid} unplanned site={site[:48]}", flush=True)
            return
        # a loop revisits the same site: only the newest epoch's jobs describe
        # THIS arrival — an older iteration's done job would report stale numbers
        newest = max(j.epoch for j in js)
        j = max((j for j in js if j.epoch == newest), key=lambda x: x.done_upto)
        print(f"[timing] {sid} err_s={t_arrive - j.deadline:+.2f} "
              f"done={j.done_upto}/{len(j.toks)} kind={j.kind} state={j.state} "
              f"site={site[:48]}", flush=True)

    def void_session(self, sid: str, keep_epoch: int) -> None:
        """The session advanced: everything planned before `keep_epoch` is stale."""
        for j in self._jobs.get(sid, {}).values():
            if j.epoch < keep_epoch and j.state in ("queued", "running"):
                j.state = "void"
        self._dirty = True

    def retire_session(self, sid: str) -> None:
        """The program has reached its end for this session: its private cache is
        retired now. The session itself stays open for stats."""
        for j in self._jobs.get(sid, {}).values():
            if j.state in ("queued", "running"):
                j.state = "void"
        self._retire.extend(self._served.pop(sid, []))
        self._dirty = True

    def drop_session(self, sid: str) -> None:
        """The session ended (quiet past its timeout, or closed): void its jobs and
        retire whatever is not retired yet."""
        self.retire_session(sid)
        self._jobs.pop(sid, None)

    def wake(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------------ the loop
    async def run(self) -> None:
        """Single-flight executor: one promotion RPC in flight at a time."""
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.TICK_S)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            now = time.monotonic()
            self._gc(now)
            await self.push_priorities(now)
            job = self._pick(now)
            if job is not None:
                await self._execute(job)

    def _latest_start(self, job: Job, now: float) -> float:
        """Deadline minus the remaining work"""
        prof = self.engine.device_profile
        unc, host = self.engine.cost(job.toks)
        return job.deadline - unc / prof.prefill_tps_loaded - host / prof.promote_tps

    def _released(self, job: Job, now: float) -> bool:
        """Release in idle window or before the deadline"""
        return (not self.engine.serving
                or now >= self._latest_start(job, now) - self.RELEASE_SLACK_S)

    promote_enabled = True      # --no-promote: jobs only steer eviction, no host->device loads

    def _pool_full(self) -> bool:
        """If the KV cache already occupied/pinned by currently running requests exceeds PROMOTE_MAX_USAGE of the GPU KV-cache capacity, stop promoting KV from host memory back to GPU."""
        live = getattr(self.engine, "live_tokens", None)
        ledger = getattr(self.engine, "ledger", None)
        if live is None or ledger is None:
            return False
        cap = ledger.device_cap_tokens()
        return bool(cap) and live / cap > self.PROMOTE_MAX_USAGE

    def _pick(self, now: float) -> Optional[Job]:
        """Return the ready job that should be promoted next.

        A job is eligible when the cache currently contains more of its prefix
        than has already been restored to the device. Prefer the earliest
        deadline, breaking ties by expected value.

        Recheck the cache instead of relying on ``job.kind`` because another
        session may have cached a shared prefix since the job was created.
        """
        if not self.promote_enabled or self._pool_full():
            return
        best: Optional[Job] = None
        for held in self._jobs.values():
            for j in held.values():
                if j.state != "queued" or not self._released(j, now):
                    continue
                if j.cooldown_cycles > 0:      # refused for space recently; one cycle per pick
                    j.cooldown_cycles -= 1
                    continue
                if len(j.toks) - self.engine.cost(j.toks)[0] <= j.done_upto:
                    continue                   # nothing cached beyond the device frontier
                if best is None or (j.deadline, -j.value) < (best.deadline, -best.value):
                    best = j
        return best

    async def _execute(self, job: Job) -> None:
        if job.state != "queued":    # voided between pick and start
            return
        job.state = "running"
        rid = f"plan-{job.kind}-{uuid.uuid4().hex[:8]}"
        end = len(job.toks) - self.engine.cost(job.toks)[0]   # the whole cached frontier at once
        ok = await self.engine.promote(job.toks[:end], rid)
        if job.state == "void":
            return
        if ok is None:               # deferred behind real work: not a failure
            job.state = "queued"
            if self.engine.promote_no_room:
                self._cool_promotions()
            return
        if not ok:                   # the RPC itself failed
            job.attempts += 1
            job.state = "void" if job.attempts >= self.MAX_ATTEMPTS else "queued"
            return
        job.done_upto = end
        job.attempts = 0
        job.state = "done" if end >= len(job.toks) else "queued"

    def _cool_promotions(self) -> None:
        """No room is pool-wide, so pause all promotions for NO_ROOM_COOLDOWN_CYCLES."""
        for held in self._jobs.values():
            for j in held.values():
                if j.state == "queued" and len(j.toks) - self.engine.cost(j.toks)[0] > j.done_upto:
                    j.cooldown_cycles = self.NO_ROOM_COOLDOWN_CYCLES

    def _gc(self, now: float) -> None:
        # Keep completed jobs until they are replaced or the session ends, so note_arrival can retrieve them when the predicted call actually arrives.
        changed = False
        for sid, held in list(self._jobs.items()):
            for k, j in list(held.items()):
                if j.state == "void":
                    del held[k]
                    changed = True
                elif j.state == "queued" and now > j.t90 + self.STALE_SLACK_S:
                    j.state = "void"  # the predicted call never came
                    changed = True
            if not held:
                del self._jobs[sid]
                changed = True
        if changed:
            self._dirty = True         # remove stale protection on this tick

    # ------------------------------------------------------------------ eviction steering
    async def push_priorities(self, now: float) -> None:
        """Mirror the plan into the engine's eviction order."""
        if not self._dirty:
            return
        self._dirty = False
        # Prioritize predicted calls by the expected value of keeping their reusable prefix resident: reusable tokens × arrival probability /  time-to-use. Protect prefixes in descending priority until they occupy PROTECT_FRAC of the device KV-cache capacity.
        cands: List[Tuple[float, int, tuple]] = []
        for held in self._jobs.values():
            live = [j for j in held.values() if j.state != "void"]
            if not live:
                continue
            newest = max(j.epoch for j in live)   # older epochs' done jobs are history
            best: Dict[tuple, Tuple[float, int]] = {}
            for j in live:
                if j.epoch != newest:
                    continue
                saved = len(j.toks) - self.engine.cost(j.toks)[0]
                if saved <= 0:
                    continue
                density = j.p * saved / max(j.deadline - now, self.MIN_HORIZON_S)
                key = tuple(j.toks)
                if density > best.get(key, (0.0, 0))[0]:
                    best[key] = (density, saved)       # one prefix, one session: its best job
            cands.extend((d, saved, k) for k, (d, saved) in best.items())
        cands.sort(key=lambda c: -c[0])
        budget = self.PROTECT_FRAC * self.engine.ledger.device_cap_tokens() if cands else 0
        protect: List[Tuple[List[int], int]] = []
        used = 0
        top = cands[0][0] if cands else 1.0
        for density, saved, key in cands:
            if used + saved > budget and protect:
                break
            used += saved
            protect.append((list(key), max(1, int(self.SCORE_SCALE * density / top))))
        demote, self._demote = self._demote, []
        retire, self._retire = self._retire, []
        # An empty map is still a meaningful replacement: the scheduler RPC
        # removes protections installed by the previous plan.
        await self.engine.set_kv_priority(demote, protect, f"plan-prio-{uuid.uuid4().hex[:8]}",
                                          retire=retire)
