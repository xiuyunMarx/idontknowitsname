"""CacheScout baseline (arXiv 2608.14624): learned agent execution for KV-cache management.

Requests are opaque: no decompiler, no re-layout, no workflow graph. CacheScout
identifies agents from the requests themselves and learns their execution order
online; everything below is that paper's mechanism mapped onto this engine.

Agent identity   prompt-prefix fingerprint over BLOCK-token block hashes (paper Sec. 4).
                 The anchor is the agent's fixed context (Sec. 2.2.1: system prompt,
                 tool definitions, few-shot): the block prefix every session running
                 the agent shares. The trie keeps per-block session support and the
                 anchor ends at the first block whose support falls below CLIFF of its
                 parent's (or below 2 sessions). Session text that a few sessions
                 happen to share (the same task) drops off that cliff once other
                 sessions disagree, so anchors stay at the fixed context instead of
                 growing into per-task prefixes. The agent id is the anchor's chain
                 hash; unknown until a prefix has recurred across sessions.
Transitions      first-order Markov chain over agents (Sec. 3.2), per session stream:
                 P_ij = (C_ij + EPS) / sum_k (C_ik + EPS), C_ij += 1 per dispatch.
Survival         Sec. 3.3, Eq. 7-8: the edges (a, b) with P_ab >= TAU form a sparse
                 execution graph; one BFS from the current agent gives hop distances
                 E[a], and p_surv(a) = 1 - min(E[a], E_MAX) / E_MAX. Every session
                 active in the last IDLE_S seconds contributes its current agent and a
                 block takes the max over them. Blocks are tagged with the agent whose
                 request produced them (its anchor and its served prompt + reply).
Eviction         Score(b) = (p_surv(a_b) + DELTA) * (exp(-lambda*age(b)) + DELTA) * |b|,
                 mapped onto the engine's eviction bands: band = quantised p_surv,
                 LRU inside a band is the age factor; |b| is not representable.
Prefetch         Sec. 3.4: when the chain's predictability R = 1 - H(next|cur)/H(next)
                 (Eq. 2) reaches R_MIN, warm the argmax successor's anchor, rate-limited
                 per anchor (WARMUP_MIN_S): a host->device promotion when it is on the
                 host tier. CacheScout's compute warmup (a max_tokens=1 request that
                 builds the anchor) is not issued: anchors here are a few hundred shared
                 tokens that stay device-resident or on the host tier, and this server
                 does no speculative prefill.
Score map        pushed to the scheduler after a dispatch, coalesced and at most once
                 per PUSH_MIN_S: the paper scores blocks inside the engine once per
                 scheduler step; here each push is a scheduler RPC that walks the
                 tagged prefixes, so it is rate-limited instead.
No termination signal: CacheScout has none, so a session close only drops
bookkeeping; its blocks age out under the recency factor like any other.
Scheduling: standard FCFS, every request PRIORITY_REAL.

python -m server.cacheScout_server [MODEL] [--kv N] [--host GB] [--r-min R] [--horizon H] [--tau T]
python -m server.cacheScout_server --relayout ...   the same policy on the re-laid prompts of `ours`
            (guard controller front: registration, callsite identification, re-layout; no host admission)
python -m server.cacheScout_server --selftest
"""
import argparse
import asyncio
import os
import json
import math
import random
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from server.wire import is_continuation   # transcript-prefix test only: requests stay opaque
from model.model import PRIORITY_CONT, PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest

BLOCK = 16           # fingerprint block size (vLLM's default block, CacheScout's substrate)
MAX_ANCHOR_BLOCKS = 256   # blocks of a prompt the fingerprint trie records (4096 tokens)
CLIFF = 0.5          # a block is fixed context while its session support >= CLIFF x its parent's
EPS = 0.01           # Laplace smoothing of the transition counts
DELTA = 0.05         # score floor so no block is ever exactly zero
TAU = 0.2            # confidence threshold of the execution graph's edges (Eq. 7)
E_MAX = 3            # hop-distance horizon of the survival score (Eq. 8)
R_MIN = 0.5          # prefetch only when the chain's predictability R (Eq. 2) is this high
WARMUP_MIN_S = 2.0   # the paper's rate limit on warmups: one per anchor per this interval
PUSH_MIN_S = 0.5     # score-map pushes to the scheduler are coalesced to one per this interval
IDLE_S = 120.0       # a session silent this long is no longer "current"
KEEP_PER_SESSION = 8 # served prompts kept tagged per session (RPC size cap)
BANDS = 8            # p_surv quantisation levels above the unclassified band
GRAMMAR_BACKEND = os.environ.get("GRAMMAR_BACKEND", "xgrammar")   # sglang grammar backend; "none" = no constrained decoding (schema not forwarded)
MAX_TOKENS = 4096

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class Fingerprint:
    """Block-hash trie over prompt prefixes with per-block session support. The
    anchor is the prefix whose support has not fallen off a cliff: the fixed
    context every session running the agent sends, not the task text a few of
    them share."""

    def __init__(self, block: int = BLOCK, cliff: float = CLIFF, max_blocks: int = MAX_ANCHOR_BLOCKS) -> None:
        self.block, self.cliff, self.max_blocks = block, cliff, max_blocks
        self.support: Dict[int, Set[str]] = defaultdict(set)   # chain hash -> sessions that sent it
        self.anchor_ids: Dict[int, List[int]] = {}             # agent id -> anchor token ids

    def chain(self, ids: List[int]) -> List[int]:
        out, h = [], 0
        for k in range(min(len(ids) // self.block, self.max_blocks)):
            i = k * self.block
            h = hash((h, tuple(ids[i:i + self.block])))
            out.append(h)
        return out

    def observe(self, sid: str, ids: List[int]) -> Tuple[Optional[int], int]:
        """Record the prompt; return (agent id, anchor token length), agent None if
        no block-prefix of it has recurred across sessions yet."""
        agent, depth, prev, fixed = None, 0, 0, True
        for k, h in enumerate(self.chain(ids)):
            s = self.support[h]
            s.add(sid)           # deeper blocks are recorded too: their support decides later anchors
            n = len(s)
            if fixed and n >= 2 and n >= self.cliff * prev:
                agent, depth, prev = h, (k + 1) * self.block, n
            else:
                fixed = False    # the anchor is the consecutive fixed prefix
        if agent is not None and agent not in self.anchor_ids:
            self.anchor_ids[agent] = list(ids[:depth])
        return agent, depth


class Markov:
    def __init__(self) -> None:
        self.counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.agents: Set[int] = set()

    def observe(self, i: int, j: int) -> None:
        self.counts[i][j] += 1
        self.agents.update((i, j))

    def prob(self, i: int, j: int) -> float:
        """Eq. 3, Laplace-smoothed over the known agents."""
        row = self.counts.get(i, {})
        z = sum(row.values()) + EPS * len(self.agents)
        return (row.get(j, 0) + EPS) / z if z > 0 else 0.0

    def predict(self, i: int) -> Tuple[Optional[int], float]:
        """Eq. 5: the most frequent observed successor and its probability."""
        row = self.counts.get(i)
        if not row:
            return None, 0.0
        j = max(row, key=row.get)  # type: ignore[arg-type]
        return j, self.prob(i, j)

    def successors(self, i: int, tau: float = TAU) -> List[int]:
        """Eq. 7: the high-confidence edges out of i. Only observed transitions can
        carry an edge: an unobserved one has P = EPS / z, smoothing mass only."""
        row = self.counts.get(i, {})
        z = sum(row.values()) + EPS * len(self.agents)
        return [j for j, c in row.items() if (c + EPS) / z >= tau]

    def hops(self, i: int, tau: float = TAU, e_max: int = E_MAX) -> Dict[int, int]:
        """Sec. 3.3: one BFS over the tau-graph from i; hop distance E[a] of every
        agent reachable in fewer than e_max hops (farther ones score 0 anyway)."""
        dist, frontier = {i: 0}, [i]
        for d in range(1, e_max):
            nxt = []
            for a in frontier:
                for b in self.successors(a, tau):
                    if b not in dist:
                        dist[b] = d
                        nxt.append(b)
            if not nxt:
                break
            frontier = nxt
        return dist

    def predictability(self) -> float:
        """Eq. 2: R = 1 - H(A_t+1 | A_t) / H(A_t+1) over the observed transitions."""
        total = sum(sum(row.values()) for row in self.counts.values())
        if total == 0:
            return 0.0
        marginal: Dict[int, int] = defaultdict(int)
        h_cond = 0.0
        for row in self.counts.values():
            n_i = sum(row.values())
            for j, c in row.items():
                marginal[j] += c
                h_cond -= (c / total) * math.log(c / n_i)
        h_next = -sum((c / total) * math.log(c / total) for c in marginal.values())
        if h_next <= 0.0:
            return 1.0   # one successor ever: fully determined
        return max(0.0, 1.0 - h_cond / h_next)


def _survival(hops: int, e_max: int) -> float:
    """Eq. 8."""
    return 1.0 - min(hops, e_max) / e_max


def _answer(body: dict, text: str) -> Any:
    """Native-tools replies carry structured tool_calls (same rule as server.server)."""
    if not body.get("tools") or "<tool_call>" not in text:
        return text
    calls = []
    for m in _TOOL_CALL.finditer(text):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": d.get("name", ""), "arguments": json.dumps(d.get("arguments", {}))}})
    if not calls:
        return text
    content = _TOOL_CALL.sub("", text).strip()
    return {"role": "assistant", "content": content or None, "tool_calls": calls}


def _band(p: float) -> int:
    return min(BANDS, int(math.ceil(p * BANDS)))


class CacheScoutController:
    def __init__(self, model: str, server: HttpServer, r_min: float = R_MIN, e_max: int = E_MAX,
                 tau: float = TAU, **engine_kwargs) -> None:
        self.engine = Engine(model, **engine_kwargs)
        self.server = server
        self.pool = server.pool
        self.r_min, self.e_max, self.tau = r_min, e_max, tau
        self.fp = Fingerprint()
        self.chain = Markov()
        self.current: Dict[str, int] = {}                  # sid -> agent of its last dispatch
        self.last_seen: Dict[str, float] = {}              # sid -> monotonic time of last dispatch
        self.served: Dict[str, List[Tuple[List[int], int]]] = defaultdict(list)  # sid -> [(ids, agent)]
        self.seq: Dict[str, int] = defaultdict(int)
        self.turn: Dict[str, int] = defaultdict(int)
        self.last_body: Dict[str, dict] = {}
        self.warmed: Dict[int, float] = {}                 # agent -> monotonic time of its last warmup
        self._push_lock = asyncio.Lock()    # coalesces score-map pushes
        self._rpc_lock = asyncio.Lock()     # the engine's collective_rpc is one ZMQ socket: one RPC at a time
        self._push_pending = False
        self._last_push = -math.inf

    async def start_serving(self) -> None:
        await self.engine.warmup()
        while True:
            req = await self.pool.get()
            if req.kind == "register":
                req.reply({"ok": True, "ignored": True, "file": req.body["file"]})
                continue
            if req.kind == "close":
                known = req.session in self.current or req.session in self.seq
                for d in (self.current, self.last_seen, self.served, self.seq, self.turn, self.last_body):
                    d.pop(req.session, None)   # bookkeeping only: no termination signal in CacheScout
                print(f"[close] {req.session}", flush=True)
                req.reply(known)
                continue
            asyncio.create_task(self._serve(req))

    async def _serve(self, req: PendingRequest) -> None:
        sid, body = req.session, req.body
        try:
            cont = sid in self.last_body and is_continuation(self.last_body[sid], body)
            if cont:
                self.turn[sid] += 1
            else:
                self.seq[sid] += 1
                self.turn[sid] = 1
            rid = f"{sid}-{self.seq[sid]}t{self.turn[sid]}-{uuid.uuid4().hex[:8]}"
            prompt = self.engine.render(body["messages"], tools=body.get("tools"))
            sp = {"temperature": 0.7 if body.get("temperature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            rf = body.get("response_format") or {}
            if GRAMMAR_BACKEND != "none" and rf.get("type") == "json_schema" and isinstance((rf.get("json_schema") or {}).get("schema"), dict):
                sp["json_schema"] = json.dumps(rf["json_schema"]["schema"])   # constrained decoding, as the client asked
            ids = self.engine.tokenize(prompt)
            agent, depth = self.fp.observe(sid, ids)
            if not cont:
                label = f"agent={agent & 0xffffffff:08x}" if agent is not None else "agent=?"
                print(f"[call] {sid} #{self.seq[sid] - 1} {label} anchor={depth}", flush=True)
                prev = self.current.get(sid)
                if agent is not None and prev is not None:
                    self.chain.observe(prev, agent)   # one dispatch = one transition
                if agent is not None:
                    self.current[sid] = agent
            self.last_seen[sid] = time.monotonic()
            text = await self.engine.generate(prompt, rid, sp, ids=ids, priority=PRIORITY_REAL)
            req.reply(_answer(body, text))
            self.last_body[sid] = body
        except Exception as e:
            req.fail(e)
            return
        if agent is not None:
            kept = self.served[sid]
            kept.append((ids + self.engine.tokenize(text), agent))   # blocks this agent produced
            del kept[:-KEEP_PER_SESSION]
            await self._after_dispatch(sid, agent, rid)
        else:
            await self._push()

    async def _after_dispatch(self, sid: str, agent: int, rid: str) -> None:
        """Sec. 3.4: gated on the chain's predictability, warm the predicted next agent's anchor."""
        nxt, p = self.chain.predict(agent)
        r = self.chain.predictability()
        now = time.monotonic()
        if (nxt is not None and r >= self.r_min and nxt in self.fp.anchor_ids
                and now - self.warmed.get(nxt, -math.inf) >= WARMUP_MIN_S):
            self.warmed[nxt] = now
            anchor = self.fp.anchor_ids[nxt]
            unc, host = self.engine.cost(anchor)
            if host > 0:   # anchor sits on the host tier: warm it onto the device
                async with self._rpc_lock:
                    ok = await self.engine.promote(anchor, f"scout-{rid}")
                print(f"[prefetch] {rid} next={nxt & 0xffffffff:08x} P={p:.2f} R={r:.2f} anchor={len(anchor)} "
                      f"host={host} started={ok}", flush=True)
        await self._push()

    def _protect(self) -> List[Tuple[List[int], int]]:
        """Every tagged block set with its agent's survival band; sessions silent
        for IDLE_S drop out of the current set (their blocks keep plain LRU)."""
        now = time.monotonic()
        surv: Dict[int, float] = defaultdict(float)
        hops: Dict[int, Dict[int, int]] = {}   # one BFS per distinct current agent
        for sid, a in self.current.items():
            if now - self.last_seen.get(sid, 0.0) > IDLE_S:
                continue
            if a not in hops:
                hops[a] = self.chain.hops(a, self.tau, self.e_max)
            for b, e in hops[a].items():
                surv[b] = max(surv[b], _survival(e, self.e_max))
        out: List[Tuple[List[int], int]] = []
        for a, p in surv.items():
            k = _band(p)
            if k > 0 and a in self.fp.anchor_ids:
                out.append((self.fp.anchor_ids[a], k))
        for sid, kept in self.served.items():
            if now - self.last_seen.get(sid, 0.0) > IDLE_S:
                continue
            for ids, a in kept:
                k = _band(surv.get(a, 0.0))
                if k > 0:
                    out.append((ids, k))
        return out

    async def _push(self) -> None:
        """Mirror the score map into the engine's eviction bands (kv_priority replaces
        the previous map wholesale). Coalesces bursts: one push in flight, one queued,
        and at most one push per PUSH_MIN_S."""
        if self._push_lock.locked():
            self._push_pending = True
            return
        async with self._push_lock:
            while True:
                wait = PUSH_MIN_S - (time.monotonic() - self._last_push)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._push_pending = False
                prot = self._protect()
                async with self._rpc_lock:
                    await self.engine.set_kv_priority([], prot, f"scout-{uuid.uuid4().hex[:8]}")
                self._last_push = time.monotonic()
                if not self._push_pending:
                    break


# ---------------------------------------------------------------- --relayout: the policy behind the guard controller's re-layout
class CacheScoutPolicy:
    """CacheScout's fingerprint + chain + survival bands in the guard controller's
    planner slot; the controller's intake calls are no-ops, the policy observes
    requests and replies at the engine call (see continuum_server._relayout_controller)."""

    promote_enabled = False

    def __init__(self, engine, r_min: float = R_MIN, e_max: int = E_MAX, tau: float = TAU) -> None:
        self.engine = engine
        self.ctrl = None
        self.r_min, self.e_max, self.tau = r_min, e_max, tau
        self.fp = Fingerprint()
        self.chain = Markov()
        self.current: Dict[str, int] = {}
        self.last_seen: Dict[str, float] = {}
        self.served: Dict[str, List[Tuple[List[int], int]]] = defaultdict(list)
        self.warmed: Dict[int, float] = {}
        self._agent: Dict[str, Optional[int]] = {}         # rid -> agent of the request
        self._push_lock = asyncio.Lock()
        self._rpc_lock = asyncio.Lock()
        self._push_pending = False
        self._last_push = -math.inf

    # the controller's planner interface: nothing to do for a request-level policy
    def note_arrival(self, sid: str, site: str, t_arrive: float) -> None: ...
    def note_served(self, sid: str, ids: List[int], fixed_len: int, *, static_len: int) -> None: ...
    def invalidate(self, sid: str, ids: List[int], keep_len: int) -> None: ...
    def void_session(self, sid: str, epoch: int) -> None: ...
    def submit(self, sid: str, epoch: int, jobs: list, extend: bool = False) -> None: ...
    def retire_session(self, sid: str) -> None: ...
    def wake(self) -> None: ...

    def drop_session(self, sid: str) -> None:
        for d in (self.current, self.last_seen, self.served):
            d.pop(sid, None)   # bookkeeping only: no termination signal in CacheScout
        print(f"[close] {sid}", flush=True)

    # what the engine call shows
    def before(self, rid: str, ids: List[int], t_arrive: float, cont: bool = False) -> None:
        sid = rid.split("-")[0]
        agent, depth = self.fp.observe(sid, ids)
        if not cont:
            label = f"agent={agent & 0xffffffff:08x}" if agent is not None else "agent=?"
            print(f"[scout] {rid} {label} anchor={depth}", flush=True)
            prev = self.current.get(sid)
            if agent is not None and prev is not None:
                self.chain.observe(prev, agent)   # one dispatch = one transition
            if agent is not None:
                self.current[sid] = agent
        self.last_seen[sid] = time.monotonic()
        self._agent[rid] = agent

    async def after(self, rid: str, ids: List[int], sp: dict, text: str, ttft_s: Optional[float], t_done: float) -> None:
        sid = rid.split("-")[0]
        agent = self._agent.pop(rid, None)
        if agent is None:
            await self._push()
            return
        kept = self.served[sid]
        kept.append((ids + self.engine.tokenize(text), agent))   # blocks this agent produced
        del kept[:-KEEP_PER_SESSION]
        nxt, p = self.chain.predict(agent)    # Sec. 3.4: warm the predicted next agent's anchor
        r = self.chain.predictability()
        now = time.monotonic()
        if (nxt is not None and r >= self.r_min and nxt in self.fp.anchor_ids
                and now - self.warmed.get(nxt, -math.inf) >= WARMUP_MIN_S):
            self.warmed[nxt] = now
            anchor = self.fp.anchor_ids[nxt]
            unc, host = self.engine.cost(anchor)
            if host > 0:
                async with self._rpc_lock:
                    ok = await self.engine.promote(anchor, f"scout-{rid}")
                print(f"[prefetch] {rid} next={nxt & 0xffffffff:08x} P={p:.2f} R={r:.2f} anchor={len(anchor)} "
                      f"host={host} started={ok}", flush=True)
        await self._push()

    def _protect(self) -> List[Tuple[List[int], int]]:
        return CacheScoutController._protect(self)   # type: ignore[arg-type]  # same score map, same fields

    async def _push(self) -> None:
        if self._push_lock.locked():
            self._push_pending = True
            return
        async with self._push_lock:
            while True:
                wait = PUSH_MIN_S - (time.monotonic() - self._last_push)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._push_pending = False
                prot = self._protect()
                async with self._rpc_lock:
                    await self.engine.set_kv_priority([], prot, f"scout-{uuid.uuid4().hex[:8]}")
                self._last_push = time.monotonic()
                if not self._push_pending:
                    break

    async def run(self) -> None:
        while True:                 # everything is event-driven; the planner slot only needs a task
            await asyncio.sleep(60)


async def main(model: str, port: int, kv_tokens: Optional[int], host_gb: Optional[int],
               hicache_io: Optional[str], r_min: float, e_max: int, tau: float, sched: str = "fcfs",
               engine_log: Optional[str] = None, relayout: bool = False) -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 32768, "radix_eviction_policy": "priority",
                              "grammar_backend": GRAMMAR_BACKEND}   # constrained decoding
    if kv_tokens:
        kwargs["max_total_tokens"] = kv_tokens
    if host_gb is not None:
        kwargs["host_cache_gb"] = host_gb
    if hicache_io:
        kwargs["hicache_io_backend"] = hicache_io
    if engine_log:
        kwargs["log_level"] = engine_log
    if sched == "lpm":
        kwargs["schedule_policy"] = "lpm"
        kwargs["enable_priority_scheduling"] = False
    elif sched == "risk":
        kwargs["schedule_policy"] = "cache-risk"
        kwargs["enable_priority_scheduling"] = False
    if relayout:
        from server.continuum_server import _relayout_controller
        print("[cachescout] learned agent execution on re-laid prompts (guard controller front, no host admission)", flush=True)
        ctrl = _relayout_controller(CacheScoutPolicy(None, r_min=r_min, e_max=e_max, tau=tau), model, server, **kwargs)
        ctrl.planner.engine = ctrl.engine   # type: ignore[union-attr]
        await ctrl.start_serving()
        return
    ctrl = CacheScoutController(model, server, r_min=r_min, e_max=e_max, tau=tau, **kwargs)
    await ctrl.start_serving()


def _selftest() -> None:
    fp = Fingerprint(block=4)
    sysp = [1, 2, 3, 4, 5, 6, 7, 8]           # shared "system prompt", two blocks
    hdr_a, hdr_b = [10, 11, 12, 13], [20, 21, 22, 23]
    # first session: nothing has recurred yet
    assert fp.observe("s1", sysp + hdr_a + [100, 101, 102, 103]) == (None, 0)
    # second session, same agent: anchor = system + header (claim block differs)
    a, d = fp.observe("s2", sysp + hdr_a + [200, 201, 202, 203])
    assert a is not None and d == 12
    b0, d2 = fp.observe("s2", sysp + hdr_b + [200, 201, 202, 203])
    assert b0 is not None and b0 != a and d2 == 8, "B's header has not recurred: system-prompt-only agent"
    b, d2 = fp.observe("s1", sysp + hdr_b + [100, 101, 102, 103])
    assert b is not None and b not in (a, b0) and d2 == 12
    assert fp.anchor_ids[a] == sysp + hdr_a and fp.anchor_ids[b] == sysp + hdr_b
    # the cliff: task text two sessions share looks fixed until other sessions disagree
    fp2 = Fingerprint(block=4)
    task = [30, 31, 32, 33]
    fp2.observe("u1", sysp + hdr_a + task)
    _, d = fp2.observe("u2", sysp + hdr_a + task)
    assert d == 16, "two sessions only: the shared task block passes as anchor"
    for i in range(3, 9):
        fp2.observe(f"u{i}", sysp + hdr_a + [50 + i] * 4)
    a2, d = fp2.observe("u9", sysp + hdr_a + task)
    assert d == 12, "3 of 9 sessions share the task: it fell off the cliff"
    a3, _ = fp2.observe("u10", sysp + hdr_a + [70, 71, 72, 73])
    assert a3 == a2 and fp2.anchor_ids[a2] == sysp + hdr_a, "one agent per fixed context"
    m = Markov()
    for _ in range(9):
        m.observe(a, b); m.observe(b, a)
    m.observe(a, a)
    nxt, p = m.predict(a)
    assert nxt == b and 0.85 < p < 0.95, (nxt, p)
    assert m.hops(a) == {a: 0, b: 1}, "the a->a edge (P=0.1) is below TAU"
    assert m.hops(b) == {b: 0, a: 1}
    assert 0.7 < m.predictability() < 0.8, m.predictability()
    rnd = random.Random(0)
    noise = Markov()
    for _ in range(4000):
        noise.observe(rnd.randrange(4), rnd.randrange(4))
    assert noise.predictability() < 0.05, noise.predictability()
    # many agents: the scorer stays a sparse BFS, not a dense matrix walk
    big = Markov()
    for i in range(2000):
        big.observe(i, (i + 1) % 2000)
    for _ in range(6000):
        big.observe(rnd.randrange(2000), rnd.randrange(2000))
    t0 = time.perf_counter()
    for i in range(40):
        big.hops(i)
    big.predictability()
    dt = time.perf_counter() - t0
    assert dt < 0.5, f"40 BFS + R over 2000 agents took {dt:.2f}s"
    assert _survival(0, 3) == 1.0 and _survival(3, 3) == 0.0 and _survival(9, 3) == 0.0
    assert _band(0.0) == 0 and _band(0.01) == 1 and _band(1.0) == BANDS
    print("cachescout selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CacheScout baseline server")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel")
    ap.add_argument("--r-min", type=float, default=R_MIN, help="prefetch when the chain's predictability R >= this")
    ap.add_argument("--horizon", type=int, default=E_MAX, help="E_max: hop-distance horizon of the survival score")
    ap.add_argument("--tau", type=float, default=TAU, help="edge confidence threshold of the execution graph")
    ap.add_argument("--sched", choices=["fcfs", "lpm", "risk"], default="fcfs",
                    help="engine queue order (fcfs is the paper's setting; the others exist for the sweep script)")
    ap.add_argument("--engine-log", choices=["info", "warning", "error"], default=None)
    ap.add_argument("--relayout", action="store_true",
                    help="the same policy on the re-laid prompts of `ours` (guard controller front)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        asyncio.run(main(a.model, a.port, a.kv, a.host, a.hicache_io, a.r_min, a.horizon, a.tau, a.sched,
                         engine_log=a.engine_log, relayout=a.relayout))
