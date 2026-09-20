"""CacheScout baseline (arXiv 2608.14624): learned agent execution for KV-cache management.

Requests are opaque: no decompiler, no re-layout, no workflow graph. CacheScout
identifies agents from the requests themselves and learns their execution order
online; everything below is that paper's mechanism mapped onto this engine.

Agent identity   prompt-prefix fingerprint over BLOCK-token block hashes. A prompt's
                 anchor is its longest block-prefix already seen from another
                 session (system prompt + tool/callsite header, never per-session
                 text); the agent id is that anchor's chain hash. Unknown until a
                 prefix has recurred across sessions.
Transitions      first-order Markov chain over agents, per session stream:
                 P_ij = (C_ij + EPS) / sum_k (C_ik + EPS), C_ij += 1 per dispatch.
Eviction         Score(b) = (p_surv(a_b) + DELTA) * (exp(-lambda*age(b)) + DELTA) * |b|.
                 p_surv(a): how soon agent a is reached again from the current agents
                 of the sessions active in the last IDLE_S seconds, a discounted
                 H-step reach probability. Blocks are tagged with the agent whose
                 request produced them (its anchor and its served prompt + reply).
                 Mapped onto the engine's eviction bands: band = quantised p_surv,
                 LRU inside a band is the age factor; |b| is not representable.
Prefetch         after a dispatch of agent i with argmax_j P_ij = j* and
                 predictability R = P_ij* >= R_MIN, warm j*'s anchor: a host->device
                 promotion when it is on the host tier. CacheScout's compute warmup
                 (a background request that builds the anchor) is not issued: anchors
                 here are ~300 shared tokens that stay device-resident, so it would
                 be a no-op, and this server does no speculative prefill.
No termination signal: CacheScout has none, so a session close only drops
bookkeeping; its blocks age out under the recency factor like any other.
Scheduling: standard FCFS, every request PRIORITY_REAL.

python -m server.cacheScout_server [MODEL] [--kv N] [--host GB] [--r-min R] [--horizon H]
python -m server.cacheScout_server --selftest
"""
import argparse
import asyncio
import os
import json
import math
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from server.wire import is_continuation   # transcript-prefix test only: requests stay opaque
from model.model import PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest

BLOCK = 16           # fingerprint block size (vLLM's default block, CacheScout's substrate)
EPS = 0.01           # Laplace smoothing of the transition counts
DELTA = 0.05         # score floor so no block is ever exactly zero
HORIZON = 3          # steps of look-ahead in p_surv
GAMMA = 0.9          # per-step discount in p_surv
R_MIN = 0.5          # prefetch only when the next agent is this predictable
IDLE_S = 120.0       # a session silent this long is no longer "current"
KEEP_PER_SESSION = 8 # served prompts kept tagged per session (RPC size cap)
BANDS = 8            # p_surv quantisation levels above the unclassified band
GRAMMAR_BACKEND = os.environ.get("GRAMMAR_BACKEND", "xgrammar")   # sglang grammar backend; "none" = no constrained decoding (schema not forwarded)
MAX_TOKENS = 4096

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class Fingerprint:
    """Block-hash trie over prompt prefixes; an anchor is the deepest block-prefix
    seen from >= 2 distinct sessions."""

    def __init__(self, block: int = BLOCK) -> None:
        self.block = block
        self.seen: Dict[int, Set[str]] = defaultdict(set)   # chain hash -> sessions (capped at 2)
        self.anchor_ids: Dict[int, List[int]] = {}           # agent id -> anchor token ids

    def chain(self, ids: List[int]) -> List[int]:
        out, h = [], 0
        for i in range(0, len(ids) - self.block + 1, self.block):
            h = hash((h, tuple(ids[i:i + self.block])))
            out.append(h)
        return out

    def observe(self, sid: str, ids: List[int]) -> Tuple[Optional[int], int]:
        """Record the prompt; return (agent id, anchor token length), agent None if
        no block-prefix of it has recurred across sessions yet."""
        agent, depth, shared = None, 0, True
        for k, h in enumerate(self.chain(ids)):
            s = self.seen[h]
            if len(s) < 2:
                s.add(sid)
            if shared and len(s) >= 2:
                agent, depth = h, (k + 1) * self.block
            else:
                shared = False   # the anchor is the consecutive shared prefix; keep recording blocks
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

    def row(self, i: int) -> Dict[int, float]:
        row = self.counts.get(i, {})
        n = len(self.agents)
        z = sum(row.values()) + EPS * n
        return {a: (row.get(a, 0) + EPS) / z for a in self.agents} if z > 0 else {}

    def predict(self, i: int) -> Tuple[Optional[int], float]:
        row = self.row(i)
        if not row:
            return None, 0.0
        j = max(row, key=row.get) #type: ignore
        return j, row[j]

    def reach(self, i: int, horizon: int = HORIZON, gamma: float = GAMMA) -> Dict[int, float]:
        """Discounted probability of visiting each agent within `horizon` steps from i."""
        dist = {i: 1.0}
        out: Dict[int, float] = defaultdict(float)
        for k in range(horizon):
            nxt: Dict[int, float] = defaultdict(float)
            for a, p in dist.items():
                for b, q in self.row(a).items():
                    nxt[b] += p * q
            for b, p in nxt.items():
                out[b] += gamma ** k * p
            dist = nxt
        return {b: min(1.0, p) for b, p in out.items()}


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
    def __init__(self, model: str, server: HttpServer, r_min: float = R_MIN, horizon: int = HORIZON,
                 **engine_kwargs) -> None:
        self.engine = Engine(model, **engine_kwargs)
        self.server = server
        self.pool = server.pool
        self.r_min, self.horizon = r_min, horizon
        self.fp = Fingerprint()
        self.chain = Markov()
        self.current: Dict[str, int] = {}                  # sid -> agent of its last dispatch
        self.last_seen: Dict[str, float] = {}              # sid -> monotonic time of last dispatch
        self.served: Dict[str, List[Tuple[List[int], int]]] = defaultdict(list)  # sid -> [(ids, agent)]
        self.seq: Dict[str, int] = defaultdict(int)
        self.turn: Dict[str, int] = defaultdict(int)
        self.last_body: Dict[str, dict] = {}
        self._push_lock = asyncio.Lock()    # coalesces score-map pushes
        self._rpc_lock = asyncio.Lock()     # the engine's collective_rpc is one ZMQ socket: one RPC at a time
        self._push_pending = False

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
        nxt, r = self.chain.predict(agent)
        if nxt is not None and r >= self.r_min and nxt in self.fp.anchor_ids:
            anchor = self.fp.anchor_ids[nxt]
            unc, host = self.engine.cost(anchor)
            if host > 0:   # anchor sits on the host tier: warm it onto the device
                async with self._rpc_lock:
                    ok = await self.engine.promote(anchor, f"scout-{rid}")
                print(f"[prefetch] {rid} next={nxt & 0xffffffff:08x} R={r:.2f} anchor={len(anchor)} "
                      f"host={host} started={ok}", flush=True)
        await self._push()

    def _protect(self) -> List[Tuple[List[int], int]]:
        """Every tagged block set with its agent's survival band; sessions silent
        for IDLE_S drop out of the current set (their blocks keep plain LRU)."""
        now = time.monotonic()
        surv: Dict[int, float] = defaultdict(float)
        for sid, a in self.current.items():
            if now - self.last_seen.get(sid, 0.0) > IDLE_S:
                continue
            for b, p in self.chain.reach(a, self.horizon).items():
                surv[b] = max(surv[b], p)
        out: List[Tuple[List[int], int]] = []
        for a, p in surv.items():
            k = _band(p)
            if k <= 0:
                continue
            if a in self.fp.anchor_ids:
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
        the previous map wholesale). Coalesces bursts: one push in flight, one queued."""
        if self._push_lock.locked():
            self._push_pending = True
            return
        async with self._push_lock:
            while True:
                self._push_pending = False
                prot = self._protect()
                async with self._rpc_lock:
                    await self.engine.set_kv_priority([], prot, f"scout-{uuid.uuid4().hex[:8]}")
                if not self._push_pending:
                    break

async def main(model: str, port: int, kv_tokens: Optional[int], host_gb: Optional[int],
               hicache_io: Optional[str], r_min: float, horizon: int, sched: str = "fcfs",
               engine_log: Optional[str] = None) -> None:
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
    ctrl = CacheScoutController(model, server, r_min=r_min, horizon=horizon, **kwargs)
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
    m = Markov()
    for _ in range(9):
        m.observe(a, b); m.observe(b, a)
    m.observe(a, a)
    nxt, r = m.predict(a)
    assert nxt == b and 0.85 < r < 0.95, (nxt, r)
    reach = m.reach(a, horizon=2)
    assert reach[b] > reach[a] > 0 and reach[b] <= 1.0
    assert _band(0.0) == 0 and _band(0.01) == 1 and _band(1.0) == BANDS
    print("cachescout selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CacheScout baseline server")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel")
    ap.add_argument("--r-min", type=float, default=R_MIN, help="prefetch when the next agent's probability >= R")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="look-ahead steps in the survival probability")
    ap.add_argument("--sched", choices=["fcfs", "lpm", "risk"], default="fcfs",
                    help="engine queue order (fcfs is the paper's setting; the others exist for the sweep script)")
    ap.add_argument("--engine-log", choices=["info", "warning", "error"], default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        asyncio.run(main(a.model, a.port, a.kv, a.host, a.hicache_io, a.r_min, a.horizon, a.sched,
                         engine_log=a.engine_log))
