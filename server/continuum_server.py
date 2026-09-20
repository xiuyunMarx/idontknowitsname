"""Continuum baseline (arXiv 2511.02230): keep a program's KV cache on the device for a
learned time-to-live, then let it go.

The server knows nothing about the program. It sees the session id (`user`), the
request bodies, the reply texts and the clock. It learns two things from them:

  how long the program takes to come back    after a reply, the program runs its own
        code (a tool, a loop, whatever) and then sends its next request. The wait is
        recorded under a key that says which kind of reply started it: the tool the
        reply called, or else the response schema the request asked for, or else
        one shared key. Continuum keys by tool name; the schema is the same idea for
        programs whose calls return typed values instead of tool calls.

  how much of the pinned prompt comes back   when the next request arrives, the
        server measures how many leading tokens it shares with each pinned prompt.
        That is the only part of the pin that saves any work, so it is what the pin
        is priced on and, once enough is known, all the pin covers. Continuum pins
        the whole context because in a chat loop the next turn extends it; here that
        is just the case where the shared prefix is the whole prompt.

Pinning   after every reply the prompt is pinned for
              tau* = argmax_tau  P(tau) * (T * eta + reload) - tau
          P(tau)   empirical CDF of past waits under the reply's key
          T        mean queueing delay of recent requests that arrived with no pin
          eta      memoryfulness, -Corr(turns done, turns left) over finished programs
          reload   expected reused tokens / profiled prefill rate
          Candidates are the observed waits plus tau = 0, which means "do not pin".
Cold start  fewer than K records under a key: the shared records; fewer than K shared
          records: T_default, the same argmax under Exp(1) waits and eta = 1, which is
          ln(T + reload). --ttl-default replaces that with a fixed number.
Expiry    a pin is dropped once its time is up and the program has nothing running.
Pressure  pins are eviction bands above ordinary cache. The program that arrived
          last is evicted first, as in the paper. A hard pin would deadlock once the
          live set exceeds the pool. The host tier stays LRU.
Scheduling  plain FCFS. Continuum also lets pinned requests jump the queue; that can
          starve new programs and is left out on purpose.

python -m server.continuum_server [MODEL] [--kv N] [--host GB] [--ttl-default S] [--no-pin]
python -m server.continuum_server --selftest
"""
import argparse
import asyncio
import bisect
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from model.device_profiler import profile_device
from model.model import PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest
from server.wire import is_continuation   # only to count turns in the log, as server.server does

K = 100              # records under a key before its distribution is trusted
QUEUE_WINDOW = 32    # how many recent unpinned arrivals the queueing delay T averages
MIN_PROGRAMS = 5     # finished programs before eta is estimated from data
TICK_S = 0.5         # how often expired pins are swept
IDLE_FORGET_S = 120  # a silent program is forgotten after this long
GRAMMAR_BACKEND = os.environ.get("GRAMMAR_BACKEND", "xgrammar")   # "none" = no constrained decoding
MAX_TOKENS = 4096    # decode cap when the request does not set one
GLOBAL = "*"         # the shared key

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class Pin:
    """One reply waiting for its program's next request."""
    sid: str
    key: str
    prompt: List[int]        # the whole prompt that was served
    kept: List[int]          # the part held on the device; empty once the pin lapses
    t_done: float            # when the reply finished
    expiry: float            # when the pin lapses



def common_prefix(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def reply_key(body: dict, text: str) -> str:
    """What kind of reply this was: the tool it called, else the schema the request
    asked for, else the shared key."""
    m = _TOOL_CALL.search(text)
    if m:
        try:
            name = json.loads(m.group(1)).get("name")
            if name:
                return f"tool:{name}"
        except json.JSONDecodeError:
            pass
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        name = js.get("name") or hashlib.sha1(json.dumps(js.get("schema"), sort_keys=True).encode()).hexdigest()[:8]
        return f"schema:{name}"
    return GLOBAL


class TTLPinner:
    def __init__(self, prefill_tps: Callable[[], float], default_s: Optional[float] = None) -> None:
        self.prefill_tps = prefill_tps
        self.default_s = default_s
        self.waits: Dict[str, List[float]] = defaultdict(list)   # key -> sorted waits until the next request
        self.reuse: Dict[str, List[int]] = defaultdict(list)     # key -> sorted reused prefix lengths
        self.pins: Dict[str, Pin] = {}                           # request id -> pin
        self.arrival: Dict[str, float] = {}                      # sid -> when the program first arrived
        self.turns: Dict[str, int] = defaultdict(int)            # sid -> requests so far
        self.queue: Deque[float] = deque(maxlen=QUEUE_WINDOW)    # queueing delays of unpinned arrivals
        self.kn: List[Tuple[int, int]] = []                      # (turns done, turns left) of finished programs

    # ---- what the server observes
    def note_arrival(self, sid: str, t: float) -> Tuple[bool, List[Pin]]:
        """A request of `sid` arrived. Every reply of that program still waiting for
        a next request is closed and its wait recorded. Returns whether any of those
        pins was still holding the device (a hit) and the closed pins, so the caller
        can measure the reused prefix once the prompt is tokenized."""
        self.arrival.setdefault(sid, t)
        self.turns[sid] += 1
        closed = [p for p in self.pins.values() if p.sid == sid]
        self.pins = {rid: p for rid, p in self.pins.items() if p.sid != sid}
        hit = any(p.kept and p.expiry > t for p in closed)
        for p in closed:
            for k in {p.key, GLOBAL}:
                bisect.insort(self.waits[k], max(0.0, t - p.t_done))
        return hit, closed

    def note_reuse(self, closed: List[Pin], prompt: List[int]) -> int:
        """How many leading tokens the new prompt shares with each closed pin."""
        best = 0
        for p in closed:
            n = common_prefix(p.prompt, prompt)
            best = max(best, n)
            for k in {p.key, GLOBAL}:
                bisect.insort(self.reuse[k], n)
        return best

    def note_queue(self, wait_s: float) -> None:
        self.queue.append(max(0.0, wait_s))

    def note_finished(self, sid: str) -> None:
        n = self.turns.get(sid, 0)
        if n >= 2:
            self.kn.extend((k, n - k) for k in range(1, n + 1))

    # ---- the cost model
    def T(self) -> float:
        return sum(self.queue) / len(self.queue) if self.queue else 0.0

    def eta(self) -> float:
        """-Corr(turns done, turns left): 1 for fixed-length programs, ~0 for memoryless ones."""
        if len({n for _, n in self.kn}) < 2 or len({k for k, _ in self.kn}) < 2:
            return 1.0
        if sum(1 for k, _ in self.kn if k == 1) < MIN_PROGRAMS:
            return 1.0
        n = len(self.kn)
        mx = sum(k for k, _ in self.kn) / n
        my = sum(r for _, r in self.kn) / n
        sxy = sum((k - mx) * (r - my) for k, r in self.kn)
        sxx = sum((k - mx) ** 2 for k, _ in self.kn)
        syy = sum((r - my) ** 2 for _, r in self.kn)
        if sxx <= 0 or syy <= 0:
            return 1.0
        return min(1.0, max(0.0, -sxy / math.sqrt(sxx * syy)))

    def _records(self, table: Dict[str, list], key: str) -> list:
        """The key's own records once there are more than K of them, else the shared ones."""
        return table[key] if len(table[key]) > K else table[GLOBAL]

    def expected_reuse(self, key: str, n_prompt: int) -> Tuple[int, int]:
        """(tokens a pin of this key is expected to save, tokens worth keeping).
        Whole prompt until more than K reuse records exist; then the mean and the
        90th percentile of what came back."""
        xs = self._records(self.reuse, key)
        if len(xs) <= K:
            return n_prompt, n_prompt
        mean = int(sum(xs) / len(xs))
        p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
        return min(mean, n_prompt), min(p90, n_prompt)

    def ttl(self, key: str, reuse_tokens: int) -> float:
        b = self.T() * self.eta() + reuse_tokens / self.prefill_tps()
        xs = self._records(self.waits, key)
        if len(xs) <= K:
            if self.default_s is not None:
                return self.default_s
            return max(0.0, math.log(b)) if b > 0 else 0.0
        best, best_score = 0.0, 0.0
        for i, tau in enumerate(xs):
            if i + 1 < len(xs) and xs[i + 1] == tau:
                continue
            score = (i + 1) / len(xs) * b - tau
            if score > best_score:
                best, best_score = tau, score
        return best

    # ---- pins
    def pin(self, rid: str, sid: str, prompt: List[int], key: str, t_done: float) -> Pin:
        """Record the reply and hold the part of its prompt the next request is expected
        to reuse. With tau = 0 nothing is held, but the reply is still recorded so its
        wait and reuse get measured."""
        save, keep = self.expected_reuse(key, len(prompt))
        tau = self.ttl(key, save)
        p = Pin(sid, key, prompt, prompt[:keep] if tau > 0 else [], t_done, t_done + tau)
        self.pins[rid] = p
        return p

    def within_ttl(self, sid: str, now: float) -> bool:
        return any(p.sid == sid and p.kept and p.expiry > now for p in self.pins.values())

    def forget(self, sid: str) -> bool:
        """Drop everything about a program. True if it was holding any device memory."""
        for d in (self.arrival, self.turns):
            d.pop(sid, None)
        held = any(p.sid == sid and p.kept for p in self.pins.values())
        self.pins = {rid: p for rid, p in self.pins.items() if p.sid != sid}
        return held

    def expire(self, now: float, inflight: Callable[[str], int]) -> int:
        """A lapsed pin stops holding the device once its program has nothing running.
        Its record stays until the next request arrives, so the wait is still learned.
        A program silent for a long time is forgotten."""
        n = 0
        for p in self.pins.values():
            if p.kept and p.expiry <= now and inflight(p.sid) == 0:
                p.kept = []
                n += 1
        for sid in {p.sid for p in self.pins.values()
                    if now - p.t_done > IDLE_FORGET_S and inflight(p.sid) == 0}:
            self.forget(sid)
        return n

    def protect(self) -> List[Tuple[List[int], int]]:
        """Pins as eviction bands. Band 1 (first victim) is the program that arrived
        last; the earliest program gets the highest band."""
        held = [p for p in self.pins.values() if p.kept]
        order = sorted({p.sid for p in held}, key=lambda s: self.arrival.get(s, 0.0), reverse=True)
        rank = {sid: i for i, sid in enumerate(order, start=1)}
        return [(p.kept, rank[p.sid]) for p in held]


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


class ContinuumController:
    def __init__(self, model: str, server: HttpServer, ttl_default: Optional[float] = None,
                 pin: bool = True, **engine_kwargs) -> None:
        self.engine = Engine(model, **engine_kwargs)
        self.server = server
        self.pool = server.pool
        self.pin_enabled = pin   # --no-pin: same bookkeeping and logs, nothing pushed to the engine
        self.pinner = TTLPinner(
            prefill_tps=lambda: (self.engine.device_profile.prefill_tps_idle
                                 if self.engine.device_profile else 2000.0),
            default_s=ttl_default)
        self.inflight: Dict[str, int] = defaultdict(int)
        self.seq: Dict[str, int] = defaultdict(int)      # sid -> calls so far (log only)
        self.turn: Dict[str, int] = defaultdict(int)     # sid -> turn within the open call (log only)
        self.last_body: Dict[str, dict] = {}
        self._push_lock = asyncio.Lock()    # one push in flight, one queued
        self._rpc_lock = asyncio.Lock()     # the engine's collective_rpc is one ZMQ socket
        self._push_pending = False

    async def start_serving(self) -> None:
        await self.engine.warmup()
        profile = f"model/{self.engine.name.replace('/', '--')}.profile.json"
        if not self.engine.load_profile(profile):
            await profile_device(self.engine)
            self.engine.save_profile(profile)
        asyncio.create_task(self._expire_loop())
        while True:
            req = await self.pool.get()
            if req.kind == "register":
                req.reply({"ok": True, "ignored": True, "file": req.body["file"]})
                continue
            if req.kind == "close":
                self.pinner.note_finished(req.session)
                closed = self.pinner.forget(req.session)
                for d in (self.seq, self.turn, self.last_body):
                    d.pop(req.session, None)
                print(f"[close] {req.session} eta={self.pinner.eta():.2f}", flush=True)
                req.reply(closed)
                if closed:
                    asyncio.create_task(self._push())
                continue
            hit, closed = self.pinner.note_arrival(req.session, req.t_arrive)
            self.inflight[req.session] += 1
            asyncio.create_task(self._serve(req, hit, closed))

    async def _serve(self, req: PendingRequest, hit: bool, closed: List[Pin]) -> None:
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
            ids = self.engine.tokenize(prompt)
            reused = self.pinner.note_reuse(closed, ids)
            if closed:
                waits = ",".join(f"{req.t_arrive - p.t_done:.2f}" for p in closed)
                print(f"[call] {sid} #{self.seq[sid] - 1} closes={len(closed)} wait_s={waits} "
                      f"reused={reused}/{len(ids)} hit={int(hit)}", flush=True)
                if any(p.kept for p in closed):
                    await self._push()   # the closed pins leave the eviction map
            sp = {"temperature": 0.7 if body.get("temperature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            rf = body.get("response_format") or {}
            if GRAMMAR_BACKEND != "none" and rf.get("type") == "json_schema" and isinstance((rf.get("json_schema") or {}).get("schema"), dict):
                sp["json_schema"] = json.dumps(rf["json_schema"]["schema"])
            first: List[float] = []
            t_call = time.perf_counter()
            text = await self.engine.generate(prompt, rid, sp, ids=ids, priority=PRIORITY_REAL,
                                              progress=lambda t_first, _n: first.append(t_first))
            req.reply(_answer(body, text))
            self.last_body[sid] = body
        except Exception as e:
            req.fail(e)
            return
        finally:
            self.inflight[sid] -= 1
        if first and closed and not hit:
            # T: the queueing delay a returning request paid when it found no pin,
            # time to first token minus its own full prefill
            self.pinner.note_queue(first[0] - t_call - len(ids) / self.pinner.prefill_tps())
        key = reply_key(body, text)
        p = self.pinner.pin(rid, sid, ids, key, time.monotonic())
        print(f"[pin] {rid} key={key} prompt={len(ids)} kept={len(p.kept)} ttl_s={p.expiry - p.t_done:.2f} "
              f"waits={len(self.pinner.waits[key])} reuse={len(self.pinner.reuse[key])} "
              f"T={self.pinner.T():.2f} pins={len(self.pinner.protect())}",
              flush=True)
        if p.kept:
            await self._push()

    async def _push(self) -> None:
        """Mirror the pins into the engine's eviction bands. kv_priority replaces the
        previous map wholesale, so bursts are coalesced: one push in flight, one queued."""
        if not self.pin_enabled:
            return
        if self._push_lock.locked():
            self._push_pending = True
            return
        async with self._push_lock:
            while True:
                self._push_pending = False
                prot = self.pinner.protect()
                async with self._rpc_lock:
                    await self.engine.set_kv_priority([], prot, f"ttl-{uuid.uuid4().hex[:8]}")
                if not self._push_pending:
                    break

    async def _expire_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            n = self.pinner.expire(time.monotonic(), lambda sid: self.inflight.get(sid, 0))
            if n:
                print(f"[unpin] expired={n} pins={len(self.pinner.protect())}", flush=True)
                await self._push()


async def main(model: str, port: int, kv_tokens: Optional[int], host_gb: Optional[int],
               hicache_io: Optional[str], ttl_default: Optional[float], sched: str = "fcfs",
               pin: bool = True, eviction: str = "priority",
               engine_log: Optional[str] = None) -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 32768,
                              "radix_eviction_policy": eviction,
                              "grammar_backend": GRAMMAR_BACKEND}
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
    ctrl = ContinuumController(model, server, ttl_default, pin=pin, **kwargs)
    await ctrl.start_serving()


def _selftest() -> None:
    p = TTLPinner(prefill_tps=lambda: 1000.0)
    # cold start: ln(T*eta + reload); reload of 1000 tokens is 1 s -> ln(1) = 0
    assert p.ttl(GLOBAL, 1000) == 0.0
    p.note_queue(math.e - 1.0)                    # T = e - 1, eta = 1 -> ln(e) = 1
    assert abs(p.ttl(GLOBAL, 1000) - 1.0) < 1e-9
    assert TTLPinner(lambda: 1000.0, default_s=5.0).ttl(GLOBAL, 1000) == 5.0
    p.queue.clear()
    for g in [1.0] * 60 + [4.0] * 60:             # more than K records: the CDF takes over
        bisect.insort(p.waits[GLOBAL], g)
    assert p.ttl(GLOBAL, 1000) == 0.0             # a 1 s reload never pays for a 1 s or 4 s pin
    p.note_queue(10.0)
    assert p.ttl(GLOBAL, 1000) == 4.0             # with 10 s of queueing to save, the 4 s pin wins
    for g in [0.5] * (K + 1):
        bisect.insort(p.waits["schema:Finding"], g)
    assert p.ttl("schema:Finding", 1000) == 0.5   # a key's own records take over past K
    # a program with two replies waiting: the next request closes both and measures reuse
    t0 = 100.0
    assert p.note_arrival("s1", t0 - 1.0) == (False, [])
    a = p.pin("r1", "s1", [1, 2, 3, 4], "schema:Finding", t0)
    b = p.pin("r2", "s1", [1, 2, 9, 9], "schema:Finding", t0 + 0.1)
    assert a.kept == [1, 2, 3, 4] and len(p.protect()) == 2
    assert p.within_ttl("s1", t0 + 0.4) and not p.within_ttl("s1", t0 + 0.7)
    assert p.expire(t0 + 0.6, lambda _: 1) == 0 and a.kept       # still running: keeps holding
    assert p.expire(t0 + 0.6, lambda _: 0) == 2 and not a.kept   # lapsed: lets go, record stays
    assert "r1" in p.pins and p.protect() == []
    hit, closed = p.note_arrival("s1", t0 + 0.8)
    assert not hit and {id(c) for c in closed} == {id(a), id(b)} and p.pins == {}
    assert any(abs(w - 0.8) < 1e-9 for w in p.waits["schema:Finding"])
    assert any(abs(w - 0.7) < 1e-9 for w in p.waits[GLOBAL])
    assert p.note_reuse(closed, [1, 2, 3, 7]) == 3 and p.reuse["schema:Finding"] == [2, 3]
    c = p.pin("r5", "s1", [5, 5], "schema:Finding", t0 + 1.0)
    assert p.note_arrival("s1", t0 + 1.2)[0] is True   # came back while held: a hit
    # once reuse is known, the pin is priced on the mean and covers the 90th percentile
    for n in [100] * 90 + [400] * 11:
        bisect.insort(p.reuse[GLOBAL], n)
    xs = p.reuse[GLOBAL]
    mean = int(sum(xs) / len(xs))
    assert p.expected_reuse("schema:Other", 1000) == (mean, 400)   # a key with no records uses the shared ones
    assert p.expected_reuse("schema:Other", 200) == (mean, 200)    # never more than the prompt has
    # victim order: the program that arrived last is band 1
    p.note_arrival("s2", t0 + 5.0); p.pin("r3", "s2", [7] * 10, GLOBAL, t0 + 6.0)
    p.pin("r4", "s1", [8] * 10, GLOBAL, t0 + 6.0)
    assert sorted((k, len(ids)) for ids, k in p.protect()) == [(1, 10), (2, 10)]
    assert dict((k, ids[0]) for ids, k in p.protect()) == {1: 7, 2: 8}
    # eta: fixed-length programs are fully memoryful, mixed lengths less so
    q = TTLPinner(lambda: 1000.0)
    for i in range(MIN_PROGRAMS):
        q.turns[f"f{i}"] = 4; q.note_finished(f"f{i}")
    assert q.eta() > 0.999
    q = TTLPinner(lambda: 1000.0)
    for i, n in enumerate([2, 3, 5, 8, 13, 2, 3, 5]):
        q.turns[f"g{i}"] = n; q.note_finished(f"g{i}")
    assert 0.0 <= q.eta() < 1.0
    # keys
    assert reply_key({}, 'x <tool_call>{"name": "search", "arguments": {}}</tool_call>') == "tool:search"
    assert reply_key({"response_format": {"type": "json_schema", "json_schema": {"name": "Finding", "schema": {}}}}, "{}") == "schema:Finding"
    k = reply_key({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}}, "{}")
    assert k.startswith("schema:") and len(k) == len("schema:") + 8
    assert reply_key({}, "final answer") == GLOBAL
    print("continuum selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Continuum TTL baseline server")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel")
    ap.add_argument("--ttl-default", type=float, default=None, metavar="S",
                    help="fixed TTL while fewer than K records exist (default: the paper's ln(T + reload))")
    ap.add_argument("--sched", choices=["fcfs", "lpm", "risk"], default="fcfs",
                    help="engine queue order (fcfs is this baseline's setting)")
    ap.add_argument("--no-pin", action="store_true",
                    help="control: same server and engine config, but never push pins to the engine")
    ap.add_argument("--eviction", choices=["priority", "lru"], default="priority",
                    help="engine radix eviction policy; pins need priority")
    ap.add_argument("--engine-log", choices=["info", "warning", "error"], default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        if not a.no_pin and a.eviction != "priority":
            ap.error("pins need --eviction priority")
        asyncio.run(main(a.model, a.port, a.kv, a.host, a.hicache_io, a.ttl_default, a.sched,
                         pin=not a.no_pin, eviction=a.eviction, engine_log=a.engine_log))
