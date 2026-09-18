"""Continuum baseline (arXiv 2511.02230): KV-cache time-to-live for multi-turn agents.

Requests are opaque: no decompiler, no re-layout, no workflow model. The only
signals are the session id (`user`, Continuum's program id), the reply text
(parsed for a tool call, as Continuum's parser does) and arrival times.

Pinning   after every reply the served prompt + reply is pinned on the device for
          tau* = argmax_tau  P(tau, f) * (T * eta + PrefillReload(r)) - tau
          P(tau, f)         empirical CDF of past tool durations for tool f: the gap
                            from a reply that called f to the program's next request
          T                 sliding-window mean queueing delay of evicted requests
                            (requests that arrived with no pin to hit)
          eta               memoryfulness, -Corr(k, N - k) over finished programs
                            (k = turn index, N = turns); 1 until enough have finished
          PrefillReload(r)  n_tokens / profiled prefill rate (the full recompute)
          candidates are the observed durations plus tau = 0 (no pin).
Cold start  |S| <= K records: T_default, the same argmax under Exp(1) durations
          and eta = 1, i.e. ln(T + PrefillReload) (or a fixed --ttl-default);
          |S[f]| <= K: the global record set stands in for tool f.
Expiry    a pin is released once its TTL has passed and the program has no request
          waiting or running, or when the program closes.
Pressure  Continuum unpins the pinned program with the latest arrival first. Pins are
          eviction bands above unclassified, ranked by program arrival (earliest =
          highest band), so they go last and in that order; a hard pin would deadlock
          once the live set exceeds the pool. Host tier stays LRU.
Scheduling  plain engine FCFS, every request PRIORITY_REAL. Continuum's TTL priority
          tier and program-level (rather than request-level) FCFS are scheduling
          policy, outside this study's regime, and are not reproduced.

python -m server.continuum_server [MODEL] [--kv N] [--host GB] [--ttl-default S] [--no-pin]
python -m server.continuum_server --selftest
"""
import argparse
import asyncio
import os
import bisect
import json
import math
import re
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from server.wire import is_continuation   # transcript-prefix test only: requests stay opaque
from model.device_profiler import profile_device
from model.model import PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest

K = 100              # records before a CDF is trusted (Continuum's K)
QUEUE_WINDOW = 32    # sliding window of evicted-request queueing delays (Continuum's T)
MIN_PROGRAMS = 5     # finished programs before eta is estimated from data
TICK_S = 0.5         # pin expiry sweep period
IDLE_FORGET_S = 120  # drop per-session bookkeeping after this much silence
MAX_TOKENS = 4096    # decode cap when the request does not set one
GLOBAL = "*"         # key of the global record set (also the tool key of a reply with no tool call)

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class TTLPinner:
    def __init__(self, prefill_tps: Callable[[], float], default_s: Optional[float] = None) -> None:
        self.prefill_tps = prefill_tps
        self.default_s = default_s                               # None: ln(T + reload), the paper's T_default
        self.gaps: Dict[str, List[float]] = defaultdict(list)   # tool key -> sorted durations (S[f]); GLOBAL = S
        self.pins: Dict[str, Tuple[List[int], float]] = {}       # sid -> (ids, expiry)
        self.arrival: Dict[str, float] = {}                      # sid -> program arrival time (first request)
        self.last_key: Dict[str, str] = {}                       # sid -> tool the last reply called
        self.last_done: Dict[str, float] = {}                    # sid -> t of the last reply
        self.turns: Dict[str, int] = defaultdict(int)            # sid -> requests served so far
        self.queue: Deque[float] = deque(maxlen=QUEUE_WINDOW)    # evicted-request queueing delays
        self.kn: List[Tuple[int, int]] = []                      # (k, N - k) over finished programs

    # ---- observations
    def note_arrival(self, sid: str, t: float) -> Tuple[bool, Optional[float]]:
        """A request of `sid` arrived. The gap since its last reply is one duration of the
        tool that reply called (and one global record). Returns (whether the program had
        a live pin to hit, that gap); a miss makes it an evicted request for T."""
        self.arrival.setdefault(sid, t)
        self.turns[sid] += 1
        hit = self.within_ttl(sid, t)
        done = self.last_done.pop(sid, None)
        if done is None:
            return hit, None
        gap = max(0.0, t - done)
        for k in {self.last_key.get(sid, GLOBAL), GLOBAL}:
            bisect.insort(self.gaps[k], gap)   # unbounded, fine for <1e5 requests per run
        return hit, gap

    def note_queue(self, wait_s: float) -> None:
        self.queue.append(max(0.0, wait_s))

    def note_finished(self, sid: str) -> None:
        n = self.turns.get(sid, 0)
        if n >= 2:
            self.kn.extend((k, n - k) for k in range(1, n + 1))

    # ---- cost model
    def T(self) -> float:
        return sum(self.queue) / len(self.queue) if self.queue else 0.0

    def eta(self) -> float:
        """-Corr(k, N - k): 1 for fixed-length programs, ~0 for memoryless ones."""
        if len({n for _, n in self.kn}) < 2 or len(set(k for k, _ in self.kn)) < 2:
            return 1.0
        programs = sum(1 for k, _ in self.kn if k == 1)
        if programs < MIN_PROGRAMS:
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

    def benefit(self, n_tokens: int) -> float:
        return self.T() * self.eta() + n_tokens / self.prefill_tps()

    def ttl(self, key: str, n_tokens: int) -> float:
        xs = self.gaps[key] if len(self.gaps[key]) > K else self.gaps[GLOBAL]
        b = self.benefit(n_tokens)
        if len(xs) <= K:   # cold start: Exp(1) durations, eta = 1 -> argmax (1 - e^-tau) b - tau = ln b
            return self.default_s if self.default_s is not None else max(0.0, math.log(b)) if b > 0 else 0.0
        best, best_score = 0.0, 0.0   # tau = 0 is a candidate: a pin that never pays for itself is not made
        for i, tau in enumerate(xs):  # candidates: the observed durations
            if i + 1 < len(xs) and xs[i + 1] == tau:
                continue
            score = (i + 1) / len(xs) * b - tau
            if score > best_score:
                best, best_score = tau, score
        return best

    # ---- pins
    def pin(self, sid: str, ids: List[int], key: str, t_done: float) -> float:
        tau = self.ttl(key, len(ids))
        self.last_key[sid], self.last_done[sid] = key, t_done
        if tau > 0:
            self.pins[sid] = (ids, t_done + tau)
        else:
            self.pins.pop(sid, None)
        return tau

    def within_ttl(self, sid: str, now: float) -> bool:
        p = self.pins.get(sid)
        return p is not None and p[1] > now

    def forget(self, sid: str) -> bool:
        for d in (self.arrival, self.last_key, self.last_done, self.turns):
            d.pop(sid, None)
        return self.pins.pop(sid, None) is not None

    def expire(self, now: float, inflight: Callable[[str], int]) -> List[str]:
        """Continuum unpins expired entries whose program is not waiting in the queue."""
        dead = [sid for sid, (_, exp) in self.pins.items() if exp <= now and inflight(sid) == 0]
        for sid in dead:
            del self.pins[sid]
        for sid, t in list(self.last_done.items()):
            if now - t > IDLE_FORGET_S and inflight(sid) == 0:
                self.forget(sid)
        return dead

    def protect(self) -> List[Tuple[List[int], int]]:
        """Pins as eviction bands: the latest-arriving program is the first victim
        (band 1), the earliest the last."""
        order = sorted(self.pins, key=lambda s: self.arrival.get(s, 0.0), reverse=True)
        return [(self.pins[sid][0], rank) for rank, sid in enumerate(order, start=1)]


def _tool_key(text: str) -> str:
    """The tool the reply calls (Qwen native format), else the global key: replies without
    a tool call are followed by program logic whose duration has no name to file under."""
    m = _TOOL_CALL.search(text)
    if m:
        try:
            return str(json.loads(m.group(1)).get("name") or GLOBAL)
        except json.JSONDecodeError:
            pass
    return GLOBAL


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
        self.pin_enabled = pin   # --no-pin control: same bookkeeping and logs, no eviction-map pushes
        self.pinner = TTLPinner(
            prefill_tps=lambda: (self.engine.device_profile.prefill_tps_idle
                                 if self.engine.device_profile else 2000.0),
            default_s=ttl_default)
        self.inflight: Dict[str, int] = defaultdict(int)
        self.seq: Dict[str, int] = defaultdict(int)      # sid -> logical calls so far
        self.turn: Dict[str, int] = defaultdict(int)     # sid -> turn within the open call
        self.last_body: Dict[str, dict] = {}             # sid -> previous request, for is_continuation
        self._push_lock = asyncio.Lock()    # coalesces pin-map pushes
        self._rpc_lock = asyncio.Lock()     # the engine's collective_rpc is one ZMQ socket: one RPC at a time
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
            hit, gap = self.pinner.note_arrival(req.session, req.t_arrive)
            self.inflight[req.session] += 1
            asyncio.create_task(self._serve(req, hit, gap))

    async def _serve(self, req: PendingRequest, hit: bool, gap: Optional[float]) -> None:
        sid, body = req.session, req.body
        try:
            # a ReAct turn / typed retry continues the open call (same rule as server.server,
            # so the driver folds turns identically across arms)
            cont = sid in self.last_body and is_continuation(self.last_body[sid], body)
            if cont:
                self.turn[sid] += 1
            else:
                self.seq[sid] += 1
                self.turn[sid] = 1
                print(f"[call] {sid} #{self.seq[sid] - 1} opaque gap_s={-1.0 if gap is None else gap:.2f} "
                      f"hit={int(hit)}", flush=True)
            rid = f"{sid}-{self.seq[sid]}t{self.turn[sid]}-{uuid.uuid4().hex[:8]}"
            prompt = self.engine.render(body["messages"], tools=body.get("tools"))
            sp = {"temperature": 0.7 if body.get("temperature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            rf = body.get("response_format") or {}
            if rf.get("type") == "json_schema" and isinstance((rf.get("json_schema") or {}).get("schema"), dict):
                sp["json_schema"] = json.dumps(rf["json_schema"]["schema"])   # constrained decoding, as the client asked
            ids = self.engine.tokenize(prompt)
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
        if first and not hit and sid in self.pinner.last_key:
            # Continuum's T: the queueing delay an evicted (unpinned, returning) request paid,
            # time to first token minus its own full prefill
            self.pinner.note_queue(first[0] - t_call - len(ids) / self.pinner.prefill_tps())
        # the reply's KV is the next turn's prefix: pin prompt + reply for the learned TTL
        key = _tool_key(text)
        tau = self.pinner.pin(sid, ids + self.engine.tokenize(text), key, time.monotonic())
        print(f"[pin] {rid} tokens={len(ids)} ttl_s={tau:.2f} key={key} hit={int(hit)} "
              f"records={len(self.pinner.gaps[key])} T={self.pinner.T():.2f} pins={len(self.pinner.pins)}",
              flush=True)
        await self._push()

    async def _push(self) -> None:
        """Mirror the pin set into the engine's eviction bands (kv_priority replaces the
        previous map wholesale). Coalesces bursts: one push in flight, one queued."""
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
            dead = self.pinner.expire(time.monotonic(), lambda sid: self.inflight.get(sid, 0))
            if dead:
                print(f"[unpin] expired={len(dead)} pins={len(self.pinner.pins)}", flush=True)
                await self._push()


async def main(model: str, port: int, kv_tokens: Optional[int], host_gb: Optional[int],
               hicache_io: Optional[str], ttl_default: Optional[float], sched: str = "fcfs",
               pin: bool = True, eviction: str = "priority",
               engine_log: Optional[str] = None) -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 16384,
                              "radix_eviction_policy": eviction,
                              "grammar_backend": os.environ.get("GRAMMAR_BACKEND", "llguidance")}   # constrained decoding
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
    # cold start: T_default = ln(T*eta + reload); reload of 1000 tokens is 1 s -> ln(1) = 0
    assert p.ttl(GLOBAL, 1000) == 0.0
    p.note_queue(math.e - 1.0)                    # T = e - 1, eta = 1 -> ln(e) = 1
    assert abs(p.ttl(GLOBAL, 1000) - 1.0) < 1e-9
    assert TTLPinner(lambda: 1000.0, default_s=5.0).ttl(GLOBAL, 1000) == 5.0, "fixed override"
    p.queue.clear()
    for g in [1.0] * 60 + [4.0] * 60:             # > K records: the empirical CDF takes over
        bisect.insort(p.gaps[GLOBAL], g)
    # reload alone (1 s) never pays for a 1 s or 4 s pin: tau = 0
    assert p.ttl(GLOBAL, 1000) == 0.0
    # with 10 s of queueing to save, the 4 s pin (P=1) beats the 1 s pin (P=0.5)
    p.note_queue(10.0)
    assert p.ttl(GLOBAL, 1000) == 4.0
    # a per-tool CDF takes over once it has > K records
    for g in [0.5] * (K + 1):
        bisect.insort(p.gaps["search"], g)
    assert p.ttl("search", 1000) == 0.5
    t0 = 100.0
    assert p.note_arrival("s1", t0 - 1.0) == (False, None)
    assert p.pin("s1", [1] * 1000, "search", t0) == 0.5 and p.within_ttl("s1", t0 + 0.4)
    assert p.expire(t0 + 1.0, lambda _: 1) == [] and p.expire(t0 + 1.0, lambda _: 0) == ["s1"]
    assert p.note_arrival("s1", t0 + 2.0) == (False, 2.0)   # gap 2.0 lands in the tool's and the global CDF
    assert p.gaps["search"][-1] == 2.0 and p.gaps[GLOBAL][-1] == 4.0 and 2.0 in p.gaps[GLOBAL]
    p.pin("s1", [1] * 1000, "search", t0 + 3.0)
    assert p.note_arrival("s1", t0 + 3.2)[0] is True    # returned inside the TTL: a hit
    # victim order: latest-arriving program gets band 1, the earliest the highest band
    p.note_arrival("s2", t0 + 5.0); p.pin("s2", [2] * 10, "search", t0 + 6.0)
    assert [k for _, k in p.protect()] == [1, 2] and p.protect()[0][0] == [2] * 10
    # eta: fixed-length programs are fully memoryful, memoryless ones are not
    q = TTLPinner(lambda: 1000.0)
    for i in range(MIN_PROGRAMS):
        q.turns[f"f{i}"] = 4; q.note_finished(f"f{i}")
    assert q.eta() > 0.999
    q = TTLPinner(lambda: 1000.0)
    for i, n in enumerate([2, 3, 5, 8, 13, 2, 3, 5]):
        q.turns[f"g{i}"] = n; q.note_finished(f"g{i}")
    assert 0.0 <= q.eta() < 1.0
    assert _tool_key('x <tool_call>{"name": "search", "arguments": {}}</tool_call>') == "search"
    assert _tool_key("final answer") == GLOBAL
    print("continuum selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Continuum TTL baseline server")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel")
    ap.add_argument("--ttl-default", type=float, default=None, metavar="S",
                    help="fixed TTL while <= K records exist (default: the paper's ln(T + reload))")
    ap.add_argument("--sched", choices=["fcfs", "lpm", "risk"], default="fcfs",
                    help="engine queue order (fcfs is this baseline's setting; the others exist for the sweep script)")
    ap.add_argument("--no-pin", action="store_true",
                    help="control: identical server and engine config, but never push pins to the engine")
    ap.add_argument("--eviction", choices=["priority", "lru"], default="priority",
                    help="engine radix eviction policy; pins need priority. lru + --no-pin = the vanilla path through this server")
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
