"""Continuum baseline (arXiv 2511.02230): KV-cache time-to-live for multi-turn agents.

Requests are opaque: no decompiler, no re-layout, no workflow model. The only
signals are the session id (`user`, Continuum's program id), the reply text
(parsed for a tool call, as Continuum's scheduler does) and arrival times.

After every reply the served prompt (+ reply) is pinned on the device for a TTL:
    tau* = argmax_tau  P(tau | tool) * Benefit(r) - Cost(tau, r)
    P            empirical CDF of past gaps (reply -> the session's next request),
                 keyed by the tool the reply called, global while < K records
    Benefit(r)   reload cost n_tokens / prefill_tps + queueing delay T * ETA
    Cost(tau, r) tau * n_tokens / mean_tokens      (memory-share opportunity cost)
A request arriving inside its session's TTL is scheduled ahead of others
(PRIORITY_CONT); a pin is released at expiry (unless the session has a request
in flight) or when the session closes.  Pins are a band above unclassified in
the engine's eviction order, evicted last rather than never: a hard pin would
deadlock once the live set exceeds the pool.  Host tier stays LRU.

python -m server.continuum_server [MODEL] [--kv N] [--host GB] [--ttl-default S] [--no-ttl-sched]
python -m server.continuum_server --selftest
"""
import argparse
import asyncio
import bisect
import json
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from decompiler.parser import is_continuation   # transcript-prefix test only: no decompiling
from model.device_profiler import profile_device
from model.model import PRIORITY_CONT, PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest

K = 100              # records before a CDF is trusted (Continuum's K)
ETA = 1.0            # weight of the queueing-delay term in Benefit
TICK_S = 0.5         # pin expiry sweep period
IDLE_FORGET_S = 120  # drop per-session bookkeeping after this much silence
MAX_TOKENS = 4096    # decode cap when the request does not set one
PIN_K = 1            # eviction band of a pin: model.promote.PLAN_BASE + PIN_K

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class TTLPinner:
    def __init__(self, prefill_tps: Callable[[], float], default_s: float) -> None:
        self.prefill_tps = prefill_tps
        self.default_s = default_s
        self.gaps: Dict[str, List[float]] = defaultdict(list)   # tool key -> sorted gap seconds
        self.pins: Dict[str, Tuple[List[int], float]] = {}       # sid -> (ids, expiry)
        self.last_key: Dict[str, str] = {}                       # sid -> tool the last reply called
        self.last_done: Dict[str, float] = {}                    # sid -> t of the last reply
        self.queue_s = 0.0                                       # EMA of per-request queueing delay
        self.mean_tokens = 0.0                                   # EMA of pinned prompt length

    # ---- observations
    def note_arrival(self, sid: str, t: float) -> None:
        """A request of `sid` arrived: the gap since its last reply is one tool duration."""
        done = self.last_done.pop(sid, None)
        if done is None:
            return
        gap = max(0.0, t - done)
        for k in {self.last_key.get(sid, "*"), "*"}:
            bisect.insort(self.gaps[k], gap)   # ponytail: unbounded, fine for <1e5 calls per run

    def note_queue(self, wait_s: float) -> None:
        self.queue_s = wait_s if self.queue_s == 0.0 else 0.9 * self.queue_s + 0.1 * wait_s

    # ---- cost model
    def ttl(self, key: str, n_tokens: int) -> float:
        xs = self.gaps[key] if len(self.gaps[key]) >= K else self.gaps["*"]
        if len(xs) < K:
            return self.default_s
        benefit = self.queue_s * ETA + n_tokens / self.prefill_tps()
        weight = n_tokens / (self.mean_tokens or n_tokens)
        best, best_score = 0.0, 0.0   # a pin that never pays for itself is not made
        for i, tau in enumerate(xs):  # candidates: the unique observed durations
            if i + 1 < len(xs) and xs[i + 1] == tau:
                continue
            score = (i + 1) / len(xs) * benefit - tau * weight
            if score > best_score:
                best, best_score = tau, score
        return best

    # ---- pins
    def pin(self, sid: str, ids: List[int], key: str, t_done: float) -> float:
        n = len(ids)
        self.mean_tokens = n if self.mean_tokens == 0.0 else 0.9 * self.mean_tokens + 0.1 * n
        tau = self.ttl(key, n)
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
        self.last_key.pop(sid, None)
        self.last_done.pop(sid, None)
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
        return [(ids, PIN_K) for ids, _ in self.pins.values()]


def _tool_key(text: str) -> str:
    """The tool the reply calls (Qwen native format), else one global key."""
    m = _TOOL_CALL.search(text)
    if m:
        try:
            return str(json.loads(m.group(1)).get("name") or "*")
        except json.JSONDecodeError:
            pass
    return "*"


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
    def __init__(self, model: str, server: HttpServer, ttl_default: float, ttl_sched: bool = True,
                 **engine_kwargs) -> None:
        self.engine = Engine(model, **engine_kwargs)
        self.server = server
        self.pool = server.pool
        self.ttl_sched = ttl_sched
        self.pinner = TTLPinner(
            prefill_tps=lambda: (self.engine.device_profile.prefill_tps_idle
                                 if self.engine.device_profile else 2000.0),
            default_s=ttl_default)
        self.inflight: Dict[str, int] = defaultdict(int)
        self.seq: Dict[str, int] = defaultdict(int)      # sid -> logical calls so far
        self.turn: Dict[str, int] = defaultdict(int)     # sid -> turn within the open call
        self.last_body: Dict[str, dict] = {}             # sid -> previous request, for is_continuation
        self._push_lock = asyncio.Lock()

    async def start_serving(self) -> None:
        await self.engine.warmup()
        profile = f"model/{self.engine.name.replace('/', '--')}.profile.json"
        if not self.engine.load_profile(profile):
            await profile_device(self.engine)
            self.engine.save_profile(profile)
        asyncio.create_task(self._expire_loop())
        while True:
            req = await self.pool.get()
            if req.kind == "close":
                closed = self.pinner.forget(req.session)
                self.seq.pop(req.session, None)
                self.turn.pop(req.session, None)
                self.last_body.pop(req.session, None)
                print(f"[close] {req.session}", flush=True)
                req.reply(closed)
                if closed:
                    asyncio.create_task(self._push())
                continue
            self.pinner.note_arrival(req.session, req.t_arrive)
            self.inflight[req.session] += 1
            asyncio.create_task(self._serve(req))

    async def _serve(self, req: PendingRequest) -> None:
        sid, body = req.session, req.body
        try:
            now = time.monotonic()
            # a ReAct turn / typed retry continues the open call (same rule as server.server,
            # so the driver folds turns identically across arms); it keeps PRIORITY_CONT there too
            cont = sid in self.last_body and is_continuation(self.last_body[sid], body)
            priority = PRIORITY_REAL
            if cont or (self.ttl_sched and self.pinner.within_ttl(sid, now)):
                priority = PRIORITY_CONT   # returned within TTL: ahead of unpinned requests
            if cont:
                self.turn[sid] += 1
            else:
                self.seq[sid] += 1
                self.turn[sid] = 1
                print(f"[call] {sid} #{self.seq[sid] - 1} opaque", flush=True)
            rid = f"{sid}-{self.seq[sid]}t{self.turn[sid]}-{uuid.uuid4().hex[:8]}"
            prompt = self.engine.render(body["messages"], tools=body.get("tools"))
            sp = {"temperature": 0.7 if body.get("temperature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            ids = self.engine.tokenize(prompt)
            first: List[float] = []
            t_call = time.perf_counter()
            text = await self.engine.generate(prompt, rid, sp, ids=ids, priority=priority,
                                              progress=lambda t_first, _n: first.append(t_first))
            req.reply(_answer(body, text))
            self.last_body[sid] = body
        except Exception as e:
            req.fail(e)
            return
        finally:
            self.inflight[sid] -= 1
        if first:  # Continuum's T: engine queueing delay = time to first token minus own prefill
            self.pinner.note_queue(max(0.0, first[0] - t_call - len(ids) / self.pinner.prefill_tps()))
        # the reply's KV is the next turn's prefix: pin prompt + reply for the learned TTL
        key = _tool_key(text)
        tau = self.pinner.pin(sid, ids + self.engine.tokenize(text), key, time.monotonic())
        print(f"[pin] {rid} tokens={len(ids)} ttl_s={tau:.2f} key={key} "
              f"records={len(self.pinner.gaps[key])} pins={len(self.pinner.pins)}", flush=True)
        if tau > 0:
            await self._push()

    async def _push(self) -> None:
        """Mirror the pin set into the engine's eviction bands (kv_priority replaces
        the previous map wholesale, so the whole set goes every time)."""
        async with self._push_lock:
            await self.engine.set_kv_priority([], self.pinner.protect(), f"ttl-{uuid.uuid4().hex[:8]}")

    async def _expire_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            dead = self.pinner.expire(time.monotonic(), lambda sid: self.inflight.get(sid, 0))
            if dead:
                print(f"[unpin] expired={len(dead)} pins={len(self.pinner.pins)}", flush=True)
                await self._push()


async def main(model: str, port: int, kv_tokens: Optional[int], host_gb: Optional[int],
               hicache_io: Optional[str], ttl_default: float, ttl_sched: bool, sched: str = "fcfs") -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 16384, "radix_eviction_policy": "priority"}
    if kv_tokens:
        kwargs["max_total_tokens"] = kv_tokens
    if host_gb is not None:
        kwargs["host_cache_gb"] = host_gb
    if hicache_io:
        kwargs["hicache_io_backend"] = hicache_io
    if sched == "lpm":   # engine orders by matched prefix; request priorities (the TTL tier) are off
        kwargs["schedule_policy"] = "lpm"
        kwargs["enable_priority_scheduling"] = False
    ctrl = ContinuumController(model, server, ttl_default, ttl_sched, **kwargs)
    await ctrl.start_serving()


def _selftest() -> None:
    p = TTLPinner(prefill_tps=lambda: 1000.0, default_s=5.0)
    assert p.ttl("*", 1000) == 5.0, "cold start uses the default"
    for g in [1.0] * 60 + [4.0] * 60:
        bisect.insort(p.gaps["*"], g)
    p.mean_tokens = 1000.0
    # reload alone (1 s) never pays for a 1 s or 4 s pin: no pin
    assert p.ttl("*", 1000) == 0.0
    # with 10 s of queueing to save, the 4 s pin (P=1) beats the 1 s pin (P=0.5)
    p.queue_s = 10.0
    assert p.ttl("*", 1000) == 4.0
    # a per-tool CDF takes over once it has K records
    for g in [0.5] * K:
        bisect.insort(p.gaps["search"], g)
    assert p.ttl("search", 1000) == 0.5
    t0 = 100.0
    assert p.pin("s1", [1] * 1000, "search", t0) == 0.5 and p.within_ttl("s1", t0 + 0.4)
    assert p.expire(t0 + 1.0, lambda _: 1) == [] and p.expire(t0 + 1.0, lambda _: 0) == ["s1"]
    p.note_arrival("s1", t0 + 2.0)   # gap 2.0 lands in both the tool's and the global CDF
    assert p.gaps["search"][-1] == 2.0 and p.gaps["*"][-1] == 4.0 and 2.0 in p.gaps["*"]
    print("continuum selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Continuum TTL baseline server")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel")
    ap.add_argument("--ttl-default", type=float, default=5.0, metavar="S",
                    help="TTL until K gap records exist (Continuum's T_default)")
    ap.add_argument("--no-ttl-sched", action="store_true",
                    help="pin only; do not prioritise requests that return within TTL")
    ap.add_argument("--sched", choices=["fcfs", "lpm"], default="fcfs",
                    help="engine waiting-queue order: fcfs + TTL priority tier, or longest-prefix-match")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        asyncio.run(main(a.model, a.port, a.kv, a.host, a.hicache_io, a.ttl_default, not a.no_ttl_sched, a.sched))
