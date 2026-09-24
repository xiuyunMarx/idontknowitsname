"""KVFlow baseline (arXiv 2507.07400): workflow-aware KV cache management for
LLM multi-agent workflows, run on the same engine, host tier and prompts as the
other arms.

KVFlow's two mechanisms, and how each is realised here:

  steps-to-execution   Every agent node of the workflow graph carries the number
        of workflow steps until it executes again. KV cache entries are evicted
        in decreasing order of the steps-to-execution of the agent that owns them
        rather than by recency, and entries of finished workflows go first.
        Here the agent step graph is the program's static callsite graph, the
        same registration the guard server receives. For a live session whose
        latest call is at callsite `cur`, the steps-to-execution of callsite s is
        the length of the shortest static path cur -> s (1 = the next call). Every
        prompt the session has been served is owned by its callsite and is
        protected with a band that falls by one per step (`kv_priority` sums the
        bands of every session covering a node, so a prefix shared by many
        sessions outranks a private one). Prompts of callsites the session can no
        longer reach drop to the transient band; prompts of ended sessions to the
        retired band; both are evicted before anything owned by a live agent.
        The static head of a prompt (the agent's fixed prefix) is never retired:
        in KVFlow it belongs to the agent, not to the workflow instance.
  prefetch   The KV of an agent whose steps-to-execution is 1 is copied from CPU
        to GPU in the background while the current step runs, so the next call
        starts from a resident prefix. Here the controller predicts each session's
        direct successors and reconstructs the prompt prefix they will send (as it
        does for the planner); the ones one step away with a host-resident cached
        prefix are promoted host -> device by the engine's own background loader.
        No deadline and no value estimate: the paper prefetches whatever is one
        step away, as space allows.


python -m server.KVFlow_server [MODEL] [--kv N] [--host GB] [--relayout] [--no-prefetch]
"""
import argparse
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from server.http_server import HttpServer
from server.kv_planner import Job
from server.server import GRAMMAR_BACKEND, Controller

NO_ADMISSION = 1 << 30   # served-head length that admits the whole prompt to the host tier


class KVFlowController(Controller):
    """The guard controller with host admission switched off: KVFlow writes every
    served prompt through to the host tier, as stock HiCache does."""

    def _served_head(self, sess, inst) -> int:
        return NO_ADMISSION


class KVFlowPolicy:
    """Drop-in for the guard controller's planner slot: same intake methods, KVFlow's rules."""

    TICK_S = 0.015              # heartbeat between wake events
    K_MAX = 100                 # band of a prompt whose agent runs next; one less per further step
    MAX_ATTEMPTS = 3            # failed promotion RPCs before a prefetch is dropped
    NO_ROOM_COOLDOWN_CYCLES = 5 # cycles every prefetch sits out after one is refused for space

    promote_enabled = True      # --no-prefetch: steps-to-execution eviction only

    def __init__(self, engine) -> None:
        self.engine = engine
        self.ctrl: Optional[Controller] = None      # set by main(): the sessions and their programs
        self._served: Dict[str, List[Tuple[str, List[int], int]]] = {}   # sid -> (callsite repr, prompt ids, static head)
        self._site: Dict[str, str] = {}             # sid -> callsite repr of the call being served
        self._jobs: Dict[str, Dict[str, Job]] = {}  # sid -> job key -> predicted-successor prefix
        self._retire: List[Tuple[List[int], int]] = []
        self._wake = asyncio.Event()
        self._dirty = False                         # the priority map no longer mirrors the sessions

    # ------------------------------------------------------------------ intake (controller interface)
    def note_arrival(self, sid: str, site: str, t_arrive: float) -> None:
        self._site[sid] = site

    def note_served(self, sid: str, ids: List[int], fixed_len: int, *, static_len: int) -> None:
        """A prompt landed: it is owned by the callsite that sent it."""
        self._served.setdefault(sid, []).append((self._site.get(sid, ""), list(ids), static_len))
        self._dirty = True

    def invalidate(self, sid: str, ids: List[int], keep_len: int) -> None:
        pass                                        # flow-rule demotion is the planner's idea, not KVFlow's

    def void_session(self, sid: str, epoch: int) -> None:
        for j in self._jobs.get(sid, {}).values():
            if j.state in ("queued", "running") and j.epoch < epoch:
                j.state = "void"

    def submit(self, sid: str, epoch: int, jobs: List[Job], extend: bool = False) -> None:
        """The controller's predicted successors and their reconstructed prompt prefixes."""
        held = self._jobs.setdefault(sid, {})
        keep = set()
        for j in jobs:
            keep.add(j.key)
            prev = held.get(j.key)
            if prev is not None and prev.state in ("queued", "running") and prev.epoch == j.epoch:
                if len(j.toks) > len(prev.toks):
                    prev.toks = j.toks
                    prev.attempts = 0
                continue
            if prev is not None and prev.state == "running":
                prev.state = "void"
            held[j.key] = j
        if not extend:
            for k, prev in held.items():
                if k not in keep and prev.state in ("queued", "running"):
                    prev.state = "void"
        self._dirty = True

    def retire_session(self, sid: str) -> None:
        """The workflow is over: its prompts past their static heads go to the retired band."""
        for _, ids, static_len in self._served.pop(sid, []):
            self._retire.append((ids, static_len))
        self._dirty = True

    def drop_session(self, sid: str) -> None:
        self.retire_session(sid)
        self._jobs.pop(sid, None)
        self._site.pop(sid, None)

    def wake(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------------ steps-to-execution
    @staticmethod
    def _distances(prog, cur) -> Dict[Any, int]:
        """Shortest static path length cur -> site, 1 = the next call; cur itself only
        via a cycle. Sites without an entry cannot execute again from here."""
        dist: Dict[Any, int] = {}
        frontier = [cur]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for k in frontier:
                for e in prog.successors(k):
                    if e.dst not in dist:
                        dist[e.dst] = d
                        nxt.append(e.dst)
            frontier = nxt
        return dist

    def _session_view(self, sid: str):
        """(program, current callsite, callsite repr -> key) of a live session, or None."""
        sess = self.ctrl.sessions.get(sid) if self.ctrl is not None else None
        if sess is None or sess.program is None:
            return None
        inst = sess.open or (sess.calls[-1] if sess.calls else None)
        if inst is None:
            return None
        prog = sess.program
        return prog, inst.key, {repr(k): k for k in prog.sites}

    async def push_priorities(self) -> None:
        """Mirror the sessions into the engine's eviction bands."""
        if not self._dirty:
            return
        self._dirty = False
        protect: List[Tuple[List[int], int]] = []
        demote: List[Tuple[List[int], int]] = []
        retire, self._retire = self._retire, []
        for sid, served in self._served.items():
            view = self._session_view(sid)
            dist: Dict[Any, int] = {}
            by_repr: Dict[str, Any] = {}
            if view is not None:
                prog, cur, by_repr = view
                dist = self._distances(prog, cur)
            for site, ids, static_len in served:
                key = by_repr.get(site)
                d = dist.get(key) if key is not None else None
                if d is None:
                    demote.append((ids, static_len))   # its agent cannot run again from here
                else:
                    protect.append((ids, max(1, self.K_MAX - d)))
        await self.engine.set_kv_priority(demote, protect, f"kvflow-{uuid.uuid4().hex[:8]}", retire=retire)

    # ------------------------------------------------------------------ prefetch
    def _pick(self) -> Optional[Job]:
        """A predicted call one step away whose cached prefix is partly on the host."""
        if not self.promote_enabled:
            return None
        best: Optional[Job] = None
        for sid, held in self._jobs.items():
            view = self._session_view(sid)
            if view is None:
                continue
            prog, cur, by_repr = view
            dist = self._distances(prog, cur)
            for j in held.values():
                if j.state != "queued":
                    continue
                if j.cooldown_cycles > 0:          # refused for space recently; one cycle per pick
                    j.cooldown_cycles -= 1
                    continue
                key = by_repr.get(j.site)
                if key is None or dist.get(key) != 1:
                    continue
                unc, host = self.engine.cost(j.toks)
                if host <= 0 or len(j.toks) - unc <= j.done_upto:
                    continue                       # nothing on the host beyond the device frontier
                if best is None or j.p > best.p:
                    best = j
        return best

    async def _execute(self, job: Job) -> None:
        if job.state != "queued":
            return
        job.state = "running"
        rid = f"kvflow-prefetch-{uuid.uuid4().hex[:8]}"
        end = len(job.toks) - self.engine.cost(job.toks)[0]   # the whole cached frontier at once
        ok = await self.engine.promote(job.toks[:end], rid)
        if job.state == "void":
            return
        if ok is None:                                 # deferred behind real work
            job.state = "queued"
            if self.engine.promote_no_room:
                for held in self._jobs.values():
                    for j in held.values():
                        if j.state == "queued":
                            j.cooldown_cycles = self.NO_ROOM_COOLDOWN_CYCLES
            return
        if not ok:
            job.attempts += 1
            job.state = "void" if job.attempts >= self.MAX_ATTEMPTS else "queued"
            return
        job.done_upto = end
        job.attempts = 0
        job.state = "done" if end >= len(job.toks) else "queued"

    # ------------------------------------------------------------------ the loop
    async def run(self) -> None:
        """Single-flight: one priority push and at most one prefetch RPC per tick."""
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.TICK_S)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            await self.push_priorities()
            job = self._pick()
            if job is not None:
                await self._execute(job)


async def main(model: str, port: int, kv_tokens: Optional[int], host_gb: Optional[int],
               hicache_io: Optional[str], relayout: bool = False, prefetch: bool = True,
               engine_log: Optional[str] = None) -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 32768,
                              "radix_eviction_policy": "priority",   # steps-to-execution bands need it
                              "grammar_backend": GRAMMAR_BACKEND}
    if kv_tokens:
        kwargs["max_total_tokens"] = kv_tokens
    if host_gb is not None:
        kwargs["host_cache_gb"] = host_gb
    if hicache_io:
        kwargs["hicache_io_backend"] = hicache_io
    if engine_log:
        kwargs["log_level"] = engine_log
    # The guard controller without its planner: registration, callsite identification,
    # successor prediction and (optionally) re-layout; KVFlow's policy takes the planner slot.
    ctrl = KVFlowController(model, server, plan=False, enable_relayout=relayout, **kwargs)
    policy = KVFlowPolicy(ctrl.engine)
    policy.ctrl = ctrl
    policy.promote_enabled = prefetch
    ctrl.planner = policy   # type: ignore[assignment]
    print(f"[kvflow] steps-to-execution eviction, prefetch={'on' if prefetch else 'off'}, "
          f"relayout={'on' if relayout else 'off'}, no host admission", flush=True)
    await ctrl.start_serving()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KVFlow baseline server (steps-to-execution eviction + one-step prefetch)")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel")
    ap.add_argument("--relayout", action="store_true",
                    help="serve the re-laid prompts of `ours` (default: the prompts as the agents send them)")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="steps-to-execution eviction only, no host->device prefetch of the next agent's KV")
    ap.add_argument("--sched", choices=["fcfs"], default="fcfs", help="engine queue order (accepted for the launcher)")
    ap.add_argument("--engine-log", choices=["info", "warning", "error"], default=None)
    a = ap.parse_args()
    asyncio.run(main(a.model, a.port, a.kv, a.host, a.hicache_io, relayout=a.relayout,
                     prefetch=not a.no_prefetch, engine_log=a.engine_log))
