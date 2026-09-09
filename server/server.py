"""
Batched controller: workflow-aware serving of byllm agents over one SGLang engine.
Predicted calls become promotion jobs and drive the engine's eviction order
(model.promote.kv_priority); see server.kv_planner.

python -m server.server [MODEL] [--lru] [--kv N] [--host GB] [--sched fcfs|lpm]
"""
import argparse
import asyncio
import json
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from decompiler.parser import _lcp, chosen_candidates, decompose, is_continuation, node_type, relayout_body
import decompiler.primitives as primitives
from decompiler.primitives import (ByLLMCallsite, Callsite, CallObservation, PredictedCall, Program,
                                   VisitByCallsite)
from model.device_profiler import profile_device
from model.model import PRIORITY_CONT, PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest
from server.kv_planner import Job, KVPlanner

P_MIN = 0.02           # plan only steps predicted at least this likely
T_BASE = 120.0         # idle seconds before a session is finalized
T_SHORT = 15.0         # idle timeout once the program says the session is over
END_PROB_SHORT = 0.5   # end_prob above this switches to T_SHORT
SWEEP_S = 5.0          # idle sweeper period
REBUILD_AT = 8         # rebuild when n_sessions reaches this, then every doubling
MAX_TOKENS = 4096      # decode cap when the request does not set one
MAX_BATCH = 32

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class LiveSession:
    id: str                                       # Session ID, from the agent
    program: Optional[Program] = None            # Bound on the first decomposed call
    walked: List[str] = field(default_factory=list)               # Callsite keys, ReAct turns folded
    pending: List[CallObservation] = field(default_factory=list)  # Closed calls awaiting update_graph
    open_obs: Optional[CallObservation] = None   # The call currently being folded
    last_raw: Optional[dict] = None              # Previous request body, for is_continuation
    last_seen: float = 0.0                       # monotonic, for the idle sweeper
    inflight: int = 0                            # HTTP requests being served right now
    epoch: int = 0                               # Bumps when walked grows; voids queued jobs
    predicted: Optional[str] = None              # Last top-1 prediction, for hit logging
    tainted: bool = False                        # Joined mid-program: serve it, keep stats clean
    plan_anchor: float = 0.0                     # monotonic t the gap predictions count from

class Controller:
    def __init__(self, model: str, server: HttpServer, plan: bool = True, enable_relayout: bool = True ,**engine_kwargs):
        self.plan = plan               # off: plain serving over the prefix cache (the LRU baseline)
        self.engine: Engine = Engine(model, **engine_kwargs)
        self.sessions: Dict[str, LiveSession] = {}    # session id -> live state
        self.programs: Dict[str, Program] = {}        # entry callsite key -> Program
        self.server = server
        self.pool = server.pool
        self._prefix_tok: Dict[str, Tuple[str, List[int]]] = {}  # site key -> (fixed_head, token ids), no re-tokenization
        self._plan_tok: "OrderedDict[Tuple[str, bool, str], Optional[List[int]]]" = OrderedDict()
        self._enable_relayout:bool = enable_relayout # Enable prompt re-layout for better KV reuse. 
        self.PLAN_TOK_CACHE = 1024
        self.planner: Optional[KVPlanner] = KVPlanner(self.engine) if plan else None

    # ------------------------------------------------------------------ lifecycle
    async def start_serving(self) -> None:
        await self.engine.warmup()
        profile = f"model/{self.engine.name.replace('/', '--')}.profile.json"
        if not self.engine.load_profile(profile):
            await profile_device(self.engine)
            self.engine.save_profile(profile)
        asyncio.create_task(self._sweep())
        if self.planner is not None:
            asyncio.create_task(self.planner.run())

        while True:
            req = await self.pool.get()          # blocks; yields to the event loop
            batch: List[PendingRequest] = []
            while True:
                if req.kind == "close":
                    req.reply(self._finalize(req.session))
                else:
                    batch.append(req)
                if len(batch) >= MAX_BATCH:
                    break
                try:
                    req = self.pool.get_nowait()  # drain whatever else arrived
                except asyncio.QueueEmpty:
                    break
            if batch:
                await self._step(batch)           # classification only; returns immediately

    async def _sweep(self) -> None:
        """Finalize sessions that have gone quiet; """
        while True:
            await asyncio.sleep(SWEEP_S)
            now = time.monotonic()
            for sid, sess in list(self.sessions.items()):
                if sess.inflight > 0:
                    continue
                t = T_BASE
                if sess.program is not None and sess.program.end_prob(sess.walked) > END_PROB_SHORT:
                    t = T_SHORT # shorter timeout once probably done
                if now - sess.last_seen > t:
                    self._finalize(sid)

    # ------------------------------------------------------------------ request path
    async def _step(self, batch: List[PendingRequest]) -> None:
        """Admit one drained batch"""
        for req in sorted(batch, key=lambda r: r.t_arrive):
            sess = self.sessions.get(req.session)
            if sess is None:
                sess = self.sessions[req.session] = LiveSession(id=req.session)
            sess.inflight += 1
            sess.last_seen = time.monotonic()
            try:
                t_cls = time.monotonic()
                self._relayout(sess, req.body)   # a ReAct turn re-sends the call message: same order
                cont = (sess.open_obs is not None and sess.last_raw is not None
                        and is_continuation(sess.last_raw, req.body))
                if cont:
                    ob = sess.open_obs
                    ob.n_turns += 1  # type: ignore[union-attr]
                    ob.tool_gaps.append(max(0.0, req.t_arrive - ob.t_done))  # type: ignore[union-attr]
                else:
                    ob = self._advance(sess, req)
                ob.t_classify = t_cls                                  # type: ignore[union-attr]
                ob.advance_ms = (time.monotonic() - t_cls) * 1000      # type: ignore[union-attr]
            except Exception as e:
                sess.inflight -= 1
                req.fail(e)
                continue
            asyncio.create_task(self._generate(sess, req, ob, cont))  # type: ignore[arg-type]
        if self.planner is not None:
            self.planner.wake()

    async def _generate(self, sess: LiveSession, req: PendingRequest,
                        ob: CallObservation, cont: bool) -> None:
        """One real request through the engine, then a planning refresh: its reply
        opens the gap window — the agent runs its own code now, the engine is free."""
        try:
            t_pre = time.perf_counter()
            body = req.body
            prompt = self.engine.render(body["messages"], tools=body.get("tools"))
            sp = {"temperature": 0.7 if body.get("temperature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            rid = f"{sess.id}-{sess.epoch}t{ob.n_turns}-{uuid.uuid4().hex[:8]}"
            ids = self.engine.tokenize(prompt)
            t0 = time.perf_counter()
            text = await self.engine.generate(prompt, rid, sp, ids=ids,
                                              priority=PRIORITY_CONT if cont else PRIORITY_REAL)
            ob.engine_s += time.perf_counter() - t0
            ob.t_done = time.monotonic()
            ob.response = text
            sess.last_raw = body
            req.reply(self._answer(body, text))
        except Exception as e:
            req.fail(e)
            return
        finally:
            sess.inflight -= 1
            sess.last_seen = time.monotonic()
        if self.planner is None or sess.id not in self.sessions:
            return
        t_post = time.perf_counter()
        prog = sess.program
        site = prog.sites.get(ob.key) if prog is not None else None
        if site is not None:
            # Past the head the flow rules can rebuild (header | session constants |
            # history so far) the prompt is one-off: the fresh bindings and the reply.
            # Only that tail is demoted; the head stays in the normal band.
            head = len(self._prefix_tokens(site))
            if prog is not None:
                text, _ = prog.resolve_user(ob.key, list(sess.pending))
                cut = self._planned_tokens(site, text, False, getattr(site, "tool_schema", None)) if text else None
                if cut is not None and len(cut) > head:
                    head = len(cut)
            self.planner.note_served(sess.id, ids, head)
        sess.plan_anchor = ob.t_done         # the anchor every gap sample counts from
        self._plan_session(sess)             # replace: the freshest view of the future
        if prog is not None and sess.walked and isinstance(prog.sites.get(sess.walked[-1]), VisitByCallsite):
            self._route_followup(sess)       # extend: the reply names the branch outright
        self.planner.wake()
        # controller-side cost of this call: queue wait before classification, classify+plan
        # at arrival (_advance), render+tokenize, and the post-reply re-plan
        print(f"[ctl] {rid} queue_ms={(ob.t_classify - req.t_arrive) * 1000:.1f} " #type: ignore[union-attr]
              f"advance_ms={ob.advance_ms:.1f} pre_ms={(t0 - t_pre) * 1000:.1f} " #type: ignore[union-attr]
              f"post_ms={(time.perf_counter() - t_post) * 1000:.1f}", flush=True)

    def _advance(self, sess: LiveSession, req: PendingRequest) -> CallObservation:
        """A new call: close the open one, identify the callsite, extend walked."""
        if sess.open_obs is not None:
            sess.pending.append(sess.open_obs)
        site, extras = decompose(req.body)
        prog = self._bind(sess, site)
        site = prog.add_callsite(site, extras.bindings)
        if self._enable_relayout and isinstance(site, ByLLMCallsite) and site.layout and relayout_body(req.body, site.layout, site.header_last):
            _, extras = decompose(req.body)   # bindings and user_text as they go on the wire
            
        
        hit = "" if sess.predicted is None else f" predicted={'hit' if sess.predicted == site.key else 'miss'}"
        print(f"[call] {sess.id} #{len(sess.walked)} {site.label}{hit}", flush=True)
        sess.walked.append(site.key)
        sess.epoch += 1
        # Anchoring at arrival is early by this call's own duration — deliberately
        # conservative; the completion re-plan tightens the deadlines.
        sess.plan_anchor = req.t_arrive
        sess.open_obs = CallObservation(key=site.key, t_arrive=req.t_arrive, t_done=req.t_arrive,
                                        candidates=extras.candidates, bindings=extras.bindings,
                                        self_view=extras.self_view, walker=extras.walker,
                                        here=extras.here, cand_block=extras.cand_block,
                                        user_text=extras.user_text)
        if self.planner is not None:
            self.planner.note_arrival(sess.id, site.key, req.t_arrive)
            self.planner.void_session(sess.id, sess.epoch)
            self._plan_session(sess)  # overlap the successors' work with this call's decode
        return sess.open_obs

    def _relayout(self, sess: LiveSession, body: dict) -> None:
        """Re-emit the request's call message in the open site's frozen binding order."""
        if not self._enable_relayout or sess.open_obs is None or sess.program is None:
            return
        site = sess.program.sites.get(sess.open_obs.key)
        if isinstance(site, ByLLMCallsite) and site.layout:
            relayout_body(body, site.layout, site.header_last)

    def _bind(self, sess: LiveSession, site: Callsite) -> Program:
        """Bind the session to its Program by content, entry callsite identifies the program"""
        if sess.program is None:
            prog = self.programs.get(site.key)
            if prog is None:
                mid = next((p for p in self.programs.values() if site.key in p.sites), None)
                if mid is not None:
                    prog, sess.tainted = mid, True
                    print(f"[program] {sess.id} joined mid-program; stats off", flush=True)
                else:
                    prog = self.programs[site.key] = Program(entry=site.key)
                    print(f"[program] new: {site.label}", flush=True)
            sess.program = prog
        return sess.program

    def _answer(self, body: dict, text: str) -> Any:
        """Native-tools responses must carry structured tool_calls; Qwen emits them as
        <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks in the text.
        (byllm's InterceptorLLM uses the text tool protocol, so this stays inert there.)"""
        if not body.get("tools") or "<tool_call>" not in text:
            return text
        calls = []
        for m in _TOOL_CALL.finditer(text):
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                          "function": {"name": d.get("name", ""),
                                       "arguments": json.dumps(d.get("arguments", {}))}})
        if not calls:
            return text
        content = _TOOL_CALL.sub("", text).strip()
        return {"role": "assistant", "content": content or None, "tool_calls": calls}

    # ------------------------------------------------------------------ planning
    def _session_obs(self, sess: LiveSession) -> List[CallObservation]:
        """The completed observations value-flow rules may draw from."""
        obs = list(sess.pending)
        if sess.open_obs is not None and sess.open_obs.response:
            obs.append(sess.open_obs)
        return obs

    def _plan_session(self, sess: LiveSession) -> None:
        """One planning refresh: fan-out prediction -> deadline-stamped KV jobs,
        replacing whatever the session had queued (its branches may have collapsed)."""
        prog, planner = sess.program, self.planner
        if prog is None or planner is None:
            return
        calls = prog.predict_tree(sess.walked, p_min=P_MIN)
        sess.predicted = calls[0].key if calls else None
        anchor = sess.plan_anchor or time.monotonic()
        jobs = []
        for c in calls:
            jobs.extend(self._plan_call(sess, c, anchor))
        planner.submit(sess.id, sess.epoch, jobs)

    def _plan_call(self, sess: LiveSession, c: PredictedCall, anchor: float) -> List[Job]:
        """ For each call we expect to happen, start preparing it early:
        - Preload the entire reconstructed call if possible. Otherwise, preload the longest prefix we can reconstruct—or, as a last resort, the fixed/static prefix.
        """
        prog = sess.program
        site = prog.sites.get(c.key) if prog else None
        if prog is None or site is None:
            return []
        sysmsg = {"role": "system", "content": site.system_prompt}
        tools = getattr(site, "tool_schema", None)
        obs = self._session_obs(sess)
        text, resolved = prog.resolve_user(c.key, obs)
        proto = prog.proto.get(c.key, {})
        complete = resolved and proto.get("ok", 0) >= 1
        now = time.monotonic()
        deadline = max(now, anchor + c.t50 - KVPlanner.SAFETY_K * (c.t90 - c.t50))
        t90 = max(deadline, anchor + c.t90)
        jobs: List[Job] = []
        if complete:
            toks = self._planned_tokens(site, text, True, tools)
        else:
            cut = self._planned_tokens(site, text, False, tools) if text else None
            static = self._prefix_tokens(site)
            toks = cut if cut is not None and len(cut) > len(static) else static
        if len(toks) < 16:  # shorter than a cache block: nothing to gain  #type: ignore
            return jobs
        unc, host = self.engine.cost(toks) #type: ignore[union-attr]
        # promote: a host-resident prefix to load back ahead of the call; hold: the
        # rest must be computed by the call itself (or is already resident), the job
        # only carries eviction protection. Same predicate as before: the kind is part
        # of the job key, so it decides when a re-plan replaces a queued job.
        kind = "promote" if unc <= self.engine._stride and host > 0 else "hold"
        resident = unc + host < self.engine._stride
        jobs.append(Job(key=f"{sess.id}|{kind}|{c.key}", sid=sess.id, epoch=sess.epoch,
                        site=c.key, kind=kind, toks=toks, p=c.p, #type: ignore[union-attr]
                        value=c.p * (unc + 0.8 * host), deadline=deadline, t90=t90,
                        host=host, done_upto=len(toks) if resident else 0, #type: ignore
                        state="done" if resident else "queued"))
        return jobs

    # ------------------------------------------------------------------ routing foresight
    def _route_followup(self, sess: LiveSession) -> None:
        """A routing reply that actually landed names the nodes the walker visits
        next: plan their calls with near-immediate deadlines instead of hedging on
        branch statistics."""
        prog, ob = sess.program, sess.open_obs
        if prog is None or ob is None:
            return
        jobs = []
        for _, node in chosen_candidates(ob):
            jobs.extend(self._succ_job(sess, sess.walked[-1], node, 1.0, 0.0, 0.0))
        if jobs:
            sess.predicted = jobs[0].site  # the reply names the next call outright
        self._submit_extra(sess, jobs)

    def _succ_job(self, sess: LiveSession, from_key: str, node: str,
                  p: float, t50: float, t90: float) -> List[Job]:
        """The call that history says runs on `node`'s type, arriving one gap after
        the routing call whose own arrival is (t50, t90) past the anchor."""
        prog = sess.program
        succ = prog.type_succ.get(node_type(node)) if prog else None
        if prog is None or not succ:
            return []
        key, cnt = succ.most_common(1)[0]
        site = prog.sites.get(from_key)
        d50 = site.exec_stats.duration_q(0.5) if site is not None else 0.0
        d90 = site.exec_stats.duration_q(0.9) if site is not None else 0.0
        c = PredictedCall(key=key, p=p * cnt / max(1, sum(succ.values())),
                          t50=t50 + d50 + prog._gap_q("", "", from_key, key, 0.5),
                          t90=t90 + d90 + prog._gap_q("", "", from_key, key, 0.9))
        return self._plan_call(sess, c, sess.plan_anchor or time.monotonic())

    def _submit_extra(self, sess: LiveSession, jobs: List[Job]) -> None:
        if jobs and self.planner is not None:
            self.planner.submit(sess.id, sess.epoch, jobs, extend=True)
            self.planner.wake()

    def _planned_tokens(self, site: Callsite, text: str, complete: bool,
                        tools: Optional[List[Dict[str, Any]]]) -> Optional[List[int]]:
        """Token ids of the prompt a predicted call would send: the whole rendered
        prompt when `text` is the complete user message, else the rendered prefix up
        to the end of `text` minus a possibly split last token (None if the render
        does not contain the text verbatim). Memoized; see _plan_tok."""
        key = (site.key, complete, text)
        hit = self._plan_tok.get(key, ...)
        if hit is not ...:
            self._plan_tok.move_to_end(key)
            return hit
        sysmsg = {"role": "system", "content": site.system_prompt}
        rendered = self.engine.render([sysmsg, {"role": "user", "content": text}], tools=tools)
        if complete:
            toks: Optional[List[int]] = self.engine.tokenize(rendered)
        else:
            i = rendered.rfind(text)
            toks = self.engine.tokenize(rendered[:i + len(text)])[:-1] if i >= 0 else None
        self._plan_tok[key] = toks
        if len(self._plan_tok) > self.PLAN_TOK_CACHE:
            self._plan_tok.popitem(last=False)
        return toks

    def _prefix_tokens(self, site: Callsite) -> List[int]:
        """Token ids of the callsite's static prompt head, rendered exactly as a real request would be. """
        stable = site.fixed_head
        cached = self._prefix_tok.get(site.key)
        if cached is not None and cached[0] == stable:
            return cached[1]
        tools = getattr(site, "tool_schema", None)
        sysmsg = {"role": "system", "content": site.system_prompt}
        if stable:
            text = self.engine.render([sysmsg, {"role": "user", "content": stable}], tools=tools)
            i = text.rfind(stable)
            prefix = text[:i + len(stable)] if i >= 0 else ""
        else:
            # The template puts tool schemas after the system content, so the shared
            # byte prefix is found by diffing two renders rather than by searching.
            a = self.engine.render([sysmsg], tools=tools)
            b = self.engine.render([sysmsg, {"role": "user", "content": "\x00"}], tools=tools)
            prefix = _lcp(a, b)
        toks = self.engine.tokenize(prefix)[:-1] if prefix else []
        self._prefix_tok[site.key] = (stable, toks)
        return toks

    # ------------------------------------------------------------------ session end
    def _finalize(self, sid: str) -> bool:
        """Session end: fold the session's observations into its Program and drop it."""
        sess = self.sessions.pop(sid, None)
        if sess is None:
            return False
        if self.planner is not None:
            self.planner.drop_session(sid)
        if sess.open_obs is not None:
            sess.pending.append(sess.open_obs)
        prog = sess.program
        if prog is not None and sess.walked and not sess.tainted:
            for k in prog.update_graph(sess.walked, sess.pending):
                print(f"[layout] {prog.sites[k].label} order={prog.sites[k].layout} header_last={prog.sites[k].header_last}", flush=True)  # type: ignore[union-attr]
            n = prog.n_sessions
            if n >= REBUILD_AT and n & (n - 1) == 0:
                prog.rebuild()
                print(f"[rebuild] n_sessions={n} nodes={len(prog.nodes)}", flush=True)
        print(f"[close] {sid} calls={len(sess.walked)}" + (" tainted" if sess.tainted else ""),
              flush=True)
        return True


async def main(model: str = "Qwen/Qwen3-8B", port: int = 8964, plan: bool = True,
               kv_tokens: Optional[int] = None,
               host_gb: Optional[int] = None, hicache_io: Optional[str] = None,
               enable_relayout: bool = True, sched: str = "fcfs", risk_aging_s: float = 10.0) -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 16384,
                              "radix_eviction_policy": "priority" if plan else "lru"}
    if kv_tokens:
        kwargs["max_total_tokens"] = kv_tokens   # real device pool cap (sglang server arg)
    if host_gb is not None:
        kwargs["host_cache_gb"] = host_gb        # host KV tier size (model.model.HOST_KV_GB default)
    if hicache_io:
        kwargs["hicache_io_backend"] = hicache_io  # host<->device copy path; model.model defaults to "direct"
    if sched == "lpm":   # engine re-sorts the waiting queue by matched prefix length every step;
        kwargs["schedule_policy"] = "lpm"          # sglang allows request priorities only with fcfs/lof,
        kwargs["enable_priority_scheduling"] = False   # so PRIORITY_CONT becomes a no-op
    elif sched == "risk":   # fork policy: cached-tokens-at-risk first, FCFS otherwise, aging bounds the reorder
        kwargs["schedule_policy"] = "cache-risk"
        kwargs["enable_priority_scheduling"] = False
        kwargs["cache_risk_aging_s"] = risk_aging_s
    ctrl = Controller(model, server, plan=plan, enable_relayout=enable_relayout, **kwargs)
    await ctrl.start_serving()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Start the server")
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-8B")
    ap.add_argument("--lru", action="store_true",
                    help="baseline: no planner, sglang's own LRU eviction (decompiler and re-layout stay on)")
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--no-header-last", action="store_true",
                    help="re-layout reorders bindings only; never move the callsite header behind the values")
    ap.add_argument("--no-relayout", action="store_true",
                    help="Disable prompt re-layout for better KV reuse (default: enabled)")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel",
                    help="HiCache host<->device copy backend (default: kernel)")
    ap.add_argument("--sched", choices=["fcfs", "lpm", "risk"], default="fcfs",
                    help="engine waiting-queue order: fcfs + request priorities, longest-prefix-match, "
                         "or cache-risk (cached tokens at risk first, FCFS otherwise)")
    ap.add_argument("--risk-aging-s", type=float, default=10.0, metavar="S",
                    help="--sched risk: an uncached request overtakes a fully cached one after waiting S s longer (0 = FCFS, negative = never age)")
    a = ap.parse_args()
    if a.no_header_last:
        primitives.HEADER_LAST = False
    asyncio.run(main(a.model, plan=not a.lru, kv_tokens=a.kv,
                     host_gb=a.host, hicache_io=a.hicache_io, enable_relayout=not a.no_relayout,
                     sched=a.sched, risk_aging_s=a.risk_aging_s))
