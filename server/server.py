"""Batched controller: Belady-with-repair serving of byllm agents over one SGLang engine.

Requests drain from the pool in batches; each batch is classified against its
sessions (ReAct continuations fold into the open call, new calls advance the
walk) and dispatched as concurrent generation tasks — the engine batches them
internally, continuations outranking fresh calls outranking speculation.

Foresight no longer fires eagerly. Every completed call refreshes a fan-out
prediction of the session's future (`Program.predict_tree`: callsites with
probabilities and arrival-time quantiles); each predicted call becomes a KV job
— probe a fully-rebuilt routing prompt, prefill a fully-rebuilt call whole,
else the longest reconstructable or static prefix — stamped with a deadline and
priced by the residency ledger (create vs promote from the host tier). The
KVPlanner runs them just-in-time, earliest deadline first, in the engine's
idle/gap windows, and under memory pressure warm-touches the cached prefixes
worth keeping ahead of their next use. An eviction repaired before its deadline
is free; the objective is min Σ p(call) × exposed prefill.

    python -m server.server [MODEL] [--no-spec]
"""
import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from math import exp
from typing import Any, Dict, List, Optional, Tuple

from decompiler.parser import _strip_hint, decompose, is_continuation, parse_candidate_line
from decompiler.primitives import (Callsite, CallObservation, PredictedCall, Program,
                                   VisitByCallsite, _lcp, chosen_candidates, node_type)
from model.model import PRIORITY_CONT, PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest
from server.kv_planner import Job, KVPlanner

P_MIN = 0.02           # plan only steps predicted at least this likely
P_ROUTE = 0.3          # act on probed routing choices at least this likely
T_BASE = 120.0         # idle seconds before a session is finalized
T_SHORT = 15.0         # idle timeout once the program says the session is over
END_PROB_SHORT = 0.7   # end_prob above this switches to T_SHORT
SWEEP_S = 5.0          # idle sweeper period
REBUILD_AT = 8         # rebuild when n_sessions reaches this, then every doubling
MAX_TOKENS = 4096      # decode cap when the request does not set one
MAX_BATCH = 32

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class LiveSession:
    id: str                                      # Session ID, from the agent
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
    def __init__(self, model: str, server: HttpServer, speculate: bool = True, **engine_kwargs):
        self.speculate = speculate     # off: plain serving over the prefix cache (the baseline)
        self.engine: Engine = Engine(model, **engine_kwargs)
        self.sessions: Dict[str, LiveSession] = {}    # session id -> live state
        self.programs: Dict[str, Program] = {}        # entry callsite key -> Program
        self.server = server
        self.pool = server.pool
        self._prefix_tok: Dict[str, Tuple[str, List[int]]] = {}  # site key -> (stable_prefix, token ids)
        self.planner: Optional[KVPlanner] = KVPlanner(self.engine) if speculate else None

    # ------------------------------------------------------------------ lifecycle
    async def start_serving(self) -> None:
        await self.engine.warmup()
        profile = f"model/{self.engine.name.replace('/', '--')}.profile.json"
        if not self.engine.load_profile(profile):
            await self.engine._profile()
            await self.engine._profile_oneshot()
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
        """Finalize sessions that have gone quiet; the timeout tightens once the
        program says the session is probably over."""
        while True:
            await asyncio.sleep(SWEEP_S)
            now = time.monotonic()
            for sid, sess in list(self.sessions.items()):
                if sess.inflight > 0:
                    continue
                t = T_BASE
                if sess.program is not None and sess.program.end_prob(sess.walked) > END_PROB_SHORT:
                    t = T_SHORT
                if now - sess.last_seen > t:
                    self._finalize(sid)

    # ------------------------------------------------------------------ request path
    async def _step(self, batch: List[PendingRequest]) -> None:
        """Admit one drained batch: classify every request against its session and
        dispatch the generations as concurrent tasks — the engine batches them.
        No engine await happens here, so the drain loop never stalls."""
        for req in sorted(batch, key=lambda r: r.t_arrive):
            sess = self.sessions.get(req.session)
            if sess is None:
                sess = self.sessions[req.session] = LiveSession(id=req.session)
            sess.inflight += 1
            sess.last_seen = time.monotonic()
            try:
                cont = (sess.open_obs is not None and sess.last_raw is not None
                        and is_continuation(sess.last_raw, req.body))
                if cont:
                    ob = sess.open_obs
                    ob.n_turns += 1  # type: ignore[union-attr]
                    ob.tool_gaps.append(max(0.0, req.t_arrive - ob.t_done))  # type: ignore[union-attr]
                else:
                    ob = self._advance(sess, req)
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
            body = req.body
            prompt = self.engine.render(body["messages"], tools=body.get("tools"))
            sp = {"temperature": 0.7 if body.get("temporature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            rid = f"{sess.id}-{sess.epoch}t{ob.n_turns}-{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()
            text = await self.engine.generate(prompt, rid, sp,
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
        sess.plan_anchor = ob.t_done         # the anchor every gap sample counts from
        self._plan_session(sess)             # replace: the freshest view of the future
        prog = sess.program
        if prog is not None and sess.walked and isinstance(prog.sites.get(sess.walked[-1]), VisitByCallsite):
            self._route_followup(sess)       # extend: the reply names the branch outright
        self.planner.wake()

    def _advance(self, sess: LiveSession, req: PendingRequest) -> CallObservation:
        """A new call: close the open one, identify the callsite, extend walked."""
        if sess.open_obs is not None:
            sess.pending.append(sess.open_obs)
        site, extras = decompose(req.body)
        prog = self._bind(sess, site)
        site = prog.add_callsite(site)
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
            self.planner.void_session(sess.id, sess.epoch)
            self._plan_session(sess)  # overlap the successors' work with this call's decode
        return sess.open_obs

    def _bind(self, sess: LiveSession, site: Callsite) -> Program:
        """Bind the session to its Program by content — the entry callsite identifies
        the program, nothing else crosses the wire. A first call matching only the
        middle of a known program is a resumed process: serve it there, but keep the
        half session out of the statistics."""
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
        """One predicted call -> its speculative jobs: probe a fully-rebuilt routing
        prompt (plus, alongside, the rendered prompt as ordinary create/promote work —
        whichever lands first collapses the other's uncached cost), prefill a
        fully-rebuilt call whole, else the longest reconstructable — or failing
        that the static — prefix."""
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
        if isinstance(site, VisitByCallsite) and not (complete and site.resp_n >= 2):
            # which probe gate is failing, and how far the rebuild got
            print(f"[plan] visit gate: resolved={resolved} ok={proto.get('ok', 0)}/{proto.get('n', 0)} "
                  f"resp_n={site.resp_n} rebuilt={len(text)}B", flush=True)
        now = time.monotonic()
        deadline = max(now, anchor + c.t50 - KVPlanner.SAFETY_K * (c.t90 - c.t50))
        t90 = max(deadline, anchor + c.t90)
        jobs: List[Job] = []
        if complete and isinstance(site, VisitByCallsite) and site.resp_n >= 2:
            prompt = self._probe_prompt(site, text)
            ptoks = self.engine.tokenize(prompt)
            unc, host = self.engine.cost(ptoks)
            jobs.append(Job(key=f"{sess.id}|probe|{c.key}", sid=sess.id, epoch=sess.epoch,
                            site=c.key, kind="probe", toks=ptoks, prompt=prompt, p=c.p,
                            value=c.p * (unc + 0.8 * host), deadline=deadline, t90=t90,
                            work=unc, host=host,
                            on_result=self._probe_handler(sess, site, c, text)))
            # fall through: the rendered prompt also queues as create/promote work,
            # so a blocked probe shrinks toward unc=0 as ordinary chunks land
        if complete:
            toks = self.engine.tokenize(
                self.engine.render([sysmsg, {"role": "user", "content": text}], tools=tools))
        else:
            cut: Optional[List[int]] = None
            if text:
                rendered = self.engine.render([sysmsg, {"role": "user", "content": text}], tools=tools)
                i = rendered.rfind(text)
                if i >= 0:
                    cut = self.engine.tokenize(rendered[:i + len(text)])[:-1]
            static = self._prefix_tokens(site)
            toks = cut if cut is not None and len(cut) > len(static) else static
        if len(toks) < 16:  # shorter than a cache block: nothing to gain
            return jobs
        unc, host = self.engine.cost(toks)
        if unc + host < self.engine._stride:
            return jobs     # device-resident; if it gets evicted, a later refresh re-plans it
        kind = "promote" if unc <= self.engine._stride and host > 0 else "create"
        jobs.append(Job(key=f"{sess.id}|{kind}|{c.key}", sid=sess.id, epoch=sess.epoch,
                        site=c.key, kind=kind, toks=toks, prompt=None, p=c.p,
                        value=c.p * (unc + 0.8 * host), deadline=deadline, t90=t90,
                        work=unc, host=host))
        return jobs

    # ------------------------------------------------------------------ routing foresight
    def _probe_prompt(self, site: VisitByCallsite, user_text: str) -> str:
        """The rebuilt routing prompt plus the shared reply head, cut just before the
        first candidate handle so the probed token is the choice itself."""
        head = site.resp_head
        for i in sorted(head.find(c["handle"]) for c in self._candidates(user_text)):
            if i >= 0:
                head = head[:i]
                break
        return self.engine.render([{"role": "system", "content": site.system_prompt},
                                   {"role": "user", "content": user_text}]) + head

    @staticmethod
    def _candidates(user_text: str) -> List[Dict[str, str]]:
        parts = _strip_hint(user_text).split("Candidates (choose by handle):\n", 1)
        lines = parts[1].split("\n") if len(parts) > 1 else []
        return [c for c in (parse_candidate_line(l) for l in lines) if c]

    def _probe_handler(self, sess: LiveSession, site: VisitByCallsite, c: PredictedCall,
                       user_text: str):
        """Bind the probe follow-up to the epoch it was planned in."""
        epoch = sess.epoch

        def on_result(res: Dict[int, Any], job: Job) -> None:
            if sess.epoch == epoch and sess.id in self.sessions:
                self._on_probe(sess, site, c, user_text, job, res)
        return on_result

    def _on_probe(self, sess: LiveSession, site: VisitByCallsite, c: PredictedCall,
                  user_text: str, job: Job, res: Dict[int, Any]) -> None:
        """The probed distribution over candidate handles tells which branch the
        routing call will take: plan the successors of the likely ones."""
        base, prompt = job.toks, job.prompt or ""
        mass: Dict[str, Tuple[float, str]] = {}
        for cand in self._candidates(user_text):
            seq = self.engine.tokenize(prompt + cand["handle"])
            if len(seq) > len(base) and seq[:len(base)] == base and seq[len(base)] in res:
                mass[cand["handle"]] = (exp(res[seq[len(base)]].logprob), cand["node"])
        total = sum(v for v, _ in mass.values())
        if not total:
            return
        dist = sorted(((v / total, h, node) for h, (v, node) in mass.items()), reverse=True)
        print("[probe] " + site.label + " -> " + " ".join(f"{h}={p:.2f}" for p, h, _ in dist),
              flush=True)
        jobs = []
        for p, _, node in dist[:3]:
            if p < P_ROUTE:
                break
            jobs.extend(self._succ_job(sess, site.key, node, c.p * p, c.t50, c.t90))
        self._submit_extra(sess, jobs)

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

    def _prefix_tokens(self, site: Callsite) -> List[int]:
        """Token ids of the callsite's static prompt head, rendered exactly as a real
        request would be. Until the LCP has converged (prefix_n >= 2) only the system
        region is trusted; the final token is dropped because the cut may split one."""
        stable = site.stable_prefix if site.prefix_n >= 2 else ""
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
            prog.update_graph(sess.walked, sess.pending)
            n = prog.n_sessions
            if n >= REBUILD_AT and n & (n - 1) == 0:
                prog.rebuild()
                print(f"[rebuild] n_sessions={n} nodes={len(prog.nodes)}", flush=True)
        print(f"[close] {sid} calls={len(sess.walked)}" + (" tainted" if sess.tainted else ""),
              flush=True)
        return True


async def main(model: str = "Qwen/Qwen3-8B", port: int = 8964, speculate: bool = True) -> None:
    server = HttpServer(port=port)
    await server.start()
    await Controller(model, server, speculate=speculate, context_length=16384).start_serving()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-spec"]
    asyncio.run(main(*args[:1], speculate="--no-spec" not in sys.argv))  # type: ignore
