"""Controller: structure-driven serving of byllm agents over one vLLM engine.

Every completed call advances its session's walked sequence through the session's
Program. The Program gives back three levels of foresight, each driving cheaper
work ahead of demand:
  - branch statistics (trie) pick which calls to expect next;
  - value-flow rules rebuild those calls' prompts from values the session already
    produced — a fully-rebuilt prompt is prefilled whole, a partial one up to its
    last known byte, and only then does the static stable-prefix fallback apply;
  - a fully-rebuilt *routing* prompt is speculatively executed with a one-token
    probe: the choice distribution read off the scaffold token tells which branch
    to prefill before the routing request even arrives. A routing reply that has
    actually landed short-circuits all of this and prefills its chosen branch.
Requests are prioritized (ReAct continuations first), and the warm map keeps hot
static prefixes alive against vLLM's LRU eviction.

    python -m server.controller [MODEL]
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

from vllm import SamplingParams

from decompiler.parser import _strip_hint, decompose, is_continuation, parse_candidate_line
from decompiler.primitives import (Callsite, CallObservation, Program, VisitByCallsite, _lcp,
                                   chosen_candidates, node_type)
from model.model import Engine
from server.http_server import HttpServer, PendingRequest

P_MIN = 0.02            # prefill only steps predicted at least this likely
P_ROUTE = 0.3          # act on probed routing choices at least this likely
REFRESH_S = 30.0       # a static prefix touched more recently than this is already warm
SPEC_WAIT_S = 5.0      # longest queued speculative work waits for budget
T_BASE = 120.0         # idle seconds before a session is finalized
T_SHORT = 15.0         # idle timeout once the program says the session is over
END_PROB_SHORT = 0.7   # end_prob above this switches to T_SHORT
SWEEP_S = 5.0          # idle sweeper period
REBUILD_AT = 8         # rebuild when n_sessions reaches this, then every doubling
MAX_TOKENS = 4096      # decode cap when the request does not set one

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
    epoch: int = 0                               # Bumps when walked grows; voids queued speculation
    predicted: Optional[str] = None              # Last top-1 prediction, for hit logging
    tainted: bool = False                        # Joined mid-program: serve it, keep stats clean


class Controller:
    def __init__(self, model: str, server: HttpServer, speculate: bool = True, **engine_kwargs):
        self.speculate = speculate     # off: plain serving over the prefix cache (the baseline)
        self.engine = Engine(model, **engine_kwargs)
        self.sessions: Dict[str, LiveSession] = {}    # session id -> live state
        self.programs: Dict[str, Program] = {}        # entry callsite key -> Program
        self.server = server
        self.pool = server.pool
        self.warm: Dict[str, float] = {}              # site key -> last static-prefix touch (monotonic)
        self._prefix_tok: Dict[str, Tuple[str, List[int]]] = {}  # site key -> (stable_prefix, token ids)
        self._spec_inflight: set = set()              # dedup keys of queued/running speculation

    # ------------------------------------------------------------------ lifecycle
    async def start_serving(self) -> None:
        await self.engine.warmup()
        profile = f"model/{self.engine.name.replace('/', '--')}.profile.json"
        if not self.engine.load_profile(profile):
            await self.engine._profile()
            self.engine.save_profile(profile)
        asyncio.create_task(self._sweep())
        while True:
            request = await self.pool.get()
            asyncio.create_task(self._serve(request))

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
    async def _serve(self, req: PendingRequest) -> None:
        try:
            if req.kind == "close":
                req.reply(self._finalize(req.session))
                return
            sess = self.sessions.get(req.session)
            if sess is None:
                sess = self.sessions[req.session] = LiveSession(id=req.session)
            sess.inflight += 1
            sess.last_seen = time.monotonic()
            try:
                await self._completion(sess, req)
            finally:
                sess.inflight -= 1
                sess.last_seen = time.monotonic()
        except Exception as e:
            req.fail(e)

    async def _completion(self, sess: LiveSession, req: PendingRequest) -> None:
        body = req.body
        cont = (sess.open_obs is not None and sess.last_raw is not None
                and is_continuation(sess.last_raw, body))
        if cont:
            ob = sess.open_obs
            ob.n_turns += 1 # type: ignore[union-attr]
            ob.tool_gaps.append(max(0.0, req.t_arrive - ob.t_done))  # type: ignore[union-attr]
        else:
            ob = self._advance(sess, req)
        prompt = self.engine.render(body["messages"], tools=body.get("tools"))
        t = body.get("temperature")
        sp = SamplingParams(temperature=0.7 if t is None else t,
                            max_tokens=body.get("max_tokens") or MAX_TOKENS,
                            stop=body.get("stop"))
        rid = f"{sess.id}-{sess.epoch}t{ob.n_turns}-{uuid.uuid4().hex[:8]}"  # type: ignore[union-attr]
        t0 = time.perf_counter()
        text = await self.engine.generate(prompt, rid, sp, priority=-1 if cont else 0)
        ob.engine_s += time.perf_counter() - t0  # type: ignore[union-attr]
        ob.t_done = time.monotonic()  # type: ignore[union-attr]
        ob.response = text  # type: ignore[union-attr]
        sess.last_raw = body
        req.reply(self._answer(body, text))
        if not self.speculate:
            return
        # The gap window: the agent runs its own code now, the engine is free.
        prog = sess.program
        if prog is not None and sess.walked and isinstance(prog.sites.get(sess.walked[-1]), VisitByCallsite):
            self._route_followup(sess)
        self._speculate(sess)

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
        sess.open_obs = CallObservation(key=site.key, t_arrive=req.t_arrive, t_done=req.t_arrive,
                                        candidates=extras.candidates, bindings=extras.bindings,
                                        self_view=extras.self_view, walker=extras.walker,
                                        here=extras.here, cand_block=extras.cand_block,
                                        user_text=extras.user_text)
        if self.speculate:
            self._speculate(sess)  # overlap the successors' prefill with this call's decode
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

    # ------------------------------------------------------------------ speculation
    def _session_obs(self, sess: LiveSession) -> List[CallObservation]:
        """The completed observations value-flow rules may draw from."""
        obs = list(sess.pending)
        if sess.open_obs is not None and sess.open_obs.response:
            obs.append(sess.open_obs)
        return obs

    def _speculate(self, sess: LiveSession) -> None:
        """Queue the best speculative work for every likely next call."""
        prog = sess.program
        if prog is None:
            return
        steps = [(k, p) for k, p, _ in prog.predict(sess.walked) if p >= P_MIN]
        sess.predicted = steps[0][0] if steps else None
        for key, _ in steps:
            self._launch(sess, key)

    def _route_followup(self, sess: LiveSession) -> None:
        """The routing reply names the nodes the walker visits next: prefill their
        calls right away instead of hedging on branch statistics."""
        prog, ob = sess.program, sess.open_obs
        if prog is None or ob is None:
            return
        for _, node in chosen_candidates(ob):
            succ = prog.type_succ.get(node_type(node))
            if succ:
                self._launch(sess, succ.most_common(1)[0][0])

    def _launch(self, sess: LiveSession, key: str) -> None:
        """One predicted call: probe a fully-rebuilt routing prompt, prefill a
        fully-rebuilt call whole, else prefill the longest reconstructable —
        or failing that the static — prefix."""
        prog = sess.program
        site = prog.sites.get(key) if prog else None
        if prog is None or site is None:
            return
        sysmsg = {"role": "system", "content": site.system_prompt}
        tools = getattr(site, "tool_schema", None)
        obs = self._session_obs(sess)
        text, complete = prog.resolve_user(key, obs)
        complete = complete and prog.proto.get(key, {}).get("ok", 0) >= 1
        stage = f"{sess.id}|{key}|{sess.epoch}|{len(obs)}"
        if complete:
            if isinstance(site, VisitByCallsite) and site.resp_n >= 2:
                dedup = "probe:" + stage
                if dedup not in self._spec_inflight:
                    self._spec_inflight.add(dedup)
                    asyncio.create_task(self._probe_route(sess, sess.epoch, dedup, site, text))
                return
            full = self.engine.render([sysmsg, {"role": "user", "content": text}], tools=tools)
            self._queue(sess, stage, self.engine.tokenize(full), warm_key=None)
            return
        cut: Optional[List[int]] = None
        if text:
            rendered = self.engine.render([sysmsg, {"role": "user", "content": text}], tools=tools)
            i = rendered.rfind(text)
            if i >= 0:
                cut = self.engine.tokenize(rendered[:i + len(text)])[:-1]
        static = self._prefix_tokens(site)
        if cut is not None and len(cut) > len(static):
            self._queue(sess, stage, cut, warm_key=None)
        else:
            self._queue(sess, key, static, warm_key=key)

    def _queue(self, sess: LiveSession, dedup: str, toks: List[int], warm_key: Optional[str]) -> None:
        if len(toks) < 16 or dedup in self._spec_inflight:  # shorter than a cache block: nothing to gain
            return
        if warm_key and time.monotonic() - self.warm.get(warm_key, -1e9) < REFRESH_S:
            return
        self._spec_inflight.add(dedup)
        asyncio.create_task(self._spec_prefill(sess, sess.epoch, dedup, toks, warm_key))

    async def _await_budget(self, sess: LiveSession, epoch: int, cost: int) -> bool:
        """Wait until the engine can absorb `cost` speculative tokens; give up when
        the session moves on, closes, or the wait exceeds SPEC_WAIT_S."""
        tick = (self.engine.tbt_ms or 50.0) / 1000
        waited = 0.0
        while self.engine.spec_allowance() < cost:
            await asyncio.sleep(tick)
            waited += tick
            if waited > SPEC_WAIT_S or sess.epoch != epoch or sess.id not in self.sessions:
                return False
        return sess.epoch == epoch and sess.id in self.sessions

    async def _spec_prefill(self, sess: LiveSession, epoch: int, dedup: str,
                            toks: List[int], warm_key: Optional[str]) -> None:
        """Land one predicted prefix in the prefix cache."""
        try:
            if not await self._await_budget(sess, epoch, len(toks)):
                return
            if await self.engine.prefill(toks, f"spec-{uuid.uuid4().hex[:8]}", cost=len(toks)):
                if warm_key:
                    self.warm[warm_key] = time.monotonic()
        finally:
            self._spec_inflight.discard(dedup)

    async def _probe_route(self, sess: LiveSession, epoch: int, dedup: str,
                           site: VisitByCallsite, user_text: str) -> None:
        """Speculatively execute a predicted routing call"""
        try:
            parts = _strip_hint(user_text).split("Candidates (choose by handle):\n", 1)
            lines = parts[1].split("\n") if len(parts) > 1 else []
            cands = [c for c in (parse_candidate_line(l) for l in lines) if c]
            head = site.resp_head
            for i in sorted(head.find(c["handle"]) for c in cands):
                if i >= 0:
                    head = head[:i]
                    break
            prompt = self.engine.render([{"role": "system", "content": site.system_prompt},
                                         {"role": "user", "content": user_text}]) + head
            base = self.engine.tokenize(prompt)
            if not await self._await_budget(sess, epoch, len(base)):
                return
            res = await self.engine.probe(prompt, f"probe-{uuid.uuid4().hex[:8]}")
            if not res or sess.epoch != epoch or sess.id not in self.sessions:
                return
            mass: Dict[str, Tuple[float, str]] = {}
            for c in cands:
                seq = self.engine.tokenize(prompt + c["handle"])
                if len(seq) > len(base) and seq[:len(base)] == base and seq[len(base)] in res:
                    mass[c["handle"]] = (exp(res[seq[len(base)]].logprob), c["node"])
            total = sum(v for v, _ in mass.values())
            if not total:
                return
            dist = sorted(((v / total, h, node) for h, (v, node) in mass.items()), reverse=True)
            print("[probe] " + site.label + " -> " + " ".join(f"{h}={p:.2f}" for p, h, _ in dist),
                  flush=True)
            prog = sess.program
            for p, _, node in dist[:3]:
                if p < P_ROUTE or prog is None:
                    break
                succ = prog.type_succ.get(node_type(node))
                if succ:
                    self._launch(sess, succ.most_common(1)[0][0])
        finally:
            self._spec_inflight.discard(dedup)

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
        """Fold the session's observations into its Program and drop it."""
        sess = self.sessions.pop(sid, None)
        if sess is None:
            return False
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
    await Controller(model, server, speculate=speculate, max_model_len=16384).start_serving()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-spec"]
    asyncio.run(main(*args[:1], speculate="--no-spec" not in sys.argv)) #type: ignore