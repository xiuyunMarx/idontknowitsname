"""Controller: structure-driven serving of byllm agents over one vLLM engine.

Every completed call advances its session's walked sequence through the session's
Program; the Program's predictions then drive speculative prefill of the next
calls' stable prompt prefixes (overlapped with the current call's decode and with
the agent's own compute between calls), a warm-set of prefixes kept alive against
vLLM's LRU eviction, and request priorities (ReAct continuations first).

    python -m server.controller [MODEL]
"""
import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vllm import SamplingParams

from decompiler.parser import decompose, is_continuation
from decompiler.primitives import Callsite, CallObservation, Program, _lcp
from model.model import Engine
from server.http_server import HttpServer, PendingRequest

P_MIN = 0.2            # prefill only steps predicted at least this likely
REFRESH_S = 30.0       # a prefix touched more recently than this is already warm
SPEC_WAIT_S = 5.0      # longest a queued speculative prefill waits for budget
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


class Controller:
    def __init__(self, model: str, server: HttpServer):
        self.engine = Engine(model)
        self.sessions: Dict[str, LiveSession] = {}    # session id -> live state
        self.programs: Dict[str, Program] = {}        # program name -> Program
        self.server = server
        self.pool = server.pool
        self.warm: Dict[str, float] = {}              # site key -> last cache touch (monotonic)
        self._prefix_tok: Dict[str, Tuple[str, List[int]]] = {}  # site key -> (stable_prefix, token ids)
        self._spec_inflight: set = set()              # site keys with a queued/running prefill

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
            ob.n_turns += 1 #type: ignore
            ob.tool_gaps.append(max(0.0, req.t_arrive - ob.t_done)) #type: ignore
        else:
            ob = self._advance(sess, req)
        prompt = self.engine.render(body["messages"], tools=body.get("tools"))
        t = body.get("temperature")
        sp = SamplingParams(temperature=0.7 if t is None else t,
                            max_tokens=body.get("max_tokens") or MAX_TOKENS)
        rid = f"{sess.id}-{sess.epoch}t{ob.n_turns}-{uuid.uuid4().hex[:8]}" #type: ignore
        t0 = time.perf_counter()
        text = await self.engine.generate(prompt, rid, sp, priority=-1 if cont else 0)
        ob.engine_s += time.perf_counter() - t0 #type: ignore
        ob.t_done = time.monotonic() #type: ignore
        sess.last_raw = body
        req.reply(self._answer(body, text))
        self._speculate(sess)  # gap window: the agent runs its own code now, the engine is free

    def _advance(self, sess: LiveSession, req: PendingRequest) -> CallObservation:
        """A new call: close the open one, identify the callsite, extend walked."""
        if sess.open_obs is not None:
            sess.pending.append(sess.open_obs)
        site, extras = decompose(req.body)
        prog = self._bind(sess, req, site)
        site = prog.add_callsite(site)
        hit = "" if sess.predicted is None else f" predicted={'hit' if sess.predicted == site.key else 'miss'}"
        print(f"[call] {sess.id} #{len(sess.walked)} {site.label}{hit}", flush=True)
        sess.walked.append(site.key)
        sess.epoch += 1
        sess.open_obs = CallObservation(key=site.key, t_arrive=req.t_arrive, t_done=req.t_arrive,
                                        candidates=extras.candidates)
        self._speculate(sess)  # overlap the successors' prefill with this call's decode
        return sess.open_obs

    def _bind(self, sess: LiveSession, req: PendingRequest, site: Callsite) -> Program:
        """Bind the session to its Program: by the name declared in `user`
        ("program:pid"), or for anonymous sessions by the entry callsite."""
        if sess.program is None:
            name = req.session.split(":", 1)[0]
            if not name or name == "anon":
                name = next((n for n, p in self.programs.items() if p.entry == site.key), site.key)
            prog = self.programs.get(name)
            if prog is None:
                prog = self.programs[name] = Program()
                print(f"[program] new: {name}", flush=True)
            sess.program = prog
        if not sess.program.entry:
            sess.program.entry = site.key
        return sess.program

    def _answer(self, body: dict, text: str) -> Any:
        """Native-tools responses must carry structured tool_calls; Qwen emits them as
        <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks in the text."""
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
    def _speculate(self, sess: LiveSession) -> None:
        """Queue prefill of the predicted next calls' stable prefixes, most valuable
        first. The warm map is the KV schedule: what gets touched stays cached."""
        prog = sess.program
        if prog is None:
            return
        steps = [(key, p) for key, p, _ in prog.predict(sess.walked) if p >= P_MIN]
        sess.predicted = steps[0][0] if steps else None
        now = time.monotonic()
        plans: List[Tuple[float, str, List[int]]] = []
        for key, p in steps:
            site = prog.sites.get(key)
            if site is None or key in self._spec_inflight or now - self.warm.get(key, -1e9) < REFRESH_S:
                continue
            toks = self._prefix_tokens(site)
            if len(toks) >= 16:  # shorter than one cache block: nothing to gain
                plans.append((p * len(toks), key, toks))
        plans.sort(key=lambda v: -v[0])
        for _, key, toks in plans:
            self._spec_inflight.add(key)
            asyncio.create_task(self._spec_prefill(sess, sess.epoch, key, toks))

    async def _spec_prefill(self, sess: LiveSession, epoch: int, key: str, toks: List[int]) -> None:
        """Wait for spec budget, then land one predicted prefix in the prefix cache.
        Queued work is dropped once the session moves on or closes."""
        try:
            tick = (self.engine.tbt_ms or 50.0) / 1000
            waited = 0.0
            while self.engine.spec_allowance() < len(toks):
                await asyncio.sleep(tick)
                waited += tick
                if waited > SPEC_WAIT_S or sess.epoch != epoch or sess.id not in self.sessions:
                    return
            if await self.engine.prefill(toks, f"spec-{uuid.uuid4().hex[:8]}", cost=len(toks)):
                self.warm[key] = time.monotonic()
        finally:
            self._spec_inflight.discard(key)

    def _prefix_tokens(self, site: Callsite) -> List[int]:
        """Token ids of the callsite's stable prompt head, rendered exactly as a real
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
        if prog is not None and sess.walked:
            prog.update_graph(sess.walked, sess.pending)
            n = prog.n_sessions
            if n >= REBUILD_AT and n & (n - 1) == 0:
                prog.rebuild()
                print(f"[rebuild] n_sessions={n} nodes={len(prog.nodes)}", flush=True)
        print(f"[close] {sid} calls={len(sess.walked)}", flush=True)
        return True


async def main(model: str = "Qwen/Qwen3-8B", port: int = 8964) -> None:
    server = HttpServer(port=port)
    await server.start()
    await Controller(model, server).start_serving()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:2])) #type: ignore
