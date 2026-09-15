"""
Controller: workflow-aware serving of compiled agents over one SGLang engine.

The agent launcher registers the program's static analysis (call sites as
PromptTemplates, binding heterogeneity and lifetimes, the call-site graph) at
/v1/programs/register; every request then names its call site. The controller

  - re-lays out each call message in the site's layout (constants and shared
    values first, fresh values last, header behind them when the leading value
    is shared across sites) so the KV prefix is reused across calls and sites;
  - marks the static graph with transition frequencies and times, predicts the
    session's next calls, speculatively renders their prompts from the values
    the session already carries, and turns them into deadline-stamped KV jobs
    (promotion and eviction protection) for server.kv_planner.

Nothing about the program is learned from traffic.

python -m server.server [MODEL] [--lru] [--no-relayout] [--no-header-last] [--kv N] [--host GB] [--sched ...]
"""
import argparse
import asyncio
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from model.device_profiler import profile_device
from model.model import PRIORITY_CONT, PRIORITY_REAL, Engine
from server.http_server import HttpServer, PendingRequest
from server.kv_planner import Job, KVPlanner
from server.wire import content, first_user, is_continuation, set_content
from static_analysis.primitives import (Binding, BindingKind, CallInstance, CallSiteID, Edge, Heterogeneity,
                                        PredictedCall, Program, PromptTemplate, field_spec, program_from_dict,
                                        render_repr)

P_MIN = 0.02           # plan only steps predicted at least this likely
T_BASE = 120.0         # idle seconds before a session is finalized
T_SHORT = 15.0         # idle timeout once the session sits on an exit site
SWEEP_S = 5.0          # idle sweeper period
MAX_TOKENS = 4096      # decode cap when the request does not set one
MAX_BATCH = 32
HEADER_LAST = True     # --no-header-last clears it: layouts never move the header behind the values

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class LiveSession:
    id: str                                       # Session id, from the agent (its pid)
    program: Optional[Program] = None            # Bound by the first request's call site
    calls: List[CallInstance] = field(default_factory=list)   # Closed calls, in order
    open: Optional[CallInstance] = None          # The call being served (ReAct turns fold into it)
    last_raw: Optional[dict] = None              # Previous request body, for is_continuation
    last_seen: float = 0.0                       # monotonic, for the idle sweeper
    inflight: int = 0                            # HTTP requests being served right now
    epoch: int = 0                               # Bumps per call; voids queued jobs
    predicted: Optional[CallSiteID] = None       # Last top-1 prediction, for hit logging
    plan_anchor: float = 0.0                     # monotonic t the gap predictions count from
    opaque: int = 0                              # requests served without a registered call site

    @property
    def history(self) -> List[CallInstance]:
        """Every call whose values the session carries, the open one included."""
        return self.calls + ([self.open] if self.open is not None else [])


class Controller:
    PLAN_TOK_CACHE = 1024

    def __init__(self, model: str, server: HttpServer, plan: bool = True,
                 enable_relayout: bool = True, **engine_kwargs):
        self.plan = plan                          # off: plain serving over the prefix cache (the LRU baseline)
        self.enable_relayout = enable_relayout
        self.engine: Engine = Engine(model, **engine_kwargs)
        self.server = server
        self.pool = server.pool
        self.sessions: Dict[str, LiveSession] = {}
        self.programs: Dict[str, Program] = {}                      # agent file -> Program
        self._by_site: Dict[Tuple[str, int, str], Tuple[Program, PromptTemplate]] = {}  # (signature, lineno, basename)
        self._prefix_tok: Dict[CallSiteID, Tuple[str, List[int]]] = {}   # site -> (fixed head, token ids)
        self._plan_tok: "OrderedDict[Tuple[CallSiteID, bool, str], Optional[List[int]]]" = OrderedDict()
        self._relayout_seen: set = set()          # sites whose served layout was logged once
        self._resets: Dict[str, Dict[str, frozenset]] = {}   # agent file -> ability -> fields it resets
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
                elif req.kind == "register":
                    try:
                        req.reply(self.register(req.body))
                    except Exception as e:
                        req.fail(e)
                else:
                    batch.append(req)
                if len(batch) >= MAX_BATCH:
                    break
                try:
                    req = self.pool.get_nowait()  # drain whatever else arrived
                except asyncio.QueueEmpty:
                    break
            if batch:
                await self._step(batch)

    async def _sweep(self) -> None:
        """Finalize sessions that have gone quiet."""
        while True:
            await asyncio.sleep(SWEEP_S)
            now = time.monotonic()
            for sid, sess in list(self.sessions.items()):
                if sess.inflight > 0:
                    continue
                t = T_BASE
                last = sess.open or (sess.calls[-1] if sess.calls else None)
                if sess.program is not None and last is not None and last.key in sess.program.exits:
                    t = T_SHORT   # the program may end here: a shorter wait
                if now - sess.last_seen > t:
                    self._finalize(sid)

    # ------------------------------------------------------------------ registration
    def register(self, body: dict) -> dict:
        """Adopt a program's static analysis."""
        program = program_from_dict(body["program"])
        file = os.path.abspath(str(body["file"]))
        if body.get("no_header_last"):          # the registrant's choice for this program (fact_check keeps its headers in front)
            for t in program.sites.values():
                t.no_header_last = True
        if self.enable_relayout:
            program.decide_layouts(header_last=HEADER_LAST)
        for t in program.sites.values():
            print(f"[layout] {t.key!r} order={t.order} header_last={t.header_last} "
                  f"bindings={[(b.name, b.heterogeneity.name) for b in t.bindings]}", flush=True)
        self.programs[file] = program
        self._resets[file] = program.resets()
        for k, t in program.sites.items():
            self._by_site[(k.signature, k.lineno, os.path.basename(k.file or file))] = (program, t)
        edges = sorted(f"{a!r}->{b!r}" for (a, b) in program.edges)
        print(f"[program] {file}: {len(program.sites)} sites, entry {program.entry!r}, "
              f"exits {sorted(repr(k) for k in program.exits)}, edges {edges}", flush=True)
        return {"ok": True, "file": file, "sites": len(program.sites)}

    def lookup(self, body: dict) -> Optional[Tuple[Program, PromptTemplate]]:
        """The registered template of a request's call site, or None (no
        registration, or a request without a call site: served opaque)."""
        cs = CallSiteID.from_request(body)
        if cs is None:
            return None
        return self._by_site.get((cs.signature, cs.lineno, os.path.basename(cs.file)))

    # ------------------------------------------------------------------ request path
    async def _step(self, batch: List[PendingRequest]) -> None:
        """Admit one drained batch: identify each request's call, re-lay it out,
        plan the session, and hand it to the engine."""
        for req in sorted(batch, key=lambda r: r.t_arrive):
            sess = self.sessions.get(req.session)
            if sess is None:
                sess = self.sessions[req.session] = LiveSession(id=req.session)
            sess.inflight += 1
            sess.last_seen = time.monotonic()
            try:
                t_cls = time.monotonic()
                # A ReAct turn or typed retry re-sends the call message in byllm's
                # native layout: re-lay it out first, so it compares equal to the
                # served body (a different call's message does not fit and is left alone).
                self._relayout(sess, req.body)
                cont = (sess.open is not None and sess.last_raw is not None
                        and is_continuation(sess.last_raw, req.body))
                if cont:
                    inst = sess.open
                    inst.gap.append(max(0.0, req.t_arrive - inst.t_done))   # type: ignore[union-attr]
                else:
                    inst = self._advance(sess, req)
                advance_ms = (time.monotonic() - t_cls) * 1000
            except Exception as e:
                sess.inflight -= 1
                req.fail(e)
                continue
            asyncio.create_task(self._generate(sess, req, inst, cont, t_cls, advance_ms))  # type: ignore[arg-type]
        if self.planner is not None:
            self.planner.wake()

    async def _generate(self, sess: LiveSession, req: PendingRequest, inst: Optional[CallInstance],
                        cont: bool, t_cls: float, advance_ms: float) -> None:
        """One real request through the engine, then a planning refresh: its reply
        opens the gap window, the agent runs its own code now."""
        try:
            t_pre = time.perf_counter()
            body = req.body
            prompt = self.engine.render(body["messages"], tools=body.get("tools"))
            sp = {"temperature": 0.7 if body.get("temperature") is None else body.get("temperature"),
                  "max_new_tokens": body.get("max_tokens") or MAX_TOKENS,
                  "stop": body.get("stop")}
            turn = len(inst.engine_time) if inst is not None else 0
            rid = f"{sess.id}-{sess.epoch}t{turn}-{uuid.uuid4().hex[:8]}"
            ids = self.engine.tokenize(prompt)
            t0 = time.perf_counter()
            text = await self.engine.generate(prompt, rid, sp, ids=ids,
                                              priority=PRIORITY_CONT if cont else PRIORITY_REAL)
            dt = time.perf_counter() - t0
            if inst is not None:
                if not inst.engine_time:
                    inst.served_ids = ids       # the call message as served (first turn)
                inst.engine_time.append(dt)
                inst.t_done = time.monotonic()
                inst.response = text
            sess.last_raw = body
            req.reply(self._answer(body, text))
        except Exception as e:
            req.fail(e)
            return
        finally:
            sess.inflight -= 1
            sess.last_seen = time.monotonic()
        if self.planner is None or sess.id not in self.sessions or inst is None:
            return
        t_post = time.perf_counter()
        tpl = inst.template
        # Past the head some later prompt can still share, this prompt is one-off:
        # its fresh bindings and the reply. Only that tail is demoted; the head
        # stays in the normal band. The head is the longer of what the session could
        # rebuild before the call (bytes earlier prompts already hold) and what a
        # successor's prompt will carry unchanged (the static edges say which
        # fields the program rewrites on the way).
        head = len(self._prefix_tokens(tpl))
        prev = sess.calls[-1] if sess.calls else None
        via = sess.program.edges.get((prev.key, tpl.key)) if (prev is not None and sess.program is not None) else None
        text_head, _ = self._resolve(sess, tpl, via, upto=inst)
        for text in (text_head, self._forward_head(sess, inst)):
            cut = self._planned_tokens(tpl, text, False) if text else None
            if cut is not None and len(cut) > head:
                head = len(cut)
        self.planner.note_served(sess.id, ids, head)
        sess.plan_anchor = inst.t_done          # the anchor every gap sample counts from
        self._plan_session(sess)                # replace: the freshest view of the future
        self.planner.wake()
        print(f"[ctl] {rid} queue_ms={(t_cls - req.t_arrive) * 1000:.1f} advance_ms={advance_ms:.1f} "
              f"pre_ms={(t0 - t_pre) * 1000:.1f} post_ms={(time.perf_counter() - t_post) * 1000:.1f}", flush=True)

    def _advance(self, sess: LiveSession, req: PendingRequest) -> Optional[CallInstance]:
        """A new call: close the open one, identify the call site, cut the values,
        re-lay out the message, plan the successors."""
        if sess.open is not None:
            self._close_call(sess)
        found = self.lookup(req.body)
        if found is None:
            sess.opaque += 1
            if sess.opaque == 1:
                cs = req.body.get("callsite")
                print(f"[call] {sess.id} opaque request (callsite={cs!r}): no registered template", flush=True)
            return None
        prog, tpl = found
        if sess.program is None:
            sess.program = prog
        msg = first_user(req.body)
        values = tpl.split(content(msg)) if msg is not None else None
        if values is None:
            print(f"[call] {sess.id} {tpl.key!r}: message does not fit the template; served as is", flush=True)
            values = {}
        inst = CallInstance(template=tpl, values=values, t_arrive=req.t_arrive, t_done=req.t_arrive)
        if values:
            self._relayout(sess, req.body, inst)
        hit = "" if sess.predicted is None else f" predicted={'hit' if sess.predicted == tpl.key else 'miss'}"
        print(f"[call] {sess.id} #{len(sess.calls)} {tpl.key!r}{hit}", flush=True)
        prev = sess.calls[-1] if sess.calls else None
        if prev is not None:
            e = prog.add_edge(prev.key, tpl.key)      # the transition is known at arrival
            e.count += 1
            e.gap.append(max(0.0, req.t_arrive - prev.t_done))
        self._invalidate(sess, tpl)
        sess.open = inst
        sess.epoch += 1
        sess.plan_anchor = req.t_arrive     # an early anchor to start planning; the reply re-anchors
        if self.planner is not None:
            self.planner.note_arrival(sess.id, repr(tpl.key), req.t_arrive)
            self.planner.void_session(sess.id, sess.epoch)
            self._plan_session(sess)        # overlap the successors' work with this call's decode
        return inst

    def _close_call(self, sess: LiveSession) -> None:
        """The open call is over: its timing goes on its site and on the edge it
        was reached by."""
        inst = sess.open
        if inst is None:
            return
        sess.open = None
        sess.calls.append(inst)
        prog = sess.program
        if prog is None or not inst.engine_time:
            return
        turns = len(inst.engine_time)
        gap = inst.gap[:turns - 1] + [0.0] * max(0, turns - 1 - len(inst.gap))
        inst.template.stats.add_record(inst.engine_time, turns, gap)
        if len(sess.calls) >= 2:
            e = prog.add_edge(sess.calls[-2].key, inst.key)
            e.stats.add_record(inst.engine_time, turns, gap)

    def _relayout(self, sess: LiveSession, body: dict, inst: Optional[CallInstance] = None) -> None:
        """Re-emit the call message in its site's layout. Idempotent: the values are
        cut from whatever layout arrived and rendered in the served one."""
        inst = inst or sess.open
        if not self.enable_relayout or inst is None:
            return
        tpl = inst.template
        if tpl.order is None and not tpl.header_last:
            return
        msg = first_user(body)
        if msg is None or content(msg).startswith("<tool_response>"):
            return
        values = inst.values
        if not values:
            return
        if inst is sess.open and inst.values and msg is not None:
            # a re-sent call message: cut its values again, they are the same bytes
            again = tpl.split(content(msg))
            if again is None:
                return                      # not this site's message: a new call
            values = again
        new = tpl.hint_join(tpl.render(values))
        if tpl.key not in self._relayout_seen:
            self._relayout_seen.add(tpl.key)
            print(f"[relayout] {tpl.key!r} {new[:160]!r}...", flush=True)
        set_content(msg, new)

    def _answer(self, body: dict, text: str) -> Any:
        """Native-tools responses must carry structured tool_calls; Qwen emits them as
        <tool_call>{...}</tool_call> blocks in the text. (byllm's InterceptorLLM uses
        the text tool protocol, so this stays inert there.)"""
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
        content_ = _TOOL_CALL.sub("", text).strip()
        return {"role": "assistant", "content": content_ or None, "tool_calls": calls}

    # ------------------------------------------------------------------ planning
    def _plan_session(self, sess: LiveSession) -> None:
        """One planning refresh: fan-out prediction -> deadline-stamped KV jobs,
        replacing whatever the session had queued (its branches may have collapsed)."""
        prog, planner = sess.program, self.planner
        cur = sess.open or (sess.calls[-1] if sess.calls else None)
        if prog is None or planner is None or cur is None:
            return
        calls = prog.predict_tree(cur.key, p_min=P_MIN)
        sess.predicted = calls[0].key if calls else None
        anchor = sess.plan_anchor or time.monotonic()
        jobs: List[Job] = []
        for c in calls:
            jobs.extend(self._plan_call(sess, c, anchor))
        planner.submit(sess.id, sess.epoch, jobs)

    def _plan_call(self, sess: LiveSession, c: PredictedCall, anchor: float) -> List[Job]:
        """For each call we expect: preload the whole rebuilt prompt when the session
        already carries every value, else the longest rebuildable prefix, else the
        static head."""
        prog = sess.program
        tpl = prog.sites.get(c.key) if prog else None
        if prog is None or tpl is None:
            return []
        cur = sess.open or (sess.calls[-1] if sess.calls else None)
        via = prog.edges.get((cur.key, c.key)) if cur is not None else None
        text, complete = self._resolve(sess, tpl, via)
        now = time.monotonic()
        deadline = max(now, anchor + c.t50 - KVPlanner.SAFETY_K * (c.t90 - c.t50))
        t90 = max(deadline, anchor + c.t90)
        if complete:
            toks = self._planned_tokens(tpl, text, True)
        else:
            cut = self._planned_tokens(tpl, text, False) if text else None
            static = self._prefix_tokens(tpl)
            toks = cut if cut is not None and len(cut) > len(static) else static
        if toks is None or len(toks) < 16:   # shorter than a cache block: nothing to gain
            return []
        unc, host = self.engine.cost(toks)
        # promote: a host-resident prefix to load back ahead of the call; hold: the
        # rest must be computed by the call itself (or is already resident), the job
        # only carries eviction protection.
        kind = "promote" if unc <= self.engine._stride and host > 0 else "hold"
        resident = unc + host < self.engine._stride
        return [Job(key=f"{sess.id}|{kind}|{c.key!r}", sid=sess.id, epoch=sess.epoch,
                    site=repr(c.key), kind=kind, toks=toks, p=c.p,
                    value=c.p * (unc + 0.8 * host), deadline=deadline, t90=t90,
                    host=host, done_upto=len(toks) if resident else 0,
                    state="done" if resident else "queued")]

    # ------------------------------------------------------------------ speculation
    def _resolve(self, sess: LiveSession, tpl: PromptTemplate, via: Optional[Edge] = None,
                 upto: Optional[CallInstance] = None) -> Tuple[str, bool]:
        """The head of `tpl`'s next user message that the session can already write,
        in the served layout, and whether it is the whole message. Walks the served
        order: a CONST is its literal, a COPY/TAKE the bytes an earlier call
        carried, an EXTEND the earlier bytes minus the closing delimiter (a known
        head, open tail) unless the edge `via` (the transition being predicted)
        writes nothing to its field, a RESP the field of an earlier reply rendered
        as the program's object; the first value with no such origin ends the head.
        `upto`: resolve as of the moment before that call (its own values excluded)."""
        lines: List[str] = []
        complete = True
        if tpl.header and not tpl.header_last:
            lines.append(tpl.header)
        for name in tpl.served_order():
            b = tpl.binding(name)
            v, prefix = self._value_of(sess, tpl, b, via, upto)
            if v is None:
                if prefix is not None:
                    lines.append(f"{b.label}{prefix}")
                complete = False
                break
            lines.append(f"{b.label}{v}")
        if complete:
            for b in tpl.bindings:
                if b.kind is BindingKind.SELF:
                    v, _ = self._value_of(sess, tpl, b, via, upto)
                    if v is None:
                        complete = False
                        break
                    lines.append("")
                    lines.append(f"{b.label}{v}{b.tail}")
        if complete and tpl.header and tpl.header_last:
            lines.append(tpl.header)
        text = "\n".join(lines)
        return (tpl.hint_join(text) if complete else text), complete

    def _value_of(self, sess: LiveSession, tpl: PromptTemplate, b: Binding, via: Optional[Edge] = None,
                  upto: Optional[CallInstance] = None) -> Tuple[Optional[str], Optional[str]]:
        """(value, known prefix) of a binding for the site's next call, from the
        part of the session's history the binding's scope still allows: a value
        reset by an ability is looked up only in calls from that ability's latest
        run onward (the call inside it carries the post-reset bytes)."""
        H = Heterogeneity
        if via is not None and b.name in via.overrides:
            ov = via.overrides[b.name]      # how this binding resolves along the predicted edge
            b = Binding(name=b.name, heterogeneity=ov.heterogeneity, kind=b.kind, source=ov.source,
                        literal=ov.literal, field=b.field, scope=b.scope, label=b.label, tail=b.tail)
        if b.heterogeneity is H.CONST:
            return b.literal, None
        hist = self._in_scope(sess, b, upto)
        if b.kind is BindingKind.SELF:      # the node revisited: the same object
            return self._latest(hist, tpl.key, b.name), None
        if b.heterogeneity in (H.COPY, H.TAKE, H.EXTEND):
            v = None
            if b.field:                     # the walker field's latest bytes, whichever call carried them
                v = self._latest_field(hist, b.field)
            if v is None and b.source is not None:
                v = self._latest(hist, b.source[0], b.source[1])
            if v is None:
                v = self._latest(hist, None, b.name)
            if v is None:
                return None, None
            untouched = via is not None and bool(b.field) and b.field not in via.writes
            if b.heterogeneity is H.EXTEND and not untouched:
                return None, (v[:-1] if len(v) >= 2 else None)
            if b.heterogeneity in (H.COPY, H.TAKE) and via is not None and b.field and b.field in via.writes:
                return None, None           # rewritten on this edge: the earlier bytes are stale
            return v, None
        if b.heterogeneity is H.RESP and b.source is not None:
            site, path = b.source
            if path.endswith("[]"):
                return None, None           # one element of a reply list: which one is the next call's is not known
            src_tpl = sess.program.sites.get(site) if sess.program else None
            for inst in reversed(hist):
                if inst.key == site and inst.response:
                    try:
                        obj = json.loads(inst.response)
                    except Exception:
                        return None, None
                    spec = src_tpl.resp_spec if src_tpl is not None else None
                    if not path:
                        return render_repr(obj, spec), None          # the whole reply, as the program's object repr
                    if isinstance(obj, dict) and path in obj:
                        return render_repr(obj[path], field_spec(spec, path) or {"k": "prim", "t": "any"}), None
                    return None, None
        return None, None

    @staticmethod
    def _in_scope(sess: LiveSession, b: Binding, upto: Optional[CallInstance] = None) -> List[CallInstance]:
        """The session's calls from the latest run of a scope-resetting ability on
        (before `upto` when given)."""
        hist = sess.history
        if upto is not None and upto in hist:
            hist = hist[:hist.index(upto)]
        if not b.scope:
            return hist
        resets = set(b.scope.split("|"))
        for i in range(len(hist) - 1, -1, -1):
            if hist[i].template.ability in resets:
                return hist[i:]
        return hist

    def _forward_head(self, sess: LiveSession, inst: CallInstance) -> str:
        """The longest leading part of a served call message that some successor's
        prompt repeats byte for byte: the bindings, in served order, whose walker
        field the successor carries in the same position and the program does not
        write on the edge to it (a CONST counts when the successor has the same
        literal there)."""
        prog, t = sess.program, inst.template
        if prog is None or not inst.values:
            return ""
        best: List[str] = []
        mine = [t.binding(n) for n in t.served_order()]
        for e in prog.successors(inst.key):
            nxt = prog.sites.get(e.dst)
            if nxt is None:
                continue
            theirs = [nxt.binding(n) for n in nxt.served_order()]
            lines: List[str] = []
            if t.header and not t.header_last:
                if not (nxt.header == t.header and not nxt.header_last):
                    continue
                lines.append(t.header)
            for a, b in zip(mine, theirs):
                same_field = bool(a.field) and a.field == b.field and a.field not in e.writes
                same_const = (a.heterogeneity is Heterogeneity.CONST and b.heterogeneity is Heterogeneity.CONST
                              and a.literal == b.literal)
                if not (same_field or same_const) or a.name not in inst.values:
                    break
                lines.append(f"{a.label}{inst.values[a.name]}")
            if len(lines) > len(best):
                best = lines
        return "\n".join(best)

    def _invalidate(self, sess: LiveSession, tpl: PromptTemplate) -> None:
        """Runtime value invalidation: the arriving call's ability resets some walker
        fields, so every earlier served prompt of the session is dead from the first
        binding that carried one of them. That tail is demoted in the KV cache (the
        surviving head, e.g. the module, stays in the normal band)."""
        prog = sess.program
        if prog is None or self.planner is None:
            return
        file = next((f for f, p in self.programs.items() if p is prog), "")
        dead = self._resets.get(file, {}).get(tpl.ability)
        if not dead:
            return
        for inst in sess.calls:
            if not inst.served_ids or not inst.values:
                continue
            t = inst.template
            lines: List[str] = []
            if t.header and not t.header_last:
                lines.append(t.header)
            cut_here = False
            for name in t.served_order():
                b = t.binding(name)
                base = b.field[:-2] if b.field.endswith("[]") else b.field
                if base and base in dead:
                    cut_here = True
                    break
                if name in inst.values:
                    lines.append(f"{b.label}{inst.values[name]}")
            if not cut_here:
                continue                    # nothing this call carried dies here
            text = "\n".join(lines)
            keep = len(self._planned_tokens(t, text, False) or []) if text else len(self._prefix_tokens(t))
            self.planner.invalidate(sess.id, inst.served_ids, keep)
        print(f"[invalidate] {sess.id} {tpl.ability} resets {sorted(dead)}", flush=True)

    @staticmethod
    def _latest_field(hist: List[CallInstance], field: str) -> Optional[str]:
        for inst in reversed(hist):
            for bb in inst.template.params:
                if bb.field == field and bb.name in inst.values:
                    return inst.values[bb.name]
        return None

    @staticmethod
    def _latest(hist: List[CallInstance], site: Optional[CallSiteID], name: str) -> Optional[str]:
        for inst in reversed(hist):
            if (site is None or inst.key == site) and name in inst.values:
                return inst.values[name]
        return None

    def _planned_tokens(self, tpl: PromptTemplate, text: str, complete: bool) -> Optional[List[int]]:
        """Token ids of the prompt a predicted call would send: the whole rendered
        prompt when `text` is the complete user message, else the rendered prefix up
        to the end of `text` minus a possibly split last token. Memoized."""
        key = (tpl.key, complete, text)
        hit = self._plan_tok.get(key, ...)
        if hit is not ...:
            self._plan_tok.move_to_end(key)
            return hit
        sysmsg = {"role": "system", "content": tpl.system_prompt}
        rendered = self.engine.render([sysmsg, {"role": "user", "content": text}], tools=tpl.tool_schema)
        if complete:
            toks: Optional[List[int]] = self.engine.tokenize(rendered)
        else:
            i = rendered.rfind(text)
            toks = self.engine.tokenize(rendered[:i + len(text)])[:-1] if i >= 0 else None
        self._plan_tok[key] = toks
        if len(self._plan_tok) > self.PLAN_TOK_CACHE:
            self._plan_tok.popitem(last=False)
        return toks

    def _prefix_tokens(self, tpl: PromptTemplate) -> List[int]:
        """Token ids of the site's static prompt head, rendered as a real request would be."""
        stable = tpl.fixed_head()
        cached = self._prefix_tok.get(tpl.key)
        if cached is not None and cached[0] == stable:
            return cached[1]
        sysmsg = {"role": "system", "content": tpl.system_prompt}
        if stable:
            text = self.engine.render([sysmsg, {"role": "user", "content": stable}], tools=tpl.tool_schema)
            i = text.rfind(stable)
            prefix = text[:i + len(stable)] if i >= 0 else ""
        else:
            # tool schemas follow the system content in the template: the shared byte
            # prefix is found by diffing two renders rather than by searching
            a = self.engine.render([sysmsg], tools=tpl.tool_schema)
            b = self.engine.render([sysmsg, {"role": "user", "content": "\x00"}], tools=tpl.tool_schema)
            prefix = os.path.commonprefix([a, b])
        toks = self.engine.tokenize(prefix)[:-1] if prefix else []
        self._prefix_tok[tpl.key] = (stable, toks)
        return toks

    # ------------------------------------------------------------------ session end
    def _finalize(self, sid: str) -> bool:
        """Session end: close the last call, retire the session's private cache."""
        sess = self.sessions.pop(sid, None)
        if sess is None:
            return False
        if self.planner is not None:
            self.planner.drop_session(sid)
        self._close_call(sess)
        print(f"[close] {sid} calls={len(sess.calls)}" + (f" opaque={sess.opaque}" if sess.opaque else ""), flush=True)
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
        kwargs["hicache_io_backend"] = hicache_io
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
                    help="baseline: no planner, sglang's own LRU eviction (re-layout stays on)")
    ap.add_argument("--kv", type=int, default=None, metavar="N", help="device KV pool cap in tokens")
    ap.add_argument("--host", type=int, default=None, metavar="GB", help="host KV tier size in GB")
    ap.add_argument("--no-header-last", action="store_true",
                    help="re-layout reorders bindings only; never move the call site header behind the values")
    ap.add_argument("--no-relayout", action="store_true",
                    help="serve call messages as they arrive (raw SGLang over the registered graph)")
    ap.add_argument("--hicache-io", choices=["direct", "kernel"], default="kernel",
                    help="HiCache host<->device copy backend (default: kernel)")
    ap.add_argument("--sched", choices=["fcfs", "lpm", "risk"], default="fcfs",
                    help="engine waiting-queue order: fcfs + request priorities, longest-prefix-match, "
                         "or cache-risk (cached tokens at risk first, FCFS otherwise)")
    ap.add_argument("--risk-aging-s", type=float, default=10.0, metavar="S",
                    help="--sched risk: an uncached request overtakes a fully cached one after waiting S s longer")
    a = ap.parse_args()
    if a.no_header_last:
        HEADER_LAST = False
    asyncio.run(main(a.model, plan=not a.lru, kv_tokens=a.kv,
                     host_gb=a.host, hicache_io=a.hicache_io, enable_relayout=not a.no_relayout,
                     sched=a.sched, risk_aging_s=a.risk_aging_s))
