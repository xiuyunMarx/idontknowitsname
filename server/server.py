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
PLAN_DEPTH = 1         # predict the direct successors only
END_SURE = 0.95        # retire a session's cache at a reply when the program is this likely to end here
T_BASE = 120.0         # idle seconds before a session is finalized
T_SHORT = 15.0         # idle timeout once the session probably ended (P(end) >= 0.5 at its last call)
SWEEP_S = 5.0          # idle sweeper period
GRAMMAR_BACKEND = os.environ.get("GRAMMAR_BACKEND", "xgrammar")   # sglang grammar backend; "none" = no constrained decoding (schema not forwarded)
MAX_TOKENS = 4096      # decode cap when the request does not set one
MAX_BATCH = 32
HEADER_LAST = True     # --no-header-last clears it: layouts never move the header behind the values

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def json_schema_of(body: dict) -> Optional[str]:
    """The JSON schema a request's `response_format` asks the reply to follow
    (byllm sends one for every typed return), serialized for sglang's
    constrained decoding; None for free text or a bare json_object request."""
    rf = body.get("response_format")
    if not isinstance(rf, dict) or rf.get("type") != "json_schema":
        return None
    schema = (rf.get("json_schema") or {}).get("schema")
    return json.dumps(schema) if isinstance(schema, dict) else None


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
    field_valid_from: Dict[str, int] = field(default_factory=dict)  # field -> first history index in its live generation
    counters: Dict[int, int] = field(default_factory=dict)   # loop index -> passes through its head, for the latest call

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
        self._registered: Dict[str, str] = {}     # agent file -> the static payload it was registered with
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
                if (sess.program is not None and last is not None
                        and sess.program.end_prob(last.key, sess.counters) >= 0.5):
                    t = T_SHORT   # the program probably ended here: a shorter wait
                if now - sess.last_seen > t:
                    self._finalize(sid)

    # ------------------------------------------------------------------ registration
    def register(self, body: dict) -> dict:
        """Adopt a program's static analysis. Registering the same analysis again
        (every launch of the agent re-analyzes and re-registers) won't change already serving Program."""
        file = os.path.abspath(str(body["file"]))
        payload = json.dumps({"program": body["program"], "no_header_last": bool(body.get("no_header_last"))},
                             sort_keys=True)
        if self._registered.get(file) == payload and file in self.programs:
            known = self.programs[file]
            print(f"[program] {file}: unchanged, statistics kept", flush=True)
            return {"ok": True, "file": file, "sites": len(known.sites), "unchanged": True}
        program = program_from_dict(body["program"])
        if body.get("no_header_last"):          # the registrant's choice for this program (fact_check keeps its headers in front)
            for t in program.sites.values():
                t.no_header_last = True
        if self.enable_relayout:
            program.decide_layouts(header_last=HEADER_LAST)
        program.decide_loops()
        for i, lp in enumerate(program.loops):
            print(f"[loop] {i}: head {lp.head!r} body {sorted(repr(k) for k in lp.body)}"
                  + (f" in loop {lp.parent}" if lp.parent is not None else ""), flush=True)
        for t in program.sites.values():
            print(f"[layout] {t.key!r} order={t.order} header_last={t.header_last} "
                  f"bindings={[(b.name, b.heterogeneity.name) for b in t.bindings]}", flush=True)
        self.programs[file] = program
        self._registered[file] = payload
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
            schema = json_schema_of(body)
            if schema is not None and GRAMMAR_BACKEND != "none":
                sp["json_schema"] = schema         # constrained decoding
            turn = len(inst.engine_time) if inst is not None else 0
            rid = f"{sess.id}-{sess.epoch}t{turn}-{uuid.uuid4().hex[:8]}"
            ids = self.engine.tokenize(prompt)
            t0 = time.perf_counter()
            admit = self._served_head(sess, inst) if (self.planner is not None and inst is not None) else None
            text = await self.engine.generate(prompt, rid, sp, ids=ids,
                                              priority=PRIORITY_CONT if cont else PRIORITY_REAL,
                                              host_admit_len=admit)   # the volatile tail stays out of the host tier
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
        head = admit if admit is not None else self._served_head(sess, inst)
        self.planner.note_served(sess.id, ids, head, static_len=head)
        prog = sess.program
        if prog is not None and (len(prog.successors(tpl.key)) == 0 # No successor
                                 or prog.end_prob(tpl.key, sess.counters) >= END_SURE):
            self.planner.retire_session(sess.id) # session ends here, retire
        else:
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
        if values and self.enable_relayout:
            # a field's first value: its token count breaks layout ties within a class
            sizes = {b.field: len(self.engine.tokenize(values[b.name])) for b in tpl.params
                     if b.field and b.name in values and b.field not in prog.field_tokens}
            for k in prog.observe_sizes(sizes):
                t = prog.sites[k]
                print(f"[layout] {k!r} order={t.order} header_last={t.header_last} "
                      f"sized={ {f: prog.field_tokens[f] for f in sorted(sizes)} }", flush=True)
        inst = CallInstance(template=tpl, values=values, t_arrive=req.t_arrive, t_done=req.t_arrive)
        if values:
            self._relayout(sess, req.body, inst)
        hit = "" if sess.predicted is None else f" predicted={'hit' if sess.predicted == tpl.key else 'miss'}"
        prev = sess.calls[-1] if sess.calls else None
        via = prog.edges.get((prev.key, tpl.key)) if prev is not None else None
        if prev is not None:
            e = prog.add_edge(prev.key, tpl.key)      # the transition is known at arrival
            e.count += 1
            e.gap.append(max(0.0, req.t_arrive - prev.t_done))
            # the branch taken, under the iteration context prev was called in
            prog.observe_branch(prev.key, prog.context(prev.key, sess.counters), tpl.key)
        sess.counters = prog.step_counters(sess.counters, prev.key if prev is not None else None, tpl.key)
        print(f"[call] {sess.id} #{len(sess.calls)} {tpl.key!r}{hit} ctx={prog.context(tpl.key, sess.counters)}",
              flush=True)
        self._invalidate(sess, tpl, via)
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
        calls = prog.predict(cur=cur.key, counters=sess.counters, p_min=P_MIN)
        # calls = prog.predict_tree(cur.key, sess.counters, p_min=P_MIN, max_depth=PLAN_DEPTH)
        sess.predicted = calls[0].key if calls else None # used for logging hits/misses on the next call
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
        paths = c.paths or ([c.path] if c.path else [])
        via = self._paths_effect(prog, paths)
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

    @staticmethod
    def _path_effect(prog: Program, path: Tuple[CallSiteID, ...]) -> Optional[Edge]:
        """Compose the value-flow facts along a predicted callsite path.

        Writes and invalidations accumulate.  For each target binding, the latest
        definite override of its walker field is carried forward; an intervening
        unknown write stops the search.  This makes a multi-hop prediction obey
        the same flow constraints as a direct successor.
        """
        if len(path) < 2:
            return None
        edges: List[Edge] = []
        for src, dst in zip(path, path[1:]):
            e = prog.edges.get((src, dst))
            if e is None:
                return None
            edges.append(e)
        if len(edges) == 1:
            return edges[0]

        effect = Edge(src=path[0], dst=path[-1])
        effect.writes = frozenset().union(*(e.writes for e in edges))
        effect.invalidates = frozenset().union(*(e.invalidates for e in edges))
        target = prog.sites.get(path[-1])
        if target is None:
            return effect

        for b in target.params:
            base = b.field[:-2] if b.field.endswith("[]") else b.field
            if not base or b.field.endswith("[]"):
                continue
            for e in reversed(edges):
                candidates = [ov for ov in e.overrides.values()
                              if (ov.field[:-2] if ov.field.endswith("[]") else ov.field) == base]
                if candidates:
                    first = candidates[0]
                    origin = (first.heterogeneity, first.source, first.literal)
                    if all((ov.heterogeneity, ov.source, ov.literal) == origin for ov in candidates):
                        effect.overrides[b.name] = Binding(
                            name=b.name, heterogeneity=first.heterogeneity,
                            source=first.source, literal=first.literal, field=b.field)
                    break
                if base in e.writes:
                    break                       # latest write has no reconstructible origin
        return effect

    @staticmethod
    def _paths_effect(prog: Program, paths: List[Tuple[CallSiteID, ...]]) -> Optional[Edge]:
        """Conservatively merge the effects of every path reaching one callsite.

        A predicted call has cumulative probability across these paths, so its
        reconstructed prefix must be valid on all of them. Writes/invalidations
        therefore union, while a value override survives only when every path
        derives the same origin.
        """
        effects = [Controller._path_effect(prog, path) for path in paths]
        if not effects or any(e is None for e in effects):
            return None
        known = [e for e in effects if e is not None]
        if len(known) == 1:
            return known[0]
        merged = Edge(src=known[0].src, dst=known[0].dst)
        merged.writes = frozenset().union(*(e.writes for e in known))
        merged.invalidates = frozenset().union(*(e.invalidates for e in known))
        first = known[0].overrides
        merged.overrides = {
            name: ov for name, ov in first.items()
            if all(name in e.overrides
                   and (ov.heterogeneity, ov.source, ov.literal, ov.field)
                   == (e.overrides[name].heterogeneity, e.overrides[name].source,
                       e.overrides[name].literal, e.overrides[name].field)
                   for e in known[1:])
        }
        return merged

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
        ov = via.overrides.get(b.name) if via is not None else None
        if ov is not None:
            # How this binding resolves along the predicted path.  A definite
            # override is strict: if its source is not in history yet, do not
            # fall back to an older value of the same field.
            b = Binding(name=b.name, heterogeneity=ov.heterogeneity, kind=b.kind, source=ov.source,
                        literal=ov.literal, field=b.field, scope=b.scope, label=b.label, tail=b.tail)
        if b.heterogeneity is H.CONST:
            return b.literal, None
        hist = self._in_scope(sess, b, upto)
        if b.kind is BindingKind.SELF:      # the node revisited: the same object
            return self._latest(hist, tpl.key, b.name), None
        if b.heterogeneity in (H.COPY, H.TAKE, H.EXTEND):
            v = None
            if ov is not None and b.source is not None:
                v = self._latest(hist, b.source[0], b.source[1])
                if v is None:
                    return None, None
            elif b.field:                   # the walker field's latest bytes, whichever call carried them
                v = self._latest_field(hist, b.field)
            if v is None and b.source is not None:
                v = self._latest(hist, b.source[0], b.source[1])
            if v is None:
                v = self._latest(hist, None, b.name)
            if v is None:
                return None, None
            base = b.field[:-2] if b.field.endswith("[]") else b.field
            untouched = via is not None and bool(base) and base not in via.writes
            if b.heterogeneity is H.EXTEND and not untouched:
                if via is not None and base in via.invalidates:
                    return None, None       # reset/assignment: the old bytes are not a prefix
                return None, (v[:-1] if len(v) >= 2 else None)
            if (b.heterogeneity in (H.COPY, H.TAKE) and ov is None
                    and via is not None and base and base in via.writes):
                return None, None           # rewritten on this edge: the earlier bytes are stale
            return v, None
        if b.heterogeneity is H.RESP and b.source is not None:
            site, path = b.source
            if path.endswith("[]"):
                return None, None           # one element of a reply list: which one is the next call's is not known
            src_tpl = sess.program.sites.get(site) if sess.program else None
            for inst in reversed(hist):
                if inst.key != site:
                    continue
                if not inst.response:
                    return None, None        # latest producer is pending; never use a prior iteration's reply
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
        base = b.field[:-2] if b.field.endswith("[]") else b.field
        if base and base in sess.field_valid_from:
            return hist[sess.field_valid_from[base]:]
        if not b.scope:
            return hist
        resets = set(b.scope.split("|"))
        for i in range(len(hist) - 1, -1, -1):
            if hist[i].template.ability in resets:
                return hist[i:]
        return hist

    def _served_head(self, sess: LiveSession, inst: CallInstance) -> int:
        """Token length of the served prompt's reusable head: the site's static prefix,
        extended to the longest prefix some later call repeats (see _forward_head).
        Everything past it is the call's volatile tail."""
        tpl = inst.template
        head = len(self._prefix_tokens(tpl))
        fwd = self._forward_head(sess, inst)
        cut = self._planned_tokens(tpl, fwd, False) if fwd else None
        if cut is not None and len(cut) > head:
            head = len(cut)
        return head

    def _forward_head(self, sess: LiveSession, inst: CallInstance, max_hops: int = 3) -> str:
        """The longest leading part of a served call message that some later call's
        prompt repeats byte for byte: along every static path of up to `max_hops`
        edges from this site, the target's served layout is compared position by
        position, and a field survives only if no edge on the path writes it."""
        prog, t = sess.program, inst.template
        if prog is None or not inst.values:
            return ""
        best: List[str] = []
        mine = [t.binding(n) for n in t.served_order()]
        paths: List[Tuple[CallSiteID, ...]] = [(inst.key,)]
        for _ in range(max_hops):
            paths = [p + (e.dst,) for p in paths for e in prog.successors(p[-1])]
            for path in paths:
                effect = self._path_effect(prog, path)
                nxt = prog.sites.get(path[-1])
                if effect is None or nxt is None:
                    continue
                # These bytes precede every user binding in the rendered prompt.
                if t.system_prompt != nxt.system_prompt or t.tool_schema != nxt.tool_schema:
                    continue
                theirs = [nxt.binding(n) for n in nxt.served_order()]
                lines: List[str] = []
                if t.header and not t.header_last:
                    if not (nxt.header == t.header and not nxt.header_last):
                        continue
                    lines.append(t.header)
                elif nxt.header and not nxt.header_last:
                    continue                # successor has a leading header; current does not
                for a, b in zip(mine, theirs):
                    same_field = bool(a.field) and a.field == b.field
                    same_const = (a.heterogeneity is Heterogeneity.CONST and b.heterogeneity is Heterogeneity.CONST
                                  and a.literal == b.literal)
                    if a.label != b.label or not (same_field or same_const) or a.name not in inst.values:
                        break
                    if same_field and a.field in effect.writes:
                        # written on the way: an append-only field keeps its bytes minus the
                        # closing delimiter as a prefix of the next value; anything else is stale
                        v = inst.values[a.name]
                        if (b.heterogeneity is Heterogeneity.EXTEND and a.field not in effect.invalidates
                                and len(v) >= 2):
                            lines.append(f"{a.label}{v[:-1]}")
                        break
                    lines.append(f"{a.label}{inst.values[a.name]}")
                if len(lines) > len(best):
                    best = lines
            if not paths:
                break
        return "\n".join(best)

    def _invalidate(self, sess: LiveSession, tpl: PromptTemplate,
                    via: Optional[Edge]) -> None:
        """Invalidate fields that is killed on the actual incoming edge.

        Current-format static edges distinguish append-only writes from resets and
        arbitrary assignments.  An unknown/dynamic edge falls back to the older
        ability-level scope table.  History generation boundaries prevent stale
        values from being resolved, and matching KV tails are demoted.
        """
        prog = sess.program
        if prog is None or self.planner is None:
            return
        file = next((f for f, p in self.programs.items() if p is prog), "")
        dead = via.invalidates if via is not None else self._resets.get(file, {}).get(tpl.ability)
        if not dead:
            return
        boundary = len(sess.calls)             # the arriving/open call will occupy this index
        for name in dead:
            start = boundary
            floor = sess.field_valid_from.get(name, 0)
            if via is not None:
                origins = [ov for ov in via.overrides.values()
                           if (ov.field[:-2] if ov.field.endswith("[]") else ov.field) == name
                           and ov.source is not None]
                # A call whose reply/binding creates the new value belongs to the
                # new generation even though the assignment executes after it.
                for ov in origins:
                    for i in range(len(sess.calls) - 1, floor - 1, -1):
                        if sess.calls[i].key == ov.source[0]:  # type: ignore[index]
                            start = min(start, i)
                            break
            sess.field_valid_from[name] = start
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
        print(f"[invalidate] {sess.id} {tpl.ability} kills {sorted(dead)}", flush=True)

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
        """Token ids of the prompt a predicted call would send"""
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
        prog = sess.program
        if prog is not None and sess.calls:
            last = sess.calls[-1]                  # the session ended after this call
            prog.observe_branch(last.key, prog.context(last.key, sess.counters), None)
        print(f"[close] {sid} calls={len(sess.calls)}" + (f" opaque={sess.opaque}" if sess.opaque else ""), flush=True)
        return True


async def main(model: str = "Qwen/Qwen3-8B", port: int = 8964, plan: bool = True,
               kv_tokens: Optional[int] = None,
               host_gb: Optional[int] = None, hicache_io: Optional[str] = None,
               enable_relayout: bool = True, sched: str = "fcfs", risk_aging_s: float = 10.0,
               promote: bool = True, eviction: Optional[str] = None,
               engine_log: Optional[str] = None) -> None:
    server = HttpServer(port=port)
    await server.start()
    kwargs: Dict[str, Any] = {"context_length": 32768,
                              "radix_eviction_policy": eviction or ("priority" if plan else "lru"),
                              "grammar_backend": GRAMMAR_BACKEND}   # constrained decoding
    if engine_log:
        kwargs["log_level"] = engine_log        # "info": sglang's own batch logs (#running-req, #queue-req, throughput)
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
    if ctrl.planner is not None:
        ctrl.planner.promote_enabled = promote #type: ignore[assignment]
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
    ap.add_argument("--no-promote", action="store_true",
                    help="planner steers eviction only: no speculative host->device loads")
    ap.add_argument("--eviction", choices=["lru", "priority"], default=None,
                    help="engine eviction policy override (default: priority with the planner, lru without)")
    ap.add_argument("--engine-log", choices=["info", "warning", "error"], default=None,
                    help="sglang log level; info prints the scheduler's batch statistics")
    ap.add_argument("--risk-aging-s", type=float, default=10.0, metavar="S",
                    help="--sched risk: an uncached request overtakes a fully cached one after waiting S s longer")
    a = ap.parse_args()
    if a.no_header_last:
        HEADER_LAST = False
    asyncio.run(main(a.model, plan=not a.lru, kv_tokens=a.kv,
                     host_gb=a.host, hicache_io=a.hicache_io, enable_relayout=not a.no_relayout,
                     sched=a.sched, risk_aging_s=a.risk_aging_s, promote=not a.no_promote, eviction=a.eviction,
                     engine_log=a.engine_log))
