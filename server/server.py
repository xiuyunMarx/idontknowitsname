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
from decompiler.primitives import (Callsite, CallObservation, Program, VisitByCallsite, _lcp,
                                   chosen_candidates, node_type)
from model.model import Engine
from server.http_server import HttpServer, PendingRequest

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

REBUILD_AT = 8         # rebuild the state machine every 8 sessions
MAX_BATCH = 32
    
class Controller:
    def __init__(self, model: str, server: HttpServer, speculate: bool = True, **engine_kwargs):
        self.speculate = speculate     # off: plain serving over the prefix cache (the baseline)
        self.engine:Engine = Engine(model, **engine_kwargs)
        self.sessions: Dict[str, LiveSession] = {}    # session id -> live state
        self.programs: Dict[str, Program] = {}        # entry callsite key -> Program
        self.server = server
        self.pool = server.pool
        self.warm: Dict[str, float] = {}              # site key -> last static-prefix touch (monotonic)
        self._prefix_tok: Dict[str, Tuple[str, List[int]]] = {}  # site key -> (stable_prefix, token ids)
        self._spec_inflight: set = set()              # dedup keys of queued/running speculation

    
    async def start_serving(self) -> None:
        # prepare
        await self.engine.warmup()
        profile = f"model/{self.engine.name.replace('/', '--')}.profile.json"
        if not self.engine.load_profile(profile):
            await self.engine._profile()
            self.engine.save_profile(profile)
            
            
        while True:
            batch: List[PendingRequest] = []
            req = await self.pool.get()          # blocks; yields to the event loop
            while True:
                if req.kind == "close":
                    self._finalize(req.session)
                else:
                    batch.append(req)
                if len(batch) >= MAX_BATCH:
                    break
                try:
                    req = self.pool.get_nowait()  # drain whatever else arrived
                except asyncio.QueueEmpty:
                    break

            if batch:
                await self._step(batch)           # your batched forward pass
        

    async def _step(self, batch:List[PendingRequest]) -> None:
        # init sessions
        for req in batch:
            s = self.sessions.get(req.session, None)
            if s is None:
                s = LiveSession(id=req.session)
                self.sessions[req.session] = s
        
        # get futures 
        for req in batch:
            session = self.sessions[req.session]
            
    def _finalize(self, sid: str) -> bool:
        """Session end, Fold the session's observations into its Program and drop it."""
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