"""One live client process ("program:pid", the OpenAI `user` field) as the controller sees it.

Replaces the old ProgramInstance: the client now owns prompts and the ReAct loop, so all
that is kept here is what the planner needs — the callsite history (folded into the
program's statistics as it grows), the current prediction, and whether the client is
mid-call (between two ReAct turns, running a tool)."""
import time
from typing import Any, Dict, List, Optional, Tuple

from trace_extractor.parser import SessionBuilder, parse_request
from trace_extractor.primitives import (CallInstance, CallsiteType, LLMCallsite, Program, TraceRecord)


class Session:
    def __init__(self, session_id: str, program, max_chain: int = 12) -> None:
        """`program` is a Program, or a resolver Callable[[LLMCallsite], Program] invoked
        with the session's first callsite — FaaS: the wire carries no program identity,
        so which program a session runs is discovered from what it asks for."""
        self.id = session_id
        self._resolve = program if callable(program) else None
        self.program: Optional[Program] = None if callable(program) else program
        self.max_chain = max_chain
        self.builder = SessionBuilder(session_id, self.program.name if self.program else "")
        self._active = self.program.begin_session() if self.program else []
        self.predicted: List[Tuple[str, str]] = []            # [(callsite key, model)] after the current call
        self.open_call: Optional[Tuple[str, str]] = None      # (key, model) while the client runs a tool
        self.inflight = 0                                     # requests submitted, not yet answered
        self.last_seen = time.perf_counter()                  # last response sent (idle clock)
        self.closed = False

    @property
    def calls(self) -> List[CallInstance]:
        return self.builder.session.calls

    @property
    def last_key(self) -> Optional[str]:
        return self.calls[-1].key if self.calls else None

    def salt(self) -> str:
        return self.id

    def begin(self, body: Dict[str, Any], t_arrive: float) -> Tuple[LLMCallsite, CallInstance, str, int]:
        """A request arrived: identify its callsite, extend the history, predict what follows.
        Returns (callsite, its instance, kind for the cost model, turn index)."""
        site, call = parse_request(TraceRecord(session=self.id, t_arrive=t_arrive, t_done=0.0, request=body))
        if self.program is None:  # first request: its callsite names the program
            self.program = self._resolve(site)
            self.builder.session.program = self.program.name
            self._active = self.program.begin_session()
        site = self.program.add_callsite(site)
        before = list(self.calls)
        inst, new = self.builder.step(call)
        if new:
            if before:  # the previous instance is complete now that its successor arrived
                self.program.observe_times(before[-1], inst)
            self._active = self.program.extend(self._active, inst.key)
        turn = len(inst.turns) - 1
        kind = "generate" if site.kind is CallsiteType.VISITBY else ("call" if turn == 0 else "tool_turn")
        self.predicted = self.program.predict_models([c.key for c in self.calls], self.max_chain)
        self.open_call = None
        self.inflight += 1
        return site, inst, kind, turn

    def finish(self, site: LLMCallsite, inst: CallInstance, text: str, t_done: float,
               engine_s: float = 0.0) -> None:
        """The reply went out: close the turn; if it was a tool call the client is now mid-call."""
        a, _ = inst.turns[-1]
        inst.turns[-1] = (a, t_done)
        inst.t_done = t_done
        inst.result = text
        inst.engine_s += max(0.0, engine_s)
        self.inflight = max(0, self.inflight - 1)
        self.last_seen = time.perf_counter()
        if site.kind is CallsiteType.TOOL and "<tool_call>" in text and "finish_tool" not in text:
            self.open_call = (site.key, site.model)

    def close(self) -> None:
        """The client process is gone (told us, or idle too long): fold the last call in."""
        if self.closed:
            return
        self.closed = True
        self.open_call = None
        self.predicted = []
        if self.program is None:
            return
        if self.calls:
            self.program.observe_times(self.calls[-1], None)
        self.program.finish_session(self._active, self.builder.session)

    def describe(self) -> str:
        if self.program is None:
            return f"{self.id}: (no requests yet)"
        return (f"{self.id}: " + " -> ".join(self.program.label(c.key) for c in self.calls)
                + (f" [open {self.program.label(self.open_call[0])}]" if self.open_call else "")
                + f" predicted={[m.split('/')[-1] for _, m in self.predicted]}")
