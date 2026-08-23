"""Dedicated TCP receiver for one InterceptorLLM client (asyncio).

One InterceptorLLMBackend serves exactly one Jac program; multi-tenancy lives a
layer above (one backend per program), so every message still carries the
tenant identity (program_name, pid).

Wire protocol (must match jaclang/byllm/llm.impl/interceptorLLM.impl.jac):
every message is a 4-byte big-endian length followed by a UTF-8 JSON body.

    client -> {"type": "register", "program_name", "pid", "model_name"}
    client -> {"type": "call", "id", "key", "site": "file.jac:line"|null,
               "program_name", "pid", "args": {name: repr}, "self": repr|null,
               "call_params"}
    server -> {"type": "tool_call", "call": <call id>, "name", "arguments", "text"}
    client -> {"type": "tool_result", "call", "content"}
    server -> {"type": "final", "call", "output", "text"}
    client -> {"type": "reject", "call", "feedback"}     # final failed typed parse; regenerate
    client -> {"type": "generate", "id", "key", "messages", ...}   # visit routing, single turn
    server -> {"type": "result", "id", "text"}

The backend is a pure transport endpoint: it parses every frame into a typed
`byLLMRequest` (bad frames are answered with an error frame and never reach the
queue), validates `register`, then puts `(backend, request)` on the shared
queue and goes back to reading.
The guard server consumes the queue, keeps the per-call conversation state
(keyed by backend + call id — a `tool_result`/`reject` arrives as its own queue
event), and replies through `await backend.send(frame)`; a `generate` reply
must echo the request's `id`. All of it on one event loop:

    queue: asyncio.Queue = asyncio.Queue()
    backends = [InterceptorLLMBackend(name, queue, comm_port=port), ...]
    await asyncio.gather(*(b.listen() for b in backends), guard.run(queue))

Blocking work (a sync vLLM generate) must go through run_in_executor or an
async engine, or it freezes every backend on the loop.
"""

import asyncio
import json
import struct

from console_helper.debug_output import console_debug, console_log, console_warn, console_error
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Required fields per client frame type ("" and {} count as present, None as missing).
_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "register": ("program_name", "model_name"),
    "call": ("id", "key", "args"),
    "tool_result": ("call", "content"),
    "reject": ("call", "feedback"),
    "generate": ("id", "key", "messages"),
}

# Wire layout per frame type, in protocol order ("self" maps to nest_scope_desc).
_WIRE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "register": ("program_name", "pid", "model_name"),
    "call": ("id", "key", "site", "program_name", "pid", "args", "self", "call_params"),
    "tool_result": ("call", "content"),
    "reject": ("call", "feedback"),
    "generate": ("id", "key", "program_name", "pid", "messages", "schema", "temperature", "max_tokens", "stop"),
}


@dataclass
class ByLLMRequest:
    """One parsed client frame; what the guard server receives off the queue."""
    type: str
    pid: int  # -1 on frames that don't carry it (tool_result/reject)
    nest_scope_desc: Optional[str] = None  # wire "self": repr of the receiver object
    args: Optional[Dict[str, str]] = None  # {param name: repr}
    id: Optional[int] = None               # wire id of a call/generate
    call: Optional[int] = None             # tool_result/reject: id of the call they belong to
    key: Optional[str] = None              # callsite key: Owner.name / name ("" for visit routing)
    site: Optional[str] = None             # call only: invocation location "file.jac:line" (may be None)
    program_name: Optional[str] = None 
    model_name: Optional[str] = None       # register only
    call_params: Optional[Dict[str, Any]] = None
    content: Optional[str] = None          # tool_result
    feedback: Optional[str] = None         # reject
    messages: Optional[list] = None        # generate (visit routing): full byllm message list
    schema: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None

    @classmethod
    def from_frame(cls, msg: Dict[str, Any]) -> "ByLLMRequest":
        """Parse one wire dict; raises ValueError on unknown type or missing fields."""
        t = msg.get("type")
        if t not in _REQUIRED:
            raise ValueError(f"unknown client frame type: {t!r}")
        req = cls(
            type=t,
            pid=int(msg.get("pid") or -1),
            nest_scope_desc=msg.get("self"),
            args=msg.get("args"),
            id=msg.get("id"),
            call=msg.get("call"),
            key=msg.get("key"),
            site=msg.get("site"),
            program_name=msg.get("program_name"),
            model_name=msg.get("model_name"),
            call_params=msg.get("call_params"),
            content=msg.get("content"),
            feedback=msg.get("feedback"),
            messages=msg.get("messages"),
            schema=msg.get("schema"),
            temperature=msg.get("temperature"),
            max_tokens=msg.get("max_tokens"),
            stop=msg.get("stop"),
        )
        missing = [f for f in _REQUIRED[t] if getattr(req, f) is None]
        if missing:
            raise ValueError(f"{t} frame missing {missing}")
        return req

    def to_frame(self) -> Dict[str, Any]:
        """Wire dict for this request: exactly its type's protocol fields."""
        out: Dict[str, Any] = {"type": self.type}
        for f in _WIRE_FIELDS[self.type]:
            out[f] = self.nest_scope_desc if f == "self" else getattr(self, f)
        return out

class InterceptorLLMBackend:
    def __init__(self, program_name: str, queue: asyncio.Queue,
                 comm_ip: str = "localhost", comm_port: int = 8964) -> None:
        self.program_name = program_name
        self._queue = queue
        self._addr = (comm_ip, comm_port)
        self._server: asyncio.Server | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._fatal = False

    async def send(self, msg: dict) -> None:
        """Guard-server side of the interface: push one frame to the client."""
        data = json.dumps(msg).encode("utf-8")
        self._writer.write(struct.pack(">I", len(data)) + data)  #type: ignore
        await self._writer.drain()  #type: ignore

    async def recv(self) -> dict | None:
        """One framed message; None means the client closed the connection."""
        try:
            header = await self._reader.readexactly(4)  #type: ignore
            body = await self._reader.readexactly(struct.unpack(">I", header)[0])  #type: ignore
        except asyncio.IncompleteReadError:
            return None
        return json.loads(body.decode("utf-8"))

    async def listen(self) -> None:
        """Serve the dedicated client forever; re-accepts if the program restarts."""
        self._server = await asyncio.start_server(self._handle, self._addr[0], self._addr[1])
        console_log(f"InterceptorLLMBackend for {self.program_name!r} listening on {self._addr[0]}:{self._addr[1]}")
        async with self._server:
            try:
                await self._server.serve_forever()
            except asyncio.CancelledError:
                if not self._fatal:
                    raise  # external cancellation (shutdown), not ours

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._writer is not None:
            console_warn("Rejecting second connection: this backend is dedicated to one client")
            writer.close()
            return
        peer = writer.get_extra_info("peername")
        console_log(f"Accepted connection from {peer}")
        self._reader, self._writer = reader, writer
        try:
            await self._pump()
        except (ConnectionError, OSError) as e:
            console_warn(f"Client connection dropped: {e}")
        except RuntimeError:
            self._fatal = True  # program-name mismatch is fatal to this backend
            self._server.close()  #type: ignore
        finally:
            self._reader = self._writer = None
            writer.close()
            console_log(f"Client {peer} disconnected")

    async def _pump(self) -> None:
        """Read frames, parse them into ByLLMRequest, hand each to the guard server."""
        while True:
            msg = await self.recv()
            if msg is None:
                return
            try:
                req = ByLLMRequest.from_frame(msg)
            except ValueError as e:
                console_error(f"Bad frame from client: {e}")
                await self.send({"type": "error", "error": str(e), "id": msg.get("id")})
                continue
            if req.type == "register":
                if req.program_name != self.program_name:
                    console_error(f"Program name mismatch: expected {self.program_name!r}, got {req.program_name!r}")
                    raise RuntimeError()
                console_log(f"Registered {req.program_name!r} (pid {req.pid}, model {req.model_name!r})")
            else:
                console_debug(f"{req.type} from {req.program_name!r} pid {req.pid} key={req.key!r}")
            await self._queue.put((self, req))
