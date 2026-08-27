"""The wire between one Program and one running InterceptorLLM instance.

The Program is the server end: it listens on (host, port), accepts the single
client the InterceptorLLM opens, answers its `register` with the observation
watch set, and then runs one reader loop that hands every frame to
`Program.on_frame`. Framing is the InterceptorLLM's: 4-byte big-endian length +
UTF-8 JSON (jaclang/byllm/llm.impl/interceptorLLM.impl.jac).

    client -> {"type": "register", "program_name", "pid", "model_name"}
    server -> {"type": "registered", "route_layout": "default"|"cache", "watch": {...}}
    client -> {"type": "state" | "enter", ...}                    # any time, observation
    client -> {"type": "call", "id", "key", "site", "pid", "model_name", "args", "self",
               "schema", "call_params"}
    server -> {"type": "tool_call", "call", "name", "arguments", "text"}   # 0..N
    client -> {"type": "tool_result", "call", "content"}
    server -> {"type": "final", "call", "output", "text"}
    client -> {"type": "reject", "call", "feedback"}
    client -> {"type": "generate", "id", "key", "site", "pid", "model_name", "messages",
               "schema", "temperature", "max_tokens", "stop"}
    server -> {"type": "result", "id", "text"}
    server -> {"type": "error", "id"|"call", "error"}
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from static_pass.primitives import Program


class InstanceLink:
    def __init__(self, program: "Program", host: str, port: int) -> None:
        self.program = program
        self.host, self.port = host, port
        self._srv: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.closed = threading.Event()
        self.pid: int = -1
        self.model_name: str = ""
        self.program_name: str = ""

    # -------------------------------------------------------------- lifecycle

    def listen(self) -> "InstanceLink":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        self.port = srv.getsockname()[1]  # a 0 port resolves here
        self._srv = srv
        self._thread = threading.Thread(target=self._serve, name=f"link-{self.program.name}", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        for s in (self._conn, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._conn = self._srv = None
        self.closed.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the client hung up (or `timeout`); True when it did."""
        return self.closed.wait(timeout)

    # -------------------------------------------------------------- transport

    def send(self, frame: dict) -> None:
        data = json.dumps(frame).encode("utf-8")
        with self._send_lock:
            if self._conn is None:
                raise ConnectionError("no client connected")
            self._conn.sendall(struct.pack(">I", len(data)) + data)

    def _recv(self) -> dict:
        assert self._conn is not None
        length = struct.unpack(">I", self._recv_exact(4))[0]
        return json.loads(self._recv_exact(length).decode("utf-8"))

    def _recv_exact(self, n: int) -> bytes:
        assert self._conn is not None
        buf = b""
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client closed the connection")
            buf += chunk
        return buf

    def _serve(self) -> None:
        assert self._srv is not None
        try:
            conn, _ = self._srv.accept()
        except OSError:
            self.closed.set()
            return
        self._conn = conn
        try:
            hello = self._recv()
            if hello.get("type") != "register":
                self.send({"type": "error", "error": f"expected register, got {hello.get('type')!r}"})
                return
            self.pid = int(hello.get("pid", -1))
            self.model_name = str(hello.get("model_name") or "")
            self.program_name = str(hello.get("program_name") or "")
            self.send(self.program.on_register(hello))
            while True:
                self.program.on_frame(self._recv())
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            self.program.on_disconnect()
            self.close()
