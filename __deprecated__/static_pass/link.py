"""Wire to InterceptorLLM clients: 4-byte big-endian length + JSON frames."""
import asyncio
import json
import struct
from typing import Awaitable, Callable


class Connection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader, self._writer = reader, writer

    def send(self, frame: dict) -> None:
        data = json.dumps(frame).encode("utf-8")
        self._writer.write(struct.pack(">I", len(data)) + data)

    async def recv(self) -> dict:
        length = struct.unpack(">I", await self._reader.readexactly(4))[0]
        return json.loads(await self._reader.readexactly(length))

    def close(self) -> None:
        self._writer.close()


async def listen(host: str, port: int,
                 on_client: Callable[[Connection, dict], Awaitable[None]]) -> asyncio.AbstractServer:
    """Accept clients; each one registers, then `on_client(conn, hello)` serves it until it drops."""
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = Connection(reader, writer)
        try:
            hello = await conn.recv()
            if hello.get("type") != "register":
                conn.send({"type": "error", "error": f"expected register, got {hello.get('type')!r}"})
                return
            await on_client(conn, hello)
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError):
            pass
        finally:
            conn.close()
    return await asyncio.start_server(serve, host, port)
