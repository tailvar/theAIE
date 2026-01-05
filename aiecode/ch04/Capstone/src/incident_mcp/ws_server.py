from __future__ import annotations

import asyncio
import json
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol
from __future__ import annotations

import asyncio
import json
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol

from .server import MCPStdioServer


class MCPWebSocketServer:
    """
    Wraps an existing MCPStdioServer (its handle(msg) method) with a WebSocket transport.

    Transport responsibility:
      - receive JSON messages over WS
      - call server.handle(msg)
      - send JSON response over WS
    """

    def __init__(self, core: MCPStdioServer) -> None:
        self.core = core

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        # IMPORTANT: never print to stdout from here if stdout is being used elsewhere.
        # In WS mode it doesn't matter, but keep the discipline.
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                # JSON-RPC parse error response
                resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                await ws.send(json.dumps(resp))
                continue

            resp = self.core.handle(msg)
            await ws.send(json.dumps(resp))

    async def serve(self, host: str = "0.0.0.0", port: int = 8765, path: str = "/mcp") -> None:
        # websockets.serve doesn't directly enforce a path by default;
        # clients can connect to any path on that port.
        # Check ws.path inside handler if want strictness.
        async with websockets.serve(self._handler, host, port):
            await asyncio.Future()  # run forever


def main() -> None:
    # Reuse existing server core (loads config/resources/tools, etc.)
    core = MCPStdioServer()

    ws_server = MCPWebSocketServer(core)
    asyncio.run(ws_server.serve(host="0.0.0.0", port=8765))


if __name__ == "__main__":
    main()




