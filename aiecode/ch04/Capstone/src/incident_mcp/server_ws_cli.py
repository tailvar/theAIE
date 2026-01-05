from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState

from .server import MCPStdioServer


def _dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def build_app(server: MCPStdioServer) -> FastAPI:
    """
    FastAPI app exposing JSON-RPC 2.0 over WebSockets.

    Contract:
      - Client sends ONE JSON-RPC object per WebSocket text message.
      - Server replies with ONE JSON-RPC response object per message.
    """
    app = FastAPI(title="incident-mcp-ws")

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        try:
            while True:
                raw = await ws.receive_text()

                # Parse request
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("JSON-RPC message must be an object")
                except Exception as e:
                    # Cannot reliably read id if parse fails -> id=None
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                            "data": {"detail": str(e)},
                        },
                    }
                    await ws.send_text(_dumps(err_resp))
                    continue

                # Handle via deterministic server logic
                resp = server.handle(msg)

                # Reply
                await ws.send_text(_dumps(resp))

        except Exception:
            # Client disconnected, socket closed, etc.
            # Avoid noisy stack traces; just exit handler.
            if ws.client_state != WebSocketState.DISCONNECTED:
                try:
                    await ws.close()
                except Exception:
                    pass
            return

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Incident MCP server over WebSockets (JSON-RPC 2.0)")
    default_root = Path(os.getenv("INCIDENT_MCP_ROOT", Path(__file__).resolve().parents[2]))

    parser.add_argument(
        "--root",
        default=str(default_root),
        help="Project root containing config/ and data/ (also supports env INCIDENT_MCP_ROOT).",
    )
    parser.add_argument("--host", default=os.getenv("MCP_WS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_WS_PORT", "8000")))
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Disable the server banner printed to stderr.",
    )

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    # Human logs to stderr (good discipline)
    if not args.no_banner:
        print("incident-mcp-ws server running (JSON-RPC over WebSockets)", file=sys.stderr)
        print(f"root: {root}", file=sys.stderr)
        print(f"ws: ws://{args.host}:{args.port}/ws", file=sys.stderr)

    # Build your existing server (same tools/resources/memory paths)
    server = MCPStdioServer(root_dir=root, stderr_banner=not args.no_banner)

    # Wrap server.handle(msg)
    app = build_app(server)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()



