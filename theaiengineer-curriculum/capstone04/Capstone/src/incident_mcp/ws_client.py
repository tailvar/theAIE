from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

import websockets

from .telemetry import Telemetry


class WebSocketJSONRPC:
    """
    JSON-RPC over WebSocket.
    Mirrors StdioJSONRPC but uses a WebSocket endpoint
    instead of subprocess pipes.
    """

    def __init__(self, endpoint: str, telemetry: Telemetry) -> None:
        self.endpoint = endpoint
        self.telemetry = telemetry

    async def request_async(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        self.telemetry.trace("out", msg)

        async with websockets.connect(self.endpoint) as ws:
            await ws.send(json.dumps(msg))
            raw = await ws.recv()

        resp = json.loads(raw)
        self.telemetry.trace("in", resp)

        if resp.get("id") != req_id:
            raise RuntimeError(f"Correlation mismatch: sent id={req_id}, got id={resp.get('id')}")

        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error: {resp['error']}")

        return resp["result"]

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Simple sync wrapper so the rest of agent code doesn't change.
        return asyncio.run(self.request_async(method, params))
