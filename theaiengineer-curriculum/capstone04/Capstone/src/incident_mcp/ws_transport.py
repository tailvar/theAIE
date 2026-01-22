from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from .telemetry import Telemetry


class WebSocketJSONRPC:
    """
    JSON-RPC 2.0 over WebSockets (one request -> one response).

    Transport contract matches StdioJSONRPC.request():
      - request(method, params) returns resp["result"] or raises on error
      - telemetry traces out/in messages
      - correlation check on id
    """

    def __init__(self, ws_url: str, telemetry: Telemetry) -> None:
        self.ws_url = ws_url
        self.telemetry = telemetry
        self._ws = None

        # Prefer the synchronous websockets API (simple drop-in for sync agent code).
        try:
            from websockets.sync.client import connect  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "websockets sync client not available. "
                "Install/upgrade websockets: pip install -U websockets"
            ) from e

        self._ws = connect(self.ws_url)

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected.")

        req_id = str(uuid.uuid4())
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        self.telemetry.trace("out", msg)

        self._ws.send(json.dumps(msg))

        raw = self._ws.recv()
        if not raw:
            raise RuntimeError("WebSocket connection closed unexpectedly.")

        resp = json.loads(raw)
        self.telemetry.trace("in", resp)

        if resp.get("id") != req_id:
            raise RuntimeError(f"Correlation mismatch: sent id={req_id}, got id={resp.get('id')}")

        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error: {resp['error']}")

        return resp["result"]

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        finally:
            self._ws = None
