import json
from pathlib import Path
from tempfile import TemporaryDirectory

from incident_mcp.server import MCPStdioServer


def _rpc(id_, method, params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def test_schema_mismatch_returns_error():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "config").mkdir(parents=True)
        (root / "data" / "alerts").mkdir(parents=True)
        (root / "data" / "telemetry").mkdir(parents=True)
        (root / "data" / "runbooks").mkdir(parents=True)
        (root / "data" / "memory").mkdir(parents=True)

        (root / "config" / "tools.json").write_text(json.dumps([
            {
                "name": "run_diagnostic",
                "description": "Run safe diagnostic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "host": {"type": "string"},
                    },
                    "required": ["command", "host"]
                }
            }
        ]), encoding="utf-8")

        (root / "config" / "resources.json").write_text(json.dumps([]), encoding="utf-8")
        (root / "data" / "alerts" / "latest.json").write_text(json.dumps({"alert_id": "A-2"}), encoding="utf-8")
        (root / "data" / "telemetry" / "recent.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

        server = MCPStdioServer(
            root_dir=root,
            tools_path=root / "config" / "tools.json",
            resources_path=root / "config" / "resources.json",
            alerts_latest_path=root / "data" / "alerts" / "latest.json",
            telemetry_recent_path=root / "data" / "telemetry" / "recent.json",
            runbook_dir=root / "data" / "runbooks",
            memory_path=root / "data" / "memory" / "memory.jsonl",
            stderr_banner=False,
        )

        # Missing required "host"
        resp = server.handle(_rpc("1", "callTool", {"name": "run_diagnostic", "arguments": {"command": "kubectl get pods"}}))
        assert "error" in resp
        assert resp["error"]["code"] == -32602
