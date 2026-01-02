import json
from pathlib import Path
from tempfile import TemporaryDirectory

from incident_mcp.server import MCPStdioServer


def _rpc(id_, method, params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def test_tool_calls_append_memory_entries():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "config").mkdir(parents=True)
        (root / "data" / "alerts").mkdir(parents=True)
        (root / "data" / "telemetry").mkdir(parents=True)
        (root / "data" / "runbooks").mkdir(parents=True)
        (root / "data" / "memory").mkdir(parents=True)

        # Minimal config/data
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
        (root / "data" / "alerts" / "latest.json").write_text(json.dumps({"alert_id": "A-1"}), encoding="utf-8")
        (root / "data" / "telemetry" / "recent.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (root / "data" / "runbooks" / "rb.md").write_text("# Runbook\n", encoding="utf-8")

        memory_path = root / "data" / "memory" / "memory.jsonl"

        server = MCPStdioServer(
            root_dir=root,
            tools_path=root / "config" / "tools.json",
            resources_path=root / "config" / "resources.json",
            alerts_latest_path=root / "data" / "alerts" / "latest.json",
            telemetry_recent_path=root / "data" / "telemetry" / "recent.json",
            runbook_dir=root / "data" / "runbooks",
            memory_path=memory_path,
            stderr_banner=False,
        )

        # Call the tool twice
        resp1 = server.handle(_rpc("1", "callTool", {"name": "run_diagnostic", "arguments": {"command": "kubectl get pods", "host": "staging"}}))
        assert "result" in resp1 and resp1["result"]["status"] in ("success", "error")

        resp2 = server.handle(_rpc("2", "callTool", {"name": "run_diagnostic", "arguments": {"command": "kubectl get pods", "host": "staging"}}))
        assert "result" in resp2

        # Memory must contain 2 entries
        lines = memory_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])

        assert e1["alert_id"] == "A-1"
        assert e1["tool_name"] == "run_diagnostic"
        assert "ts_utc" in e1
        assert e2["alert_id"] == "A-1"
