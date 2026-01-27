from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from jsonschema import Draft7Validator

from .telemetry import Telemetry

load_dotenv()

class MCPStdioServer:
    """
    MCP-like server over stdio (newline-delimited JSON-RPC 2.0).

    - Resources: alerts, telemetry, runbooks catalog, memory (persistent JSONL)
    - Tools: retrieve_runbook, run_diagnostic (simulated allowlist), summarize_incident
    - Validation: jsonschema + defaults
    - Side effect: every tool call appends a concise delta entry to memory store
    """

    def __init__(
            self,
            *,
            root_dir: Optional[Path] = None,
            tools_path: Optional[Path] = None,
            resources_path: Optional[Path] = None,
            alerts_latest_path: Optional[Path] = None,
            telemetry_recent_path: Optional[Path] = None,
            runbook_dir: Optional[Path] = None,
            memory_path: Optional[Path] = None,
            stderr_banner: bool = True,
    ) -> None:
        import os

        # ------------------------------------------------------------------
        # Resolve project root (priority: explicit arg > env var > fallback)
        # This file lives at ".../Capstone/src/incident_mcp/server.py"
        # default_root => ".../Capstone"
        # ------------------------------------------------------------------
        default_root = Path(__file__).resolve().parents[2]
        env_root = os.getenv("INCIDENT_MCP_ROOT")

        if root_dir is not None:
            self.root = Path(root_dir).expanduser().resolve()
        elif env_root:
            self.root = Path(env_root).expanduser().resolve()
        else:
            self.root = default_root.resolve()

        # If someone accidentally passed ".../Capstone/src" as root, normalize it.
        # (Common when running from inside src/)
        if self.root.name == "src" and (self.root.parent / "config").exists():
            self.root = self.root.parent.resolve()

        # ------------------------------------------------------------------
        # Standard directory layout under project root
        # ------------------------------------------------------------------
        self.config_dir = (self.root / "config").resolve()
        self.data_dir = (self.root / "data").resolve()

        # Ensure expected dirs exist (harmless if already present)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Helper: resolve optional paths.
        # If caller passes a relative path, treat it as relative to project root.
        # ------------------------------------------------------------------
        def _resolve_under_root(p: Optional[Path], default_rel: Path) -> Path:
            if p is None:
                return (self.root / default_rel).resolve()
            p = Path(p).expanduser()
            return p.resolve() if p.is_absolute() else (self.root / p).resolve()

        # ------------------------------------------------------------------
        # Config files (schemas, tool definitions, resource catalog)
        # ------------------------------------------------------------------
        self.tools_path = _resolve_under_root(tools_path, Path("config/tools.json"))
        self.resources_path = _resolve_under_root(resources_path, Path("config/resources.json"))

        # Fail early with a helpful error message if config is missing
        if not self.tools_path.exists():
            raise FileNotFoundError(
                f"Missing tools.json. Expected at: {self.tools_path}\n"
                f"Resolved project root: {self.root}\n"
                f"Hint: ensure Capstone/config/tools.json exists, or pass tools_path=..., "
                f"or set INCIDENT_MCP_ROOT to the Capstone directory."
            )
        if not self.resources_path.exists():
            raise FileNotFoundError(
                f"Missing resources.json. Expected at: {self.resources_path}\n"
                f"Resolved project root: {self.root}\n"
                f"Hint: ensure Capstone/config/resources.json exists, or pass resources_path=..., "
                f"or set INCIDENT_MCP_ROOT to the Capstone directory."
            )

        # ------------------------------------------------------------------
        # Resource backing files
        # ------------------------------------------------------------------
        self.alerts_latest_path = _resolve_under_root(
            alerts_latest_path, Path("data/alerts/latest.json")
        )
        self.telemetry_recent_path = _resolve_under_root(
            telemetry_recent_path, Path("data/telemetry/recent.json")
        )

        self.runbook_dir = _resolve_under_root(runbook_dir, Path("data/runbooks"))
        self.runbook_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Memory store (persistent JSONL)
        # ------------------------------------------------------------------
        mem_dir = (self.data_dir / "memory").resolve()
        mem_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = _resolve_under_root(memory_path, Path("data/memory/memory.jsonl"))

        # ------------------------------------------------------------------
        # Load static server configuration
        # ------------------------------------------------------------------
        self.tools: List[Dict[str, Any]] = self._load_json(self.tools_path)
        self.resources: List[Dict[str, Any]] = self._load_json(self.resources_path)

        # ------------------------------------------------------------------
        # Tool dispatch table
        # ------------------------------------------------------------------
        self.tool_handlers = {
            "retrieve_runbook": self._tool_retrieve_runbook,
            "run_diagnostic": self._tool_run_diagnostic,
            "summarize_incident": self._tool_summarize_incident,
        }

        # ------------------------------------------------------------------
        # Telemetry (server-side)
        # ------------------------------------------------------------------
        self.telemetry = Telemetry(run_id="server")
        self.stderr_banner = stderr_banner

    # ----------------------------
    # IO helpers
    # ----------------------------
    @staticmethod
    def _load_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _jsonrpc_result(req_id: str, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _jsonrpc_error(req_id: str, code: int, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    @staticmethod
    def _read_json_lines(stdin=None) -> Any:
        """Server is running as a separate process, reading stdin line
        by line."""
        stream = stdin or sys.stdin
        for line in stream:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

    @staticmethod
    def _write_json(obj: Dict[str, Any], stdout=None) -> None:
        stream = stdout or sys.stdout
        stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stream.flush()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # ----------------------------
    # Tool schema lookup + validation
    # ----------------------------
    def _find_tool(self, name: str) -> Optional[Dict[str, Any]]:
        for t in self.tools:
            if t.get("name") == name:
                return t
        return None

    @staticmethod
    def _apply_defaults(schema: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        props = schema.get("properties", {})
        out = dict(args)
        for k, spec in props.items():
            if k not in out and "default" in spec:
                out[k] = spec["default"]
        return out

    @staticmethod
    def _validate_against_schema(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[bool, str]:
        v = Draft7Validator(schema)
        errors = sorted(v.iter_errors(args), key=lambda e: e.path)
        if not errors:
            return True, "ok"
        e0 = errors[0]
        loc = ".".join([str(p) for p in e0.path]) if e0.path else "(root)"
        return False, f"{loc}: {e0.message}"

    # ----------------------------
    # Memory store
    # ----------------------------
    def _append_memory(self, entry: Dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault("ts_utc", self.telemetry.utc_iso())
        Telemetry.write_jsonl(self.memory_path, entry)

    def _load_memory(self, limit: int = 200, alert_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.memory_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if alert_id and obj.get("alert_id") != alert_id:
                    continue
                rows.append(obj)
        return rows[-limit:]

    # ----------------------------
    # Resources
    # ----------------------------
    def _get_resource(self, uri: str, cursor: Optional[str]) -> Dict[str, Any]:
        parsed = urlparse(uri)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        q = parse_qs(parsed.query)

        if base == "memory://alerts/latest":
            alert = self._load_json(self.alerts_latest_path)
            return {"uri": uri, "cursor": None, "chunks": [{"id": "chunk-0", "content_type": "application/json", "data": alert}]}

        if base == "memory://telemetry/recent":
            tel = self._load_json(self.telemetry_recent_path)
            return {"uri": uri, "cursor": None, "chunks": [{"id": "chunk-0", "content_type": "application/json", "data": tel}]}

        if base == "memory://runbooks/catalog":
            files = sorted([p.name for p in self.runbook_dir.glob("*.md")])
            return {"uri": uri, "cursor": None, "chunks": [{"id": "chunk-0", "content_type": "application/json", "data": {"runbooks": files}}]}

        if base == "memory://memory/recent":
            limit = int(q.get("limit", ["20"])[0])
            limit = max(1, min(200, limit))
            aid = q.get("alert_id", [None])[0]
            entries = self._load_memory(limit=limit, alert_id=aid)
            return {"uri": uri, "cursor": None, "chunks": [{"id": "chunk-0", "content_type": "application/json", "data": {"entries": entries}}]}

        if base == "memory://memory/all":
            limit = int(q.get("limit", ["200"])[0])
            limit = max(1, min(500, limit))
            entries = self._load_memory(limit=limit, alert_id=None)
            return {"uri": uri, "cursor": None, "chunks": [{"id": "chunk-0", "content_type": "application/json", "data": {"entries": entries}}]}

        raise KeyError(f"Unknown resource URI: {uri}")

    # ----------------------------
    # Tool handlers
    # ----------------------------
    def _tool_retrieve_runbook(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args["query"].lower()
        top_k = int(args.get("top_k", 5))
        max_chars = int(args.get("max_chars_per_hit", 800))
        tokens = [t for t in query.replace("-", " ").split() if t]

        hits: List[Dict[str, Any]] = []
        for path in self.runbook_dir.glob("*.md"):
            txt = path.read_text(encoding="utf-8", errors="ignore")
            low = txt.lower()
            score = float(sum(low.count(t) for t in tokens))
            if score <= 0:
                continue
            hits.append(
                {"doc_id": path.name, "chunk_id": 0, "score": score, "text": txt[:max_chars], "citation": f"{path.name}#0"}
            )

        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = hits[:top_k]
        return {"query": args["query"], "hits": hits}

    def _tool_run_diagnostic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        This tool returns a simulated diagnostic result rather than executing
        a real shell command. The stdout is embedded directly in the code so it
        can be consumed immediately by the agent, logged into traces, and persisted
        into memory in a deterministic and reproducible way.

        In a production system, this method would execute a real command (e.g. via
        subprocess or a remote API), apply timeouts and output limits, and optionally
        persist full logs to artifacts while still returning a structured stdout/stderr
        summary to the agent.
        """
        command = args["command"]
        host = args["host"]

        allowed_prefixes = ("kubectl get pods", "kubectl describe pod", "kubectl logs")
        if not any(command.startswith(p) for p in allowed_prefixes):
            return {
                "host": host,
                "command": command,
                "exit_code": 2,
                "stdout": "",
                "stderr": f"Command not allowed. Allowed prefixes: {allowed_prefixes}",
            }

        stdout = (
            "NAME                          READY   STATUS             RESTARTS   AGE\n"
            "staging-api-7c9d9f9f8f-abc    0/1     CrashLoopBackOff   7          12m\n"
            "staging-api-7c9d9f9f8f-def    0/1     CrashLoopBackOff   6          12m\n"
            "staging-api-7c9d9f9f8f-ghi    0/1     CrashLoopBackOff   8          12m\n"
        )
        return {"host": host, "command": command, "exit_code": 0, "stdout": stdout, "stderr": ""}

    def _tool_summarize_incident(self, args: Dict[str, Any]) -> Dict[str, Any]:
        alert_id = args["alert_id"]
        evidence = args["evidence"]

        md: List[str] = []
        md.append("# Incident Handoff")
        md.append("")
        md.append(f"**Alert ID:** {alert_id}")
        md.append("")
        md.append("## Evidence (high signal)")
        for e in evidence[:12]:
            md.append(f"- {e.strip()}")
        md.append("")
        md.append("## Recommendation")
        md.append("- Inspect logs of one failing pod (`kubectl logs <pod> --previous`).")
        md.append("- Check recent deployment diff (image/env/secrets).")
        md.append("- If cause identified, consider rollback/restart per runbook.")
        md.append("")
        return {"handoff_markdown": "\n".join(md)}

    @staticmethod
    def _summarize_tool_delta(tool_name: str, status: str, payload: Dict[str, Any]) -> str:
        if tool_name == "retrieve_runbook":
            hits = payload.get("hits", [])
            if not hits:
                s = "Searched runbooks but found no matching passages."
            else:
                best = hits[0]
                s = f"Retrieved {len(hits)} runbook passages; best={best.get('citation')}."
        elif tool_name == "run_diagnostic":
            if status != "success":
                s = f"Diagnostic blocked/failed: {payload.get('stderr','unknown error')}."
            else:
                stdout = payload.get("stdout", "")
                s = "Ran diagnostic successfully."
                if "CrashLoopBackOff" in stdout:
                    s += " Pods appear in CrashLoopBackOff."
        elif tool_name == "summarize_incident":
            s = "Drafted incident handoff note."
        else:
            s = f"Tool {tool_name} completed with status={status}."

        s = s.strip()
        if len(s) > 600:
            s = s[:597] + "..."
        return s

    # ----------------------------
    # Public JSON-RPC handler
    # ----------------------------
    def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Server side the server loop reads the JSON from stdin,
        it checks the method name. The method assumes it has already read JSON fron
        stdin and parse it into `msg`. Handles job id to interpret the result and return
        a JSOM-RPC response dict"""

        """First extract the basic fields which correspond to the client envelope fields"""
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})
        """validate the JSON-RPC structure, this is protocol hygiene -32600 is a standard
        JSON-RPC error code"""
        if msg.get("jsonrpc") != "2.0" or not req_id or not method:
            return self._jsonrpc_error(req_id or "unknown", -32600, "Invalid Request")
        """the first JSON message is the initialise method, this is the server saying "heres what i
        am and heres what i can do. The tools resources comes from the JSON config files in the
        Capstone/config folder"""
        if method == "initialize":
            result = {
                "serverName": "incident-mcp-stdio",
                "serverVersion": "0.6.0",
                "capabilities": {"tools": self.tools, "resources": self.resources},
            }
            """the server uses the _jsonrpc_result handler to return the 
            initialise message payload withe the same id as the inialise
            message so it can correlate the response to the initial
            request"""
            return self._jsonrpc_result(req_id, result)
        """The server then supports getResource a read only of structured
        data from a URI"""
        if method == "getResource":
            try:
                uri = params["uri"]
                cursor = params.get("cursor")
                result = self._get_resource(uri, cursor)
                return self._jsonrpc_result(req_id, result)
            except Exception as e:
                return self._jsonrpc_error(req_id, -32000, "getResource failed", {"detail": str(e)})
        """callTool is the action method..."""
        if method == "callTool":
            try:
                """ it extracts the tool name and arguments"""
                name = params["name"]
                raw_args = params.get("arguments", {})
                """finds the tool schema"""
                tool = self._find_tool(name)
                if tool is None:
                    return self._jsonrpc_error(req_id, -32601, f"Unknown tool: {name}")
                schema = tool.get("inputSchema", {})
                """Applies defaults and then validates...a requirement of the Capstone validate
                before executing"""
                args = self._apply_defaults(schema, raw_args)
                ok, msg2 = self._validate_against_schema(schema, args)
                if not ok:
                    return self._jsonrpc_error(req_id, -32602, "Invalid params", {"detail": msg2})
                """We then execute the tool handler function"""
                handler = self.tool_handlers.get(name)
                if handler is None:
                    return self._jsonrpc_error(req_id, -32601, f"No handler implemented for tool: {name}")
                """execute the tool and measure time"""
                t0 = self._now_ms()
                payload = handler(args)
                t1 = self._now_ms()
                """decide the status and build a result object"""
                status = "success"
                if name == "run_diagnostic" and int(payload.get("exit_code", 0)) != 0:
                    status = "error"
                result = {
                    "status": status,
                    "payload": payload,
                    "metrics": {"latency_ms": t1 - t0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0},
                }

                # Side effect: append memory delta
                """This is A BIG DEAL, the server is the only place that appends memory. This
                is the 'Learn' but implemented server side as a persistant log of actions"""
                try:
                    alert_id = args.get("alert_id")
                    if not alert_id:
                        alert = self._load_json(self.alerts_latest_path)
                        alert_id = alert.get("alert_id", "unknown")
                except Exception:
                    alert_id = "unknown"

                entry: Dict[str, Any] = {
                    "alert_id": alert_id,
                    "kind": "tool_delta",
                    "tool_name": name,
                    "status": status,
                    "summary": self._summarize_tool_delta(name, status, payload),
                    "metrics": result["metrics"],
                }
                if name == "retrieve_runbook":
                    hits = payload.get("hits", [])
                    entry["refs"] = [h.get("citation") for h in hits[:5] if h.get("citation")]
                self._append_memory(entry)

                return self._jsonrpc_result(req_id, result)

            except Exception as e:
                return self._jsonrpc_error(req_id, -32001, "callTool failed", {"detail": str(e)})

        return self._jsonrpc_error(req_id, -32601, f"Method not found: {method}")

    """serve_forever is the missing "outer loop" it is the part of the server that continuously reads
    from stdin, calls the handle() method to compute a response and then writes that response to stdout."""
    def serve_forever(self, stdin=None, stdout=None) -> None:
        """A method is defined that runs "forever" (until input ends), stdin and stdout are optional
        so you can either - use real process stdio ro inject fake streams for testing"""
        if self.stderr_banner:
            print("incident-mcp-stdio server running (newline-delimited JSON-RPC over stdio)", file=sys.stderr)
            print(f"root: {self.root}", file=sys.stderr)
            print(f"memory store: {self.memory_path}", file=sys.stderr)
        """This is the servers ears, it continuously reads lines from stdin, parses each line as JSON
        and then yield it as a dict called msg"""
        for msg in self._read_json_lines(stdin=stdin):
            """this is the servers brain, handle(msg) looks at msg['method'] ie initialise, getResource,
            callTool and returns a dict that is a JSON-RPC resonse"""
            resp = self.handle(msg)
            """This is the servers mouth, it seroialises resp to JSON and writes a singke newline JSON line 
            to stdout."""
            self._write_json(resp, stdout=stdout)
