from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .telemetry import Telemetry

load_dotenv()


@dataclass
class Budgets:
    max_tool_calls: int
    max_run_ms: int
    max_steps: int
    max_tokens: int
    max_cost_usd: float

    tool_calls_used: int = 0
    elapsed_ms: int = 0
    steps_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0

    def breach_reason(self) -> Optional[str]:
        if self.steps_used > self.max_steps:
            return f"steps {self.steps_used} > {self.max_steps}"
        if self.tool_calls_used > self.max_tool_calls:
            return f"tool_calls {self.tool_calls_used} > {self.max_tool_calls}"
        if self.elapsed_ms > self.max_run_ms:
            return f"elapsed_ms {self.elapsed_ms} > {self.max_run_ms}"
        if self.tokens_used > self.max_tokens:
            return f"tokens_used {self.tokens_used} > {self.max_tokens}"
        if self.cost_usd > self.max_cost_usd:
            return f"cost_usd {self.cost_usd:.4f} > {self.max_cost_usd:.4f}"
        return None

    def as_log_dict(self) -> Dict[str, Any]:
        return {
            "steps_used": self.steps_used,
            "tool_calls_used": self.tool_calls_used,
            "elapsed_ms": self.elapsed_ms,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
        }


class StdioJSONRPC:
    """JSON-RPC over a subprocess stdio, with transcript logging.
    This is the place where 'agent calls server begins self.proc starts
    the server proram as a separate process"""

    def __init__(self, cmd: List[str], telemetry: Telemetry) -> None:
        self.telemetry = telemetry
        """Given thh agent a writable handle to the servers
        stdin and a readable handle to its stdout. Capture its 
        stderr (so errors dont mess up the output"""
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        """agent can write JSON lines into proc.stdin
        as well as read JSON lines from proc.stdout"""

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """we request a nuique id (label) so we can match the response
        to the request. This step is essential because JSON-RPC is designed for
        general messaging where many requests could be in flight at the same
        time"""
        req_id = str(uuid.uuid4())
        """Build the JSON-RPC request object. This is a strictly formatted message,
        that the server will understand"""
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        """telemetry.trace logs the exact request into the trace file, this is part of why 
        it can be replayed later"""
        self.telemetry.trace("out", msg)
        """then ensure stdin exists...if the server process wasnt started correctly,
        stdin might be missing. This assert fails the process early instead of failing 
        later in a confusing way"""
        assert self.proc.stdin is not None
        """the serialise the message turning the Python dict into a JSON string, 
        add a new line "/n" as protocol is one JSON object per line"""
        self.proc.stdin.write(json.dumps(msg) + "\n")
        """Flush so it gets delivered to the server immediately"""
        self.proc.stdin.flush()
        """in the same way we test stdin we also ensure that stdout exists"""
        assert self.proc.stdout is not None
        """Wait until the sever replies with exactly one line of JSON, if we dont receive 
        this the server crashes and closes stdout. This is the simplest request/response 
        model, * send one line * receive one line"""
        line = self.proc.stdout.readline()
        """if we dont have one line, we know the server died, if stdout closes it usually
        means the server has crashed or exited. Then try and read stderr and include it
        in the exception.
        IMPORTANT NOTE:
            - protocol messages travel on stdout
            - hunam debug output and crash messages go on stderr"""
        if not line:
            # Server likely crashed; show stderr to explain why.
            err = ""
            if self.proc.stderr is not None:
                try:
                    # read whatever is available without blocking too much
                    err = self.proc.stderr.read()
                except Exception:
                    err = ""
            raise RuntimeError(
                f"Server stdout closed unexpectedly.\n--- server stderr ---\n{err}\n---------------------")
        """decode the JSON response into a dict and log it as an inoming trace
        event - important for replay"""
        resp = json.loads(line)
        self.telemetry.trace("in", resp)
        """CORRELATION CHECK
            - this enforces the server must respond with the same id, in real systems 
            especially WebSockets where requests can overlap, it is an essential step"""
        if resp.get("id") != req_id:
            raise RuntimeError(f"Correlation mismatch: sent id={req_id}, got id={resp.get('id')}")
        """Verify the servers response is valid and return the result
        if the server says its failed, then the agent treats that as an exception,
        otherwise the agents get the servers "results" payload..."""
        if "error" in resp:
            raise RuntimeError(f"JSON-RPC error: {resp['error']}")
        return resp["result"]

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()


class IncidentCommandAgent:
    def __init__(
            self,
            *,
            server_cmd: List[str],
            artifacts_dir: Path = Path("artifacts"),
            planner_backend: str = "anthropic",  # anthropic|openai|rules
            openai_model: str = "gpt-4o-mini",
            anthropic_model: str = "claude-3-5-sonnet-20241022",
            max_repeat_actions: int = 2,
            planner_prompt_path: Optional[Path] = None,
    ) -> None:
        self.server_cmd = server_cmd

        # Project root (Capstone/)
        self.project_root = Path(__file__).resolve().parents[2]

        # Artifacts root
        if artifacts_dir.is_absolute():
            self.artifacts_dir = artifacts_dir
        else:
            self.artifacts_dir = self.project_root / artifacts_dir

        self.traces_dir = self.artifacts_dir / "traces"
        self.handoffs_dir = self.artifacts_dir / "handoffs"
        self.runs_jsonl = self.artifacts_dir / "runs.jsonl"

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)

        # Prompts dir
        self.prompts_dir = self.project_root / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Prompt path (override via arg > env > default)
        if planner_prompt_path is None:
            env_prompt = os.getenv("PLANNER_PROMPT_PATH")
            if env_prompt:
                planner_prompt_path = Path(env_prompt)
            else:
                planner_prompt_path = self.prompts_dir / "planner_system.txt"

        self.planner_prompt_path = (
            planner_prompt_path
            if planner_prompt_path.is_absolute()
            else (self.project_root / planner_prompt_path)
        )

        # Telemetry
        self.run_id = f"run-{uuid.uuid4().hex[:10]}"
        self.telemetry = Telemetry(
            run_id=self.run_id,
            trace_path=self.traces_dir / f"{self.run_id}.jsonl",
            runs_path=self.runs_jsonl,
        )

        # Planner configuration
        self.planner_backend = planner_backend.strip().lower()
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.max_repeat_actions = max_repeat_actions

        # --- RPC transport to server -----
        self.rpc = StdioJSONRPC(self.server_cmd, self.telemetry)

        # ---- State -----
        self.step = 0
        self.recent_actions: List[Tuple[str, str]] = []
        self.evidence: List[str] = []

        self.tools: List[Dict[str, Any]] = []
        self.tool_allow: set[str] = set()
        self.alert_id: str = "unknown"


    @staticmethod
    def extract_alert_id(alert_resource: Dict[str, Any]) -> str:
        try:
            return str(alert_resource["chunks"][0]["data"].get("alert_id", "unknown"))
        except Exception:
            return "unknown"

    def log(self, budgets: Budgets, event: str, **fields: Any) -> None:
        self.telemetry.event(self.step, budgets.as_log_dict(), event, **fields)

    def initialize(self, budgets: Budgets) -> None:
        self.step += 1
        budgets.steps_used += 1
        self.log(budgets, "initialize_request")

        """this is the important line as its the first real message
        between the agent and the server"""
        caps = self.rpc.request("initialize", {"clientName": "incident-agent", "clientVersion": "0.6.0"})
        self.tools = caps["capabilities"]["tools"]
        self.tool_allow = {t["name"] for t in self.tools}

        self.step += 1
        budgets.steps_used += 1
        self.log(budgets, "initialize_response", tools=len(self.tools), resources=len(caps["capabilities"]["resources"]))

    def observe(self, budgets: Budgets) -> Dict[str, Any]:
        self.step += 1
        budgets.steps_used += 1

        alert_res = self.rpc.request("getResource", {"uri": "memory://alerts/latest", "cursor": None})
        telemetry_res = self.rpc.request("getResource", {"uri": "memory://telemetry/recent", "cursor": None})

        self.alert_id = self.extract_alert_id(alert_res)

        mem_res = self.rpc.request(
            "getResource",
            {"uri": f"memory://memory/recent?alert_id={self.alert_id}&limit=20", "cursor": None},
        )
        recent_memory_entries = mem_res["chunks"][0]["data"].get("entries", [])

        obs = {"alert": alert_res, "telemetry": telemetry_res, "recent_memory": recent_memory_entries}
        self.log(budgets, "observe", alert_id=self.alert_id)
        return obs

    def plan(self, budgets: Budgets, observation: Dict[str, Any]) -> Dict[str, Any]:
        self.step += 1
        budgets.steps_used += 1

        tool_brief = [
            {"name": t["name"], "description": t.get("description", ""), "inputSchema": t.get("inputSchema", {})}
            for t in self.tools
        ]

        system_text = self.planner_prompt_path.read_text(encoding="utf-8")

        payload = {
            "alert_id": self.alert_id,
            "observation": observation,
            "evidence_so_far": self.evidence[-8:],
            "tools": tool_brief,
            "budgets_remaining": {
                "tool_calls_left": max(0, budgets.max_tool_calls - budgets.tool_calls_used),
                "ms_left": max(0, budgets.max_run_ms - budgets.elapsed_ms),
                "steps_left": max(0, budgets.max_steps - budgets.steps_used),
                "tokens_left": max(0, budgets.max_tokens - budgets.tokens_used),
                "usd_left": max(0.0, budgets.max_cost_usd - budgets.cost_usd),
            },
        }

        # --- Deterministic rules planner (no LLM) ---
        if self.planner_backend == "rules":
            action = self.rule_plan(budgets, observation)
            self.log(budgets, "decision", action=action, planner="rules")
            return action

        # --- LLM planners produce JSON text we parse ---
        text: str

        if self.planner_backend == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            if resp.usage:
                budgets.tokens_used += int(resp.usage.total_tokens or 0)

        else:
            from anthropic import Anthropic

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=self.anthropic_model,
                max_tokens=300,
                temperature=0.2,
                system=system_text,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content).strip()
            usage = getattr(resp, "usage", None)
            if usage:
                budgets.tokens_used += int(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))

        # Parse model output (must be a JSON object)
        try:
            action = json.loads(text)
        except json.JSONDecodeError:
            # Log raw text for debugging and halt cleanly
            self.log(budgets, "halt", reason="Planner returned non-JSON", raw=text[:500])
            return {"action": "finish", "reason": "Planner returned invalid JSON"}

        self.log(budgets, "decision", action=action, planner=self.planner_backend)
        return action

    def rule_plan(self, budgets: Budgets, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic planner used when planner_backend == "rules".
        Returns the same action schema as the LLM planner:
          {"action":"callTool","name":...,"arguments":{...}}
          {"action":"finish","reason":...}
        """
        alert = observation.get("alert", {}) or {}
        telemetry = observation.get("telemetry", {}) or {}

        summary = (alert.get("summary") or "").lower()
        service = alert.get("service") or telemetry.get("service") or "staging-api"
        alert_id = alert.get("id") or observation.get("alert_id") or self.alert_id

        # 1) If we haven't retrieved a runbook yet, do that first.
        if not any(e.startswith("[retrieve_runbook]") for e in self.evidence):
            query = "CrashLoopBackOff kubernetes pods" if "crashloop" in summary else (summary or "incident triage")
            return {
                "action": "callTool",
                "name": "retrieve_runbook",
                "arguments": {"query": query, "top_k": 3},
            }

        # 2) If we haven't run diagnostics yet, do that next.
        if not any(e.startswith("[run_diagnostic]") for e in self.evidence):
            command = "kubectl get pods"
            if "5xx" in summary or "error rate" in summary:
                command = "kubectl get pods && kubectl get deploy"
            return {
                "action": "callTool",
                "name": "run_diagnostic",
                "arguments": {"command": command, "host": service},
            }

        # 3) If budgets are too tight to be useful, finish.
        if (budgets.max_tool_calls - budgets.tool_calls_used) <= 1:
            return {"action": "finish", "reason": "Tool-call budget nearly exhausted"}

        # 4) Otherwise, summarize for a human handoff.
        return {
            "action": "callTool",
            "name": "summarize_incident",
            "arguments": {"alert_id": alert_id, "evidence": self.evidence[-10:]},
        }

    def act(self, budgets: Budgets, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute the planner action.

        Supports two planner formats:
          A) {"action":"callTool","name":"tool","arguments":{...}}
          B) {"action":"tool_name","arguments":{...}}  (shorthand)

        Returns the tool result dict, or None if finishing/halting.
        """

        # ----------------------------
        # Finish action
        # ----------------------------
        if action.get("action") == "finish":
            self.step += 1
            budgets.steps_used += 1
            self.log(
                budgets,
                "finish",
                reason=action.get("reason", "planner finished"),
            )
            return None

        # ----------------------------
        # Normalize planner output
        # ----------------------------
        tool_name: Optional[str] = None
        tool_args: Dict[str, Any] = {}

        if action.get("action") == "callTool":
            tool_name = action.get("name")
            tool_args = action.get("arguments", {})
        else:
            # Shorthand: action == tool name
            candidate = action.get("action")
            if isinstance(candidate, str) and candidate in self.tool_allow:
                tool_name = candidate
                tool_args = action.get("arguments", {})
            else:
                self.step += 1
                budgets.steps_used += 1
                self.log(
                    budgets,
                    "halt",
                    reason=f"Invalid planner action: {action}",
                )
                return None

        if not tool_name:
            self.step += 1
            budgets.steps_used += 1
            self.log(
                budgets,
                "halt",
                reason=f"No tool name resolved from action: {action}",
            )
            return None

        # ----------------------------
        # Budget enforcement
        # ----------------------------
        if budgets.tool_calls_used >= budgets.max_tool_calls:
            self.step += 1
            budgets.steps_used += 1
            self.log(
                budgets,
                "halt",
                reason="Tool call budget exhausted",
            )
            return None

        # ----------------------------
        # Call tool via MCP
        # ----------------------------
        self.step += 1
        budgets.steps_used += 1
        budgets.tool_calls_used += 1

        self.log(
            budgets,
            "tool_call",
            tool=tool_name,
            arguments=tool_args,
        )

        result = self.rpc.request(
            "callTool",
            {
                "name": tool_name,
                "arguments": tool_args,
            },
        )

        self.log(
            budgets,
            "tool_result",
            tool=tool_name,
            status=result.get("status"),
        )

        # ----------------------------
        # Evidence + memory handling
        # ----------------------------
        payload = result.get("payload")

        if payload is not None:
            # Store a compact textual delta for planner context
            try:
                # Summaries can be large; keep them tighter
                max_len = 600 if tool_name == "summarize_incident" else 1500
                delta = json.dumps(payload, ensure_ascii=False)[:max_len]
            except Exception:
                delta = str(payload)[:max_len]

            self.evidence.append(
                f"[{tool_name}] {delta}"
            )

        # ----------------------------
        # Write handoff artifact if summarizer
        # ----------------------------
        if tool_name == "summarize_incident":
            out_path = self.handoffs_dir / f"handoff_{self.run_id}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            handoff_md = ""
            if isinstance(payload, dict):
                # Preferred key (matches your server + evidence)
                handoff_md = payload.get("handoff_markdown") or payload.get("handoff") or payload.get("text") or ""
            else:
                handoff_md = str(payload)

            if handoff_md.strip():
                out_path.write_text(handoff_md, encoding="utf-8")
                self.log(budgets, "artifact_written", path=str(out_path))
            else:
                # Don't create empty artifacts; log for debugging
                self.log(
                    budgets,
                    "halt",
                    reason="summarize_incident returned empty handoff content",
                    payload_preview=str(payload)[:300] if payload is not None else None,
                )
                return None

        return result


    def learn(self, budgets: Budgets, act_out: Optional[Dict[str, Any]]) -> None:
        self.step += 1
        budgets.steps_used += 1
        self.log(budgets, "learn", note="Server persisted tool delta to memory store.", evidence_tail=self.evidence[-6:])

    def run(self) -> None:
        budgets = Budgets(
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "6")),
            max_run_ms=int(os.getenv("MAX_RUN_MS", "20000")),
            max_steps=int(os.getenv("MAX_STEPS", "25")),  # <-- was 10
            max_tokens=int(os.getenv("MAX_TOKENS", "8000")),
            max_cost_usd=float(os.getenv("MAX_COST_USD", "0.25")),
        )
        t0 = time.time()

        try:
            self.initialize(budgets)
            observation = self.observe(budgets)

            while True:
                budgets.elapsed_ms = int((time.time() - t0) * 1000)
                breach = budgets.breach_reason()
                if breach:
                    self.step += 1
                    self.log(budgets, "halt", reason=f"Budget exceeded: {breach}")
                    break

                action = self.plan(budgets, observation)
                act_out = self.act(budgets, action)
                if act_out is None:
                    break
                self.learn(budgets, act_out)

                observation = self.observe(budgets)

        finally:
            self.rpc.close()

