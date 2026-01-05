from __future__ import annotations

import argparse
import os
from pathlib import Path

from .client import IncidentCommandAgent


def main() -> None:
    parser = argparse.ArgumentParser()

    default_root = Path(__file__).resolve().parents[2]  # Capstone/

    parser.add_argument(
        "--server-cmd",
        default="python -m incident_mcp server",
        help="Command to start server (stdio transport only).",
    )
    parser.add_argument(
        "--root",
        default=str(default_root),
        help="Project root containing config/ and data/ (sets INCIDENT_MCP_ROOT).",
    )

    # NEW: transport selection
    parser.add_argument(
        "--transport",
        choices=["stdio", "ws"],
        default=os.getenv("MCP_TRANSPORT", "stdio").strip().lower(),
        help="Transport for MCP JSON-RPC: stdio (default) or ws.",
    )
    parser.add_argument(
        "--ws-url",
        default=os.getenv("MCP_WS_URL", "ws://127.0.0.1:8000/ws"),
        help="WebSocket URL when --transport ws (default: ws://127.0.0.1:8000/ws).",
    )

    # NEW: planner backend selection
    parser.add_argument(
        "--planner-backend",
        choices=["rules", "anthropic", "openai"],
        default=os.getenv("PLANNER_BACKEND", "anthropic").strip().lower(),
        help="Planner backend: rules|anthropic|openai (default from env PLANNER_BACKEND).",
    )

    args = parser.parse_args()

    # Ensure server & resources resolve correctly (both stdio + ws server can read this)
    os.environ["INCIDENT_MCP_ROOT"] = str(Path(args.root).expanduser().resolve())

    agent = IncidentCommandAgent(
        server_cmd=args.server_cmd.split(),
        planner_backend=args.planner_backend,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        max_repeat_actions=int(os.getenv("MAX_REPEAT_ACTIONS", "2")),
        transport=args.transport,
        ws_url=args.ws_url if args.transport == "ws" else None,
    )
    agent.run()

