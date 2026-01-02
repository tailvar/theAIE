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
        help="Command to start server (stdio).",
    )
    parser.add_argument(
        "--root",
        default=str(default_root),
        help="Project root containing config/ and data/ (passed via env INCIDENT_MCP_ROOT).",
    )

    # NEW: planner choice via CLI (overrides env)
    parser.add_argument(
        "--planner-backend",
        choices=["anthropic", "openai", "rules"],
        default=os.getenv("PLANNER_BACKEND", "anthropic"),
        help="Planner backend to use: anthropic, openai, or rules.",
    )

    # OPTIONAL: allow model overrides via CLI too
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name (if planner-backend=openai).",
    )
    parser.add_argument(
        "--anthropic-model",
        default=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        help="Anthropic model name (if planner-backend=anthropic).",
    )

    args = parser.parse_args()

    os.environ["INCIDENT_MCP_ROOT"] = args.root

    agent = IncidentCommandAgent(
        server_cmd=args.server_cmd.split(),
        planner_backend=args.planner_backend,
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model,
        max_repeat_actions=int(os.getenv("MAX_REPEAT_ACTIONS", "2")),
    )
    agent.run()

