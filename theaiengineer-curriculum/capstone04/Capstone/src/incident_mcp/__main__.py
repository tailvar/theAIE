# src/incident_mcp/__main__.py
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m incident_mcp [server|agent|ws_server] ...")
        raise SystemExit(2)

    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if mode == "server":
        from .server_cli import main as server_main
        server_main()

    elif mode == "ws_server":
        from .server_ws_cli import main as ws_server_main
        ws_server_main()

    elif mode == "agent":
        from .agent_cli import main as agent_main
        agent_main()

    else:
        print(f"Unknown mode: {mode}. Use server|agent|ws_server.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()


