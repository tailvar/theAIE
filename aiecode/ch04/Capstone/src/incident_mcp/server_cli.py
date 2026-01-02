from __future__ import annotations

from .server import MCPStdioServer


def main() -> None:
    server = MCPStdioServer()
    server.serve_forever()
