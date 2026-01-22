import asyncio
import json
import uuid
import websockets

ENDPOINT = "ws://127.0.0.1:8765/" # local MCP server address

async def run_loop(): # orchestrate handshake, resource read and tool call
    async with websockets.connect(ENDPOINT, subprotocols=["mcp"]) as ws: # open MCP session
        init_id = str(uuid.uuid4()) # unique initialise request_id
        """HANDSHAKE - This is the hello what can you do step
        the client introduces itself, clientName/clientVersion...
        the server responds with its capabilities, what it supports and what tools/resources exist"""
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {"clientName": "incident-cli", "clientVersion": "0.1.0"},
        })) # capabilities handshake payload
        caps = await ws.recv() # capture server capabilities

        res_id = str(uuid.uuid4()) # resource fetch request id
        """This is the OBSERVE-SUPPORT part"""
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": res_id,
            "method": "getResource",
            "params": {"uri": "memory://alerts/latest", "cursor": None},
        })) # receive snapshot payload
        alert_snapshot = await ws.recv()

        tool_id = str(uuid.uuid4())
        """This is the ACT step
        1) tool is named 'run_diagnostic'
        2) arguments are structured {'command':'...', 'host':'...'
        3) in a real MCP flow those arguments are validated against a schema discovered earlier
        => the server responds with a diagnostic oncluding * status(ok/error), results payload,
        metrics(latency/cost/etc)so your orchestrator can enforce budgets"""
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": tool_id,
            "method": "callTool",
            "params": {
                "name": "run_diagnostic",
                "arguments": {"command": "kubectl get pods", "host": "staging-api"},
            },
        }))
        diagnostic = await ws.recv()

        print("CAPS:", caps)
        print("ALERT:", alert_snapshot)
        print("DIAG:", diagnostic)

if __name__ == "__main__":
    asyncio.run(run_loop())
