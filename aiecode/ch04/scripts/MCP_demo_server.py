import asyncio
import json
import time
import websockets

HOST = "127.0.0.1"
PORT = 8765

RUNBOOK_SNIPPETS = [
    "Restart service if CPU > 90% for 5 minutes.",
    "If pods crashloop, capture logs before redeploying"
]

ALERT_PAYLOAD = {
    "id":"ALRT-2025-07",
    "service":"staging-api",
    "symptom":"CPU spike on node 3",
    "severity":"high"
}

CAPABILITIES = {
    "tools": [{
        "name":"run_diagnostic",
        "description":"Return canned diagnostic for a host",
        "schema":{
            "type":"object",
            "properties":{
                "command":{"type":"string"},
                "host":{"type":"string"},
            },
            "required":["command","host"],
        },
    }],
    "resources": [{
        "uri":"memory://alerts/latest",
        "description":"Latest alert snapshot",
    }],
}

async def handle_session(ws):
    async for raw in ws:
        req = json.loads(raw)
        method = req.get("method")
        req_id = req.get("id")

        if method == "initialize":
            result = {"capabilities": CAPABILITIES}

        elif method == "getResource":
            result = {
                "uri": "memory://alerts/latest",
                "data": {"alert": ALERT_PAYLOAD, "recommendations": RUNBOOK_SNIPPETS},
            }

        elif method == "callTool":
            args = req.get("params", {}).get("arguments", {})
            latency_ms = int((time.perf_counter() % 0.02) * 1000)
            result = {
                "status": "ok",
                "data": {
                    "command": args.get("command"),
                    "host": args.get("host"),
                    "stdout": "All pods healthy",
                },
                "metrics": {"latency_ms": latency_ms},
            }

        else:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }))
            continue

        await ws.send(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}))

async def main():
    async with websockets.serve(handle_session, HOST, PORT, subprotocols=["mcp"]):
        print(f"Server ready on ws://{HOST}:{PORT}/  (subprotocol=mcp)")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())