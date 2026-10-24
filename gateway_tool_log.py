"""
Gateway tool-call log
=====================
Calls one API-based tool (order-tracker) and one Lambda-based tool
(refund-processor) directly through the Gateway and prints the tool name,
the input and the raw result of each call. Screenshot or save the output.

Run:  uv run python gateway_tool_log.py | tee gateway_tool_log.txt
"""

import json

from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client

GATEWAY_URL = "https://customersupportgateway-dazkqhzicj.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"

client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))

with client:
    tools = list(client.list_tools_sync())
    names = [t.tool_name for t in tools]
    print("GATEWAY TOOLS:")
    for n in names:
        print("  -", n)
    print()

    def find(suffix):
        return next(n for n in names if n.endswith(suffix))

    calls = [
        # API Gateway target (order-tracker)
        (find("track_order"), {"order_id": "ORD-001"}),
        # Lambda target (refund-processor), with the real item price as the amount
        (find("initiate_refund"), {"order_id": "ORD-002", "amount": 139.99}),
    ]

    for i, (name, args) in enumerate(calls, start=1):
        print(f"=== CALL {i} ===")
        print("TOOL :", name)
        print("INPUT:", json.dumps(args))
        result = client.call_tool_sync(
            tool_use_id=f"log-{i}", name=name, arguments=args
        )
        print("RESULT:")
        print(json.dumps(result, indent=2, default=str))
        print()
