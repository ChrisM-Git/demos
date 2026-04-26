#!/usr/bin/env python3
"""
Notification MCP Server - VERO Fashion
Handles internal notifications and email
Port: 9006
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict
from datetime import datetime
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VERO Notification MCP Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

NOTIFICATIONS_SENT = []


class MCPToolRequest(BaseModel):
    tool: str
    arguments: Dict


@app.get("/")
async def root():
    return {"service": "VERO Notification MCP Server", "version": "1.0.0", "protocol": "MCP"}


@app.get("/health")
async def health():
    return {"status": "healthy", "notifications_sent": len(NOTIFICATIONS_SENT)}


@app.post("/mcp/tools/list")
async def list_tools():
    return {"tools": [
        {
            "name": "send_internal_notification",
            "description": "Send VERO internal notification through secure channel",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "type": {"type": "string"},
                    "recommendations": {"type": "array"}
                },
                "required": ["customer_id", "type"]
            }
        },
        {
            "name": "send_email",
            "description": "Send email to VERO customer",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    ]}


@app.post("/mcp/tools/call")
async def call_tool(request: MCPToolRequest):
    logger.info(f"MCP Tool Call: {request.tool}")

    if request.tool == "send_internal_notification":
        notif_id = "NOTIF-" + str(uuid.uuid4())[:8]
        notif = {
            "notification_id": notif_id,
            "customer_id": request.arguments.get("customer_id"),
            "type": request.arguments.get("type"),
            "channel": "vero_internal_dashboard",
            "timestamp": datetime.now().isoformat(),
            "status": "delivered"
        }
        NOTIFICATIONS_SENT.append(notif)
        logger.info(f"VERO internal notification sent: {notif_id}")
        return {"success": True, "data": notif}

    elif request.tool == "send_email":
        msg_id = "MSG-" + str(uuid.uuid4())[:8]
        email = {
            "message_id": msg_id,
            "to": request.arguments.get("to"),
            "subject": request.arguments.get("subject"),
            "timestamp": datetime.now().isoformat(),
            "status": "sent"
        }
        logger.info(f"VERO email sent: {msg_id}")
        return {"success": True, "data": email}

    return {"error": "unknown_tool"}


if __name__ == "__main__":
    print("VERO NOTIFICATION MCP SERVER - Port 9006")
    uvicorn.run("notification_mcp_server:app", host="0.0.0.0", port=9006, reload=False)
