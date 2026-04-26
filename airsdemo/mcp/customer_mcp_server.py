#!/usr/bin/env python3
"""
Customer Service MCP Server - VERO Fashion
Provides customer profile and preferences data from verodata.db
Port: 9001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict
from pathlib import Path
import sqlite3
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).parent / "verodata.db")

app = FastAPI(title="VERO Customer Service MCP Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class MCPToolRequest(BaseModel):
    tool: str
    arguments: Dict


@app.get("/")
async def root():
    return {"service": "VERO Customer Service MCP Server", "version": "1.0.0", "protocol": "MCP"}


@app.get("/health")
async def health():
    try:
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        conn.close()
        return {"status": "healthy", "customers": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/mcp/tools/list")
async def list_tools():
    return {"tools": [
        {
            "name": "get_customer_profile",
            "description": "Get VERO customer profile including name, contact, tier, and loyalty points",
            "inputSchema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "Customer ID (e.g., CUST-10001)"}},
                "required": ["customer_id"]
            }
        },
        {
            "name": "get_customer_preferences",
            "description": "Get VERO customer preferences including favorite categories, brands, and notification settings",
            "inputSchema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "Customer ID"}},
                "required": ["customer_id"]
            }
        }
    ]}


@app.post("/mcp/tools/call")
async def call_tool(request: MCPToolRequest):
    tool = request.tool
    arguments = request.arguments
    logger.info(f"MCP Tool Call: {tool} | Args: {arguments}")

    if tool == "get_customer_profile":
        return await get_customer_profile(arguments.get("customer_id"))
    elif tool == "get_customer_preferences":
        return await get_customer_preferences(arguments.get("customer_id"))
    return {"error": "unknown_tool", "message": f"Tool '{tool}' not found"}


async def get_customer_profile(customer_id: str):
    logger.info(f"Getting profile for {customer_id}")
    conn = get_db()
    row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    conn.close()

    if row:
        profile = {
            "customer_id": row["customer_id"],
            "name": row["name"],
            "email": row["email"],
            "tier": row["tier"],
            "member_since": row["member_since"],
            "loyalty_points": row["loyalty_points"],
            "phone": row["phone"],
            "address": {
                "street": row["address_street"],
                "city": row["address_city"],
                "state": row["address_state"],
                "zip": row["address_zip"]
            }
        }
        logger.info(f"Found customer: {profile['name']}")
        return {"success": True, "data": profile}

    logger.warning(f"Customer {customer_id} not found")
    return {"success": False, "error": "customer_not_found", "message": f"Customer {customer_id} not found"}


async def get_customer_preferences(customer_id: str):
    logger.info(f"Getting preferences for {customer_id}")
    conn = get_db()
    row = conn.execute("SELECT * FROM customer_preferences WHERE customer_id = ?", (customer_id,)).fetchone()
    conn.close()

    if row:
        prefs = {
            "categories": json.loads(row["favorite_categories"]),
            "brands": json.loads(row["favorite_brands"]),
            "price_range": row["price_range"],
            "preferred_delivery": row["preferred_delivery"],
            "notification_preferences": {
                "email": bool(row["notify_email"]),
                "sms": bool(row["notify_sms"]),
                "push": bool(row["notify_push"])
            }
        }
        logger.info(f"Found preferences: {prefs['categories']}")
        return {"success": True, "data": prefs}

    logger.warning(f"Preferences for {customer_id} not found")
    return {"success": False, "error": "preferences_not_found", "message": f"Preferences for {customer_id} not found"}


if __name__ == "__main__":
    print("=" * 60)
    print("VERO CUSTOMER SERVICE MCP SERVER")
    print("=" * 60)
    print("Port: 9001 | Protocol: MCP")
    print("Database:", DB_PATH)
    print("=" * 60)
    uvicorn.run("customer_mcp_server:app", host="0.0.0.0", port=9001, reload=False)
