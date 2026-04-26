#!/usr/bin/env python3
"""
Analytics MCP Server - VERO Fashion
Provides browsing behavior and customer segmentation from verodata.db
Port: 9003
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

app = FastAPI(title="VERO Analytics MCP Server", version="1.0.0")
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
    return {"service": "VERO Analytics MCP Server", "version": "1.0.0", "protocol": "MCP"}


@app.get("/health")
async def health():
    try:
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) FROM browsing_activity").fetchone()[0]
        conn.close()
        return {"status": "healthy", "records": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/mcp/tools/list")
async def list_tools():
    return {"tools": [
        {
            "name": "get_browsing_behavior",
            "description": "Get VERO customer browsing and engagement data",
            "inputSchema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"]
            }
        },
        {
            "name": "get_customer_segment",
            "description": "Get VERO customer segmentation and predictive analytics",
            "inputSchema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"]
            }
        }
    ]}


@app.post("/mcp/tools/call")
async def call_tool(request: MCPToolRequest):
    logger.info(f"MCP Tool Call: {request.tool}")

    if request.tool == "get_browsing_behavior":
        cust_id = request.arguments.get("customer_id")
        conn = get_db()
        row = conn.execute("SELECT * FROM browsing_activity WHERE customer_id = ?", (cust_id,)).fetchone()
        conn.close()
        if row:
            data = {
                "pages_viewed": row["pages_viewed"],
                "time_on_site": row["time_on_site"],
                "sessions": row["sessions"],
                "categories_viewed": json.loads(row["categories_viewed"]),
                "products_viewed": json.loads(row["products_viewed"]),
                "cart_abandonment": row["cart_abandonment"],
                "wishlist_items": json.loads(row["wishlist_items"]),
                "search_queries": json.loads(row["search_queries"])
            }
            return {"success": True, "data": data}
        return {"success": False, "error": "not_found"}

    elif request.tool == "get_customer_segment":
        cust_id = request.arguments.get("customer_id")
        conn = get_db()
        row = conn.execute("SELECT * FROM customer_segments WHERE customer_id = ?", (cust_id,)).fetchone()
        conn.close()
        if row:
            data = {
                "segment": row["segment"],
                "segment_id": row["segment_id"],
                "engagement_score": row["engagement_score"],
                "purchase_probability": row["purchase_probability"],
                "churn_risk": row["churn_risk"],
                "lifetime_value": row["lifetime_value"],
                "predicted_ltv": row["predicted_ltv"],
                "characteristics": json.loads(row["characteristics"])
            }
            return {"success": True, "data": data}
        return {"success": False, "error": "not_found"}

    return {"error": "unknown_tool"}


if __name__ == "__main__":
    print("VERO ANALYTICS MCP SERVER - Port 9003")
    uvicorn.run("analytics_mcp_server:app", host="0.0.0.0", port=9003, reload=False)
