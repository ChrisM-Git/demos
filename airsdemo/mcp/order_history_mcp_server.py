#!/usr/bin/env python3
"""
Order History MCP Server - VERO Fashion
Provides purchase history and order data from verodata.db
Port: 9002
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict
from pathlib import Path
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).parent / "verodata.db")

app = FastAPI(title="VERO Order History MCP Server", version="1.0.0")
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
    return {"service": "VERO Order History MCP Server", "version": "1.0.0", "protocol": "MCP"}


@app.get("/health")
async def health():
    try:
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        return {"status": "healthy", "total_orders": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/mcp/tools/list")
async def list_tools():
    return {"tools": [
        {
            "name": "get_order_history",
            "description": "Get complete VERO order history for a customer",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "limit": {"type": "integer", "description": "Max orders to return", "default": 10}
                },
                "required": ["customer_id"]
            }
        },
        {
            "name": "get_order_summary",
            "description": "Get summary statistics for VERO customer orders",
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

    if tool == "get_order_history":
        return await get_order_history(arguments.get("customer_id"), arguments.get("limit", 10))
    elif tool == "get_order_summary":
        return await get_order_summary(arguments.get("customer_id"))
    return {"error": "unknown_tool", "message": f"Tool '{tool}' not found"}


async def get_order_history(customer_id: str, limit: int = 10):
    logger.info(f"Getting order history for {customer_id} (limit: {limit})")
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM orders WHERE customer_id = ? ORDER BY date DESC LIMIT ?",
        (customer_id, limit)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM orders WHERE customer_id = ?", (customer_id,)).fetchone()[0]
    conn.close()

    if rows:
        orders = []
        for r in rows:
            orders.append({
                "order_id": r["order_id"],
                "date": r["date"],
                "product": r["product_name"],
                "product_id": r["product_id"],
                "category": r["category"],
                "amount": r["amount"],
                "quantity": r["quantity"],
                "status": r["status"],
                "rating": r["rating"],
                "review": r["review"]
            })
        logger.info(f"Found {len(orders)} orders (total: {total})")
        return {"success": True, "data": {"customer_id": customer_id, "orders": orders, "total_orders": total}}

    logger.warning(f"No orders found for {customer_id}")
    return {"success": False, "error": "no_orders_found", "message": f"No orders found for {customer_id}"}


async def get_order_summary(customer_id: str):
    logger.info(f"Getting order summary for {customer_id}")
    conn = get_db()
    rows = conn.execute("SELECT * FROM orders WHERE customer_id = ?", (customer_id,)).fetchall()
    conn.close()

    if rows:
        total_orders = len(rows)
        total_spent = sum(r["amount"] * r["quantity"] for r in rows)
        avg_order = total_spent / total_orders

        categories = {}
        for r in rows:
            cat = r["category"]
            categories[cat] = categories.get(cat, 0) + 1

        ratings = [r["rating"] for r in rows if r["rating"]]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        recent = []
        for r in sorted(rows, key=lambda x: x["date"], reverse=True)[:3]:
            recent.append({
                "order_id": r["order_id"],
                "date": r["date"],
                "product": r["product_name"],
                "amount": r["amount"]
            })

        summary = {
            "customer_id": customer_id,
            "total_orders": total_orders,
            "total_spent": round(total_spent, 2),
            "average_order": round(avg_order, 2),
            "categories": categories,
            "average_rating": round(avg_rating, 1),
            "recent_orders": recent
        }
        logger.info(f"Summary: {total_orders} orders, ${total_spent:.2f} total")
        return {"success": True, "data": summary}

    logger.warning(f"No orders found for {customer_id}")
    return {"success": False, "error": "no_orders_found", "message": f"No orders found for {customer_id}"}


if __name__ == "__main__":
    print("=" * 60)
    print("VERO ORDER HISTORY MCP SERVER")
    print("=" * 60)
    print("Port: 9002 | Protocol: MCP")
    print("Database:", DB_PATH)
    print("=" * 60)
    uvicorn.run("order_history_mcp_server:app", host="0.0.0.0", port=9002, reload=False)
