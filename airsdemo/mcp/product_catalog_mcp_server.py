#!/usr/bin/env python3
"""
Product Catalog MCP Server - VERO Fashion
Provides product information and inventory from verodata.db
Port: 9005
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

app = FastAPI(title="VERO Product Catalog MCP Server", version="1.0.0")
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
    return {"service": "VERO Product Catalog MCP Server", "version": "1.0.0", "protocol": "MCP"}


@app.get("/health")
async def health():
    try:
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        conn.close()
        return {"status": "healthy", "products": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/mcp/tools/list")
async def list_tools():
    return {"tools": [
        {
            "name": "check_inventory",
            "description": "Check VERO inventory status for products",
            "inputSchema": {
                "type": "object",
                "properties": {"product_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["product_ids"]
            }
        },
        {
            "name": "get_product_details",
            "description": "Get detailed VERO product information",
            "inputSchema": {
                "type": "object",
                "properties": {"product_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["product_ids"]
            }
        }
    ]}


@app.post("/mcp/tools/call")
async def call_tool(request: MCPToolRequest):
    logger.info(f"MCP Tool Call: {request.tool}")

    if request.tool == "check_inventory":
        product_ids = request.arguments.get("product_ids", [])
        conn = get_db()
        inventory = []
        for pid in product_ids:
            row = conn.execute(
                "SELECT product_id, in_stock, stock_quantity FROM products WHERE product_id = ?",
                (pid,)
            ).fetchone()
            if row:
                inventory.append({
                    "product_id": row["product_id"],
                    "in_stock": bool(row["in_stock"]),
                    "quantity": row["stock_quantity"]
                })
        conn.close()
        return {"success": True, "data": {"products_checked": product_ids, "inventory_status": inventory}}

    elif request.tool == "get_product_details":
        product_ids = request.arguments.get("product_ids", [])
        conn = get_db()
        products = []
        for pid in product_ids:
            row = conn.execute("SELECT * FROM products WHERE product_id = ?", (pid,)).fetchone()
            if row:
                products.append({
                    "id": row["product_id"],
                    "name": row["name"],
                    "category": row["category"],
                    "price": row["price"],
                    "sale_price": row["sale_price"],
                    "material": row["material"],
                    "color": row["color"],
                    "description": row["description"],
                    "rating": row["rating"],
                    "reviews": row["review_count"],
                    "in_stock": bool(row["in_stock"]),
                    "quantity": row["stock_quantity"]
                })
        conn.close()
        return {"success": True, "data": {"products": products}}

    return {"error": "unknown_tool"}


if __name__ == "__main__":
    print("VERO PRODUCT CATALOG MCP SERVER - Port 9005")
    uvicorn.run("product_catalog_mcp_server:app", host="0.0.0.0", port=9005, reload=False)
