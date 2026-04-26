#!/usr/bin/env python3
"""
ML Recommendation MCP Server - VERO Fashion
Provides ML-based product recommendations from verodata.db
Port: 9004
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

app = FastAPI(title="VERO ML Recommendation MCP Server", version="1.0.0")
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
    return {"service": "VERO ML Recommendation MCP Server", "version": "1.0.0", "protocol": "MCP"}


@app.get("/health")
async def health():
    try:
        conn = get_db()
        count = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        conn.close()
        return {"status": "healthy", "recommendations": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/mcp/tools/list")
async def list_tools():
    return {"tools": [
        {
            "name": "find_similar_customers",
            "description": "Find similar VERO customers using collaborative filtering",
            "inputSchema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"]
            }
        },
        {
            "name": "get_product_recommendations",
            "description": "Get ML-based VERO product recommendations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["customer_id"]
            }
        }
    ]}


@app.post("/mcp/tools/call")
async def call_tool(request: MCPToolRequest):
    logger.info(f"MCP Tool Call: {request.tool}")

    if request.tool == "find_similar_customers":
        cust_id = request.arguments.get("customer_id")
        conn = get_db()
        row = conn.execute("SELECT * FROM similar_customers WHERE customer_id = ?", (cust_id,)).fetchone()
        conn.close()
        if row:
            data = {
                "similar_customers": row["similar_count"],
                "similarity_score": row["similarity_score"],
                "common_purchases": json.loads(row["common_purchases"]),
                "collaborative_filtering_applied": True
            }
            return {"success": True, "data": data}
        return {"success": False, "error": "not_found"}

    elif request.tool == "get_product_recommendations":
        cust_id = request.arguments.get("customer_id")
        limit = request.arguments.get("limit", 5)
        conn = get_db()
        rows = conn.execute("""
            SELECT r.product_id, r.ml_score, r.reason, r.model_version,
                   p.name, p.price, p.category, p.material, p.color
            FROM recommendations r
            JOIN products p ON r.product_id = p.product_id
            WHERE r.customer_id = ?
            ORDER BY r.ml_score DESC
            LIMIT ?
        """, (cust_id, limit)).fetchall()
        conn.close()

        if rows:
            recs = []
            for r in rows:
                recs.append({
                    "product_id": r["product_id"],
                    "name": r["name"],
                    "price": r["price"],
                    "category": r["category"],
                    "material": r["material"],
                    "color": r["color"],
                    "ml_score": r["ml_score"],
                    "reason": r["reason"]
                })
            data = {
                "recommendations": recs,
                "model_version": rows[0]["model_version"] if rows else "v2.4.1",
                "confidence": "high"
            }
            return {"success": True, "data": data}
        return {"success": False, "error": "not_found"}

    return {"error": "unknown_tool"}


if __name__ == "__main__":
    print("VERO ML RECOMMENDATION MCP SERVER - Port 9004")
    uvicorn.run("ml_recommendation_mcp_server:app", host="0.0.0.0", port=9004, reload=False)
