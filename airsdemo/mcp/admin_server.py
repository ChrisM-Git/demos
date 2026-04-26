#!/usr/bin/env python3
"""
VERO Product Admin API
Simple CRUD for managing products in verodata.db
Port: 9010
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).parent / "verodata.db")

app = FastAPI(title="VERO Product Admin API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class ProductIn(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    sale_price: Optional[float] = None
    material: str = ""
    color: str = ""
    description: str = ""
    rating: float = 4.8
    review_count: int = 100
    in_stock: int = 1
    stock_quantity: int = 20


@app.get("/")
async def root():
    return {"service": "VERO Product Admin API"}


@app.get("/health")
async def health():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    conn.close()
    return {"status": "healthy", "products": count}


@app.get("/api/products")
async def list_products(category: Optional[str] = None):
    conn = get_db()
    if category:
        rows = conn.execute("SELECT * FROM products WHERE category = ? ORDER BY name", (category,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM products ORDER BY category, name").fetchall()
    conn.close()
    return {"products": [dict(r) for r in rows], "count": len(rows)}


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")
    return dict(row)


@app.post("/api/products")
async def create_product(product: ProductIn):
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO products (product_id, name, category, price, sale_price, material, color,
                                  description, rating, review_count, in_stock, stock_quantity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (product.product_id, product.name, product.category, product.price, product.sale_price,
              product.material, product.color, product.description, product.rating,
              product.review_count, product.in_stock, product.stock_quantity))
        conn.commit()
        conn.close()
        logger.info(f"Created product: {product.product_id} - {product.name}")
        return {"success": True, "product_id": product.product_id}
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail=f"Product {product.product_id} already exists")


@app.put("/api/products/{product_id}")
async def update_product(product_id: str, product: ProductIn):
    conn = get_db()
    existing = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Product not found")
    conn.execute("""
        UPDATE products SET name=?, category=?, price=?, sale_price=?, material=?, color=?,
                            description=?, rating=?, review_count=?, in_stock=?, stock_quantity=?
        WHERE product_id=?
    """, (product.name, product.category, product.price, product.sale_price, product.material,
          product.color, product.description, product.rating, product.review_count,
          product.in_stock, product.stock_quantity, product_id))
    conn.commit()
    conn.close()
    logger.info(f"Updated product: {product_id}")
    return {"success": True, "product_id": product_id}


@app.delete("/api/products/{product_id}")
async def delete_product(product_id: str):
    conn = get_db()
    conn.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted product: {product_id}")
    return {"success": True}


@app.get("/api/categories")
async def list_categories():
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT category, COUNT(*) as count FROM products GROUP BY category ORDER BY category").fetchall()
    conn.close()
    return {"categories": [{"name": r["category"], "count": r["count"]} for r in rows]}


if __name__ == "__main__":
    import uvicorn
    print("VERO PRODUCT ADMIN API - Port 9010")
    print("Database:", DB_PATH)
    uvicorn.run("admin_server:app", host="0.0.0.0", port=9010, reload=False)
