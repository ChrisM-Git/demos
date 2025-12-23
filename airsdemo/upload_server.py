
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uvicorn
from datetime import datetime
from typing import List, Optional, Dict, Any
import sqlite3
import json


app = FastAPI(title="MCP Document Upload Server")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Document directories
DOCUMENT_DIRS = {
    "retail": "./retail/documents",
    "healthcare": "./healthcare/documents",
    "enterprise": "./ent/documents",
    "finance": "./finance/documents",
    "gaming": "./gaming/documents"
}
# Database configuration
DATABASE_FILES = {
    "retail": "/var/www/airsdemo/retail/luna_tech.db",  # Absolute path for production
    "healthcare": None,
    "enterprise": None,
    "finance": None,
    "gaming": "/var/www/airsdemo/gaming/mt_olympus.db"
}


# Ensure directories exist
for dir_path in DOCUMENT_DIRS.values():
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_allowed_extensions():
    """Get list of allowed file extensions"""
    return {'.txt', '.md', '.pdf', '.docx', '.doc'}


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in get_allowed_extensions()


def get_document_list(vertical: str) -> List[dict]:
    """Get list of documents in a vertical"""
    doc_dir = Path(DOCUMENT_DIRS[vertical])
    if not doc_dir.exists():
        return []

    documents = []
    for file_path in doc_dir.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            documents.append({
                'name': file_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'type': file_path.suffix.lower()
            })

    return sorted(documents, key=lambda x: x['modified'], reverse=True)


def get_db_connection(vertical: str):
    """Get database connection for vertical"""
    if vertical not in DATABASE_FILES or DATABASE_FILES[vertical] is None:
        return None

    db_path = Path(DATABASE_FILES[vertical])
    if not db_path.exists():
        return None

    return sqlite3.connect(DATABASE_FILES[vertical])


def dict_factory(cursor, row):
    """Convert SQLite row to dict"""
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}



@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MCP Document Upload Server",
        "status": "running",
        "verticals": list(DOCUMENT_DIRS.keys()),
        "total_documents": sum(len(get_document_list(v)) for v in DOCUMENT_DIRS.keys())
    }


@app.post("/upload/{vertical}")
async def upload_documents(vertical: str, files: List[UploadFile] = File(...)):
    """Upload documents to specified vertical"""
    if vertical not in DOCUMENT_DIRS:
        raise HTTPException(status_code=400, detail=f"Invalid vertical: {vertical}")

    upload_dir = Path(DOCUMENT_DIRS[vertical])
    uploaded_files = []
    skipped_files = []

    for file in files:
        if not is_allowed_file(file.filename):
            skipped_files.append(f"{file.filename} (unsupported format)")
            continue

        # Create safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in '.-_ ').strip()
        if not safe_filename:
            safe_filename = f"uploaded_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        file_path = upload_dir / safe_filename

        # Handle duplicate filenames
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            file_path = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file_path.name)
        except Exception as e:
            skipped_files.append(f"{file.filename} (save error: {str(e)})")

    return {
        "message": f"Upload completed for {vertical}",
        "uploaded_count": len(uploaded_files),
        "files": uploaded_files,
        "skipped": skipped_files,
        "skipped_count": len(skipped_files)
    }


@app.delete("/delete/{vertical}/{filename}")
async def delete_document(vertical: str, filename: str):
    """Delete a document"""
    if vertical not in DOCUMENT_DIRS:
        raise HTTPException(status_code=400, detail=f"Invalid vertical: {vertical}")

    file_path = Path(DOCUMENT_DIRS[vertical]) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    try:
        file_path.unlink()
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/api/documents/{vertical}")
async def get_documents(vertical: str):
    """Get list of documents for a vertical"""
    if vertical not in DOCUMENT_DIRS:
        raise HTTPException(status_code=400, detail=f"Invalid vertical: {vertical}")

    return get_document_list(vertical)


@app.get("/api/db-tables/{vertical}")
async def get_database_tables(vertical: str):
    """Get all tables and their data for a vertical"""
    conn = get_db_connection(vertical)
    if conn is None:
        return {"tables": [], "message": f"No database configured for {vertical}"}

    try:
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()

        result = {}
        for table in tables:
            table_name = table['name']

            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Get table data (limit to 100 rows for safety)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
            rows = cursor.fetchall()

            result[table_name] = {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()


@app.post("/api/db-insert/{vertical}/{table}")
async def insert_table_row(vertical: str, table: str, data: Dict[str, Any]):
    """Insert a new row into a table"""
    conn = get_db_connection(vertical)
    if conn is None:
        raise HTTPException(status_code=400, detail=f"No database for {vertical}")

    try:
        cursor = conn.cursor()

        # Build INSERT query
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.values()])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        cursor.execute(query, list(data.values()))
        conn.commit()

        return {"message": "Row inserted successfully", "row_id": cursor.lastrowid}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")
    finally:
        conn.close()


@app.put("/api/db-update/{vertical}/{table}/{row_id}")
async def update_table_row(vertical: str, table: str, row_id: int, data: Dict[str, Any]):
    """Update an existing row"""
    conn = get_db_connection(vertical)
    if conn is None:
        raise HTTPException(status_code=400, detail=f"No database for {vertical}")

    try:
        cursor = conn.cursor()

        # Determine correct ID column
        id_column = {
            'products': 'product_id',
            'categories': 'category_id',
            'product_reviews': 'review_id',
            'support_articles': 'article_id',
            'policies': 'policy_id',
            'room_types': 'room_type_id',
            'reservations': 'reservation_id',
            'packages': 'package_id'
        }.get(table, 'id')

        # Build UPDATE query with correct ID column
        set_clause = ', '.join([f"{key} = ?" for key in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {id_column} = ?"

        cursor.execute(query, list(data.values()) + [row_id])
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Row not found")

        return {"message": "Row updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")
    finally:
        conn.close()


@app.delete("/api/db-delete/{vertical}/{table}/{row_id}")
async def delete_table_row(vertical: str, table: str, row_id: int):
    """Delete a row from a table"""
    conn = get_db_connection(vertical)
    if conn is None:
        raise HTTPException(status_code=400, detail=f"No database for {vertical}")

    try:
        cursor = conn.cursor()

        # Determine correct ID column
        id_column = {
            'products': 'product_id',
            'categories': 'category_id',
            'product_reviews': 'review_id',
            'support_articles': 'article_id',
            'policies': 'policy_id',
            'room_types': 'room_type_id',
            'reservations': 'reservation_id',
            'packages': 'package_id'
        }.get(table, 'id')

        cursor.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (row_id,))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Row not found")

        return {"message": "Row deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    finally:
        conn.close()


@app.get("/api/status")
async def get_status():
    """Get overall system status"""
    status = {}
    for vertical in DOCUMENT_DIRS.keys():
        docs = get_document_list(vertical)
        status[vertical] = {
            "document_count": len(docs),
            "total_size": sum(doc['size'] for doc in docs),
            "directory": DOCUMENT_DIRS[vertical],
            "exists": Path(DOCUMENT_DIRS[vertical]).exists()
        }

    return status


if __name__ == "__main__":
    print("=" * 60)
    print("📁 MCP DOCUMENT UPLOAD SERVER")
    print("=" * 60)
    print("Server Details:")
    print(f"  • Port: 8096")
    print(f"  • CORS: Enabled")
    print(f"  • Supported formats: {', '.join(get_allowed_extensions())}")
    print()
    print("Document Directories:")
    for vertical, path in DOCUMENT_DIRS.items():
        doc_count = len(get_document_list(vertical))
        db_status = "✓ DB" if DATABASE_FILES.get(vertical) else ""
        print(f"  • {vertical.title()}: {path} ({doc_count} files) {db_status}")
    print()
    print("Endpoints:")
    print("  • GET  /                              - Health check")
    print("  • POST /upload/{vertical}             - Upload files")
    print("  • DELETE /delete/{vertical}/{file}    - Delete file")
    print("  • GET  /api/documents/{vertical}      - List files")
    print("  • GET  /api/db-tables/{vertical}      - List database tables")
    print("  • POST /api/db-insert/{v}/{table}     - Insert row")
    print("  • PUT  /api/db-update/{v}/{table}/{id} - Update row")
    print("  • DELETE /api/db-delete/{v}/{table}/{id} - Delete row")
    print("  • GET  /api/status                    - System status")
    print("=" * 60)
    print("🌐 Starting server on http://localhost:8096")
    print("💡 Use this with your enhanced landing page!")
    print("=" * 60)

    uvicorn.run(
        "upload_server:app",
        host="0.0.0.0",
        port=8096,
        reload=False,
        log_level="info"
    )