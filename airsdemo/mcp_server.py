import re
import asyncio
import json
import logging
import sys
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from mcp.server.models import InitializationOptions
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class SemanticDatabaseProvider:
    """Semantic search using embeddings - understands meaning, not just keywords"""

    def __init__(self, db_path: str, vertical_name: str):
        self.db_path = db_path
        self.vertical_name = vertical_name

        print(f"🧠 {vertical_name}: Loading semantic search...", file=sys.stderr)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"✅ {vertical_name}: Ready with real-time database queries", file=sys.stderr)

    def _load_products(self):
        """Load products from database in real-time with category information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Join with categories table to get category name for better semantic matching
        cursor.execute("""
            SELECT p.product_id, p.name, p.brand, p.price, p.sale_price, p.stock_quantity,
                   p.description, p.specifications, c.name as category_name
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.category_id
        """)
        rows = cursor.fetchall()
        conn.close()

        products = []
        product_texts = []
        for row in rows:
            # Include category in the text for semantic matching
            category = row[8] or ''
            text = f"{category} {row[1]} {row[2] or ''} {row[6] or ''} {row[7] or ''}".strip()

            products.append({
                'product_id': row[0], 'name': row[1], 'brand': row[2],
                'price': row[3], 'sale_price': row[4], 'stock_quantity': row[5],
                'description': row[6], 'specifications': row[7], 'category': category
            })
            product_texts.append(text)

        # Create embeddings
        embeddings = self.model.encode(product_texts, convert_to_numpy=True)
        return products, embeddings, product_texts

    async def search_content(self, query: str, limit: int = 5, user_context: Dict = None) -> Dict[str, Any]:
        """Search by meaning, not exact keywords - loads fresh data from database each time"""
        try:
            print(f"🔍 Semantic search: '{query}'", file=sys.stderr)

            # Load fresh data from database
            products, embeddings, product_texts = self._load_products()
            print(f"📊 Loaded {len(products)} products from database", file=sys.stderr)

            # Encode query and find similar products
            query_emb = self.model.encode([query])[0]
            similarities = np.dot(embeddings, query_emb) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
            )

            # Get top matches above threshold
            top_idx = np.argsort(similarities)[::-1][:limit]
            results = []

            for idx in top_idx:
                if similarities[idx] < 0.25:  # Skip low matches
                    continue

                p = products[idx]
                price = f"${p['sale_price']}" if p['sale_price'] else f"${p['price']}"

                results.append({
                    "document": "Luna Tech Product Catalog",
                    "snippets": [
                        f"Product: {p['name']}" + (f" by {p['brand']}" if p['brand'] else ""),
                        f"Price: {price}, Stock: {p['stock_quantity']} units available",
                        f"Description: {p['description']}" if p['description'] else None,
                        f"Specs: {p['specifications']}" if p['specifications'] else None
                    ],
                    "relevance_score": float(similarities[idx]) * 100
                })
                print(f"  📊 {p['name']}: {similarities[idx]:.2f}", file=sys.stderr)

            print(f"✅ Found {len(results)} matches", file=sys.stderr)
            return {
                "query": query,
                "results": results,
                "result_count": len(results),
                "vertical": self.vertical_name
            }
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return {"query": query, "results": [], "result_count": 0}


class FastDocumentProvider:
    """Fast keyword document search with name boosting"""

    def __init__(self, docs_dir: Path, vertical_name: str):
        self.docs_dir = docs_dir
        self.vertical_name = vertical_name
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.documents = []
        self._load_documents()
        print(f"⚡ {vertical_name}: {len(self.documents)} documents ready", file=sys.stderr)

    def _read_document(self, file_path: Path) -> str:
        """Read document"""
        try:
            if file_path.suffix.lower() == '.docx':
                try:
                    from docx import Document
                    doc = Document(file_path)
                    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
                except:
                    return ""
            return file_path.read_text(encoding='utf-8', errors='replace')
        except:
            return ""

    def _load_documents(self):
        """Load all documents"""
        print(f"📂 {self.vertical_name}: Loading documents from {self.docs_dir}", file=sys.stderr)
        for file_path in self.docs_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.docx']:
                content = self._read_document(file_path)
                if content.strip():
                    self.documents.append({
                        'path': file_path,
                        'relative_path': file_path.relative_to(self.docs_dir),
                        'content': content,
                        'filename': file_path.stem.lower(),
                        'folder': file_path.parent.name
                    })
                    print(f"  ✅ Loaded: {file_path.name} ({len(content)} chars)", file=sys.stderr)
                else:
                    print(f"  ⚠️  Skipped (empty): {file_path.name}", file=sys.stderr)

    def _extract_snippets(self, content: str, keywords: List[str], doctor_name: str = None) -> List[str]:
        """Extract relevant snippets - prioritize complete sections"""

        # If looking for specific doctor, find their paragraph
        if doctor_name:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if doctor_name.lower() in line.lower():
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    snippet = '\n'.join(lines[start:end]).strip()
                    return [snippet] if snippet else [content[:500]]

        content_lower = content.lower()

        # For insurance queries, extract the complete insurance section
        if any(kw in ['insurance', 'carrier', 'carriers', 'plan', 'plans', 'accept'] for kw in keywords):
            insurance_markers = [
                'insurance do',
                'accept',
                'insurance plans',
                'insurance carriers',
                'insurance providers'
            ]

            for marker in insurance_markers:
                if marker in content_lower:
                    idx = content_lower.find(marker)
                    snippet_start = max(0, idx - 100)
                    snippet_end = min(len(content), idx + 2000)
                    insurance_section = content[snippet_start:snippet_end]

                    lines = insurance_section.split('\n')
                    clean_lines = []
                    heading_count = 0
                    for line in lines:
                        if re.match(r'^\d+\.', line.strip()):
                            heading_count += 1
                            if heading_count > 1:
                                break
                        clean_lines.append(line)

                    return ['\n'.join(clean_lines).strip()]

        # For doctor directory queries
        if any(kw in ['doctor', 'doctors', 'directory', 'provider', 'providers'] for kw in keywords):
            if 'featured provider' in content_lower:
                idx = content_lower.find('featured provider')
                snippet_start = max(0, idx - 50)
                snippet_end = min(len(content), idx + 1500)
                featured_section = content[snippet_start:snippet_end]

                intro_idx = content_lower.find('at healthbridge')
                if intro_idx > 0:
                    intro = content[max(0, intro_idx - 100):intro_idx + 300]
                    return [intro.strip(), featured_section.strip()]

                return [featured_section.strip()]

        # General keyword search
        sentences = re.split(r'[.!?\n]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return [content[:1000]]

        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                relevant.append(sent)
                if len(relevant) >= 10:
                    break

        return relevant[:10] if relevant else sentences[:5]

    async def search_content(self, query: str, limit: int = 5, user_context: Dict = None) -> Dict[str, Any]:
        """Fast keyword search with name boosting"""
        query_lower = query.lower()

        # Extract keywords
        stopwords = {'show', 'me', 'your', 'the', 'a', 'an', 'what', 'how', 'can', 'do', 'you', 'tell', 'about', 'more'}
        keywords = [w for w in re.findall(r'\b\w{3,}\b', query_lower) if w not in stopwords]

        print(f"🔍 {self.vertical_name} - Query: '{query}'", file=sys.stderr)
        print(f"📝 {self.vertical_name} - Keywords: {keywords}", file=sys.stderr)
        print(f"📚 {self.vertical_name} - Total documents loaded: {len(self.documents)}", file=sys.stderr)

        # Check for doctor name
        doctor_name = None
        doctor_match = re.search(r'(?:Dr\.|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
        if doctor_match:
            doctor_name = doctor_match.group(1)
            print(f"🔍 Looking for doctor: {doctor_name}", file=sys.stderr)

        results = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            score = 0

            # MASSIVE BOOST for doctor name match
            if doctor_name and doctor_name.lower() in content_lower:
                score += 1000
                print(f"  🎯 Found {doctor_name} in {doc['relative_path']}", file=sys.stderr)

            # Keyword matching
            for kw in keywords:
                if kw in doc['filename']:
                    score += 20
                score += content_lower.count(kw) * 5

            # Folder boost
            if 'doctor' in doc['folder'].lower() or 'dr' in doc['folder'].lower():
                score += 50

            if score > 0:
                snippets = self._extract_snippets(doc['content'], keywords, doctor_name)
                results.append({
                    'document': str(doc['relative_path']),
                    'snippets': snippets,
                    'relevance_score': score,
                    'folder': doc['folder']
                })

        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        print(f"✅ {self.vertical_name} - Found {len(results)} results (returning top {min(limit, len(results))})", file=sys.stderr)
        for i, r in enumerate(results[:limit], 1):
            print(f"  {i}. {r['document']} (score: {r['relevance_score']})", file=sys.stderr)

        return {
            "query": query,
            "results": results[:limit],
            "result_count": len(results),
            "vertical": self.vertical_name
        }


class FastMCPServer:
    """Fast MCP server - semantic search for retail, keyword for others"""

    def __init__(self, webroot_dir: str = "."):
        self.server = Server("fast-multi-vertical")
        self.webroot_dir = Path(webroot_dir)

        print(f"\n⚡ FAST MCP SERVER", file=sys.stderr)

        # FIXED: Using SemanticDatabaseProvider instead of FastDatabaseProvider
        self.providers = {
            "luna-tech": SemanticDatabaseProvider(
                str(self.webroot_dir / "retail" / "luna_tech.db"),
                "Luna Tech"
            ),
            "healthbridge": FastDocumentProvider(
                self.webroot_dir / "healthcare" / "documents",
                "HealthBridge"
            ),
            "enterprise": FastDocumentProvider(
                self.webroot_dir / "ent" / "documents",
                "Enterprise Corp"
            ),
            "finance": FastDocumentProvider (
                self.webroot_dir / "finance" / "documents",
                "SecureBank"
            ),
            "gaming": FastDocumentProvider(
                self.webroot_dir / "gaming" / "documents",
                "Mt Olympus Casino & Hotel"
            )
        }

        print(f"✅ All providers ready\n", file=sys.stderr)

        self.setup_resources()
        self.setup_tools()

    def setup_resources(self):
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(uri="docs://luna-tech/catalog", name="Luna Tech Products", mimeType="text/plain"),
                Resource(uri="docs://healthbridge/services", name="HealthBridge Services", mimeType="text/plain"),
                Resource(uri="docs://enterprise/policies", name="Enterprise Policies", mimeType="text/plain")
            ]

    def setup_tools(self):
        @self.server.list_tools()
        async def list_tools () -> List[Tool]:
            return [
                Tool (
                    name="search_documents",
                    description="Fast keyword search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vertical": {
                                "type": "string",
                                "enum": ["luna-tech", "healthbridge", "enterprise", "finance", "gaming"]
                            },
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 5},
                            "user_context": {"type": "object"}
                        },
                        "required": ["vertical", "query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "search_documents":
                vertical = arguments["vertical"]
                query = arguments["query"]
                limit = arguments.get("limit", 5)
                user_context = arguments.get("user_context")

                if vertical not in self.providers:
                    return [TextContent(type="text", text=json.dumps(
                        {"error": f"Unknown vertical", "results": [], "result_count": 0}))]

                results = await self.providers[vertical].search_content(query, limit, user_context)

                if not results:
                    results = {"results": [], "result_count": 0, "query": query}
                if "result_count" not in results:
                    results["result_count"] = len(results.get("results", []))

                return [TextContent(type="text", text=json.dumps(results, indent=2))]

            return [TextContent(type="text", text=json.dumps({"error": "Unknown tool", "results": []}))]


async def main():
    mcp_server = FastMCPServer(webroot_dir="/var/www/airsdemo")
    init_options = InitializationOptions(server_name="fast-multi-vertical", server_version="4.0.0", capabilities={})
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(sys.stderr)])
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass