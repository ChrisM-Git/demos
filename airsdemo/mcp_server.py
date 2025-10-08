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

logger = logging.getLogger(__name__)


class FastDatabaseProvider:
    """Fast keyword database search"""

    def __init__(self, db_path: str, vertical_name: str):
        self.db_path = db_path
        self.vertical_name = vertical_name
        print(f"⚡ {vertical_name}: Database ready", file=sys.stderr)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords"""
        stopwords = {'show', 'me', 'your', 'the', 'a', 'an', 'what', 'how', 'are', 'is', 'do', 'you', 'have', 'tell',
                     'about', 'want', 'looking', 'for'}
        words = re.findall(r'\b\w{2,}\b', query.lower())
        return [w for w in words if w not in stopwords]

    async def search_content(self, query: str, limit: int = 5, user_context: Dict = None) -> Dict[str, Any]:
        """Fast keyword search"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            keywords = self._extract_keywords(query)
            if not keywords:
                keywords = ['']

            # Build OR conditions
            conditions = []
            params = []
            for kw in keywords[:10]:  # Limit to 10 keywords
                pattern = f'%{kw}%'
                conditions.append(
                    "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(specifications) LIKE ? OR LOWER(brand) LIKE ?)")
                params.extend([pattern] * 4)

            sql = f"""
                SELECT product_id, name, brand, price, sale_price, stock_quantity, description, specifications
                FROM products
                WHERE {' OR '.join(conditions)}
                LIMIT ?
            """
            params.append(limit * 2)

            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()

            # Format results
            results = []
            for row in rows[:limit]:
                price = f"${row[4]}" if row[4] else f"${row[3]}"
                snippets = [
                    f"Product: {row[1]}" + (f" by {row[2]}" if row[2] else ""),
                    f"Price: {price}, Stock: {row[5]} units available"
                ]
                if row[6]:
                    snippets.append(f"Description: {row[6]}")
                if row[7]:
                    snippets.append(f"Specs: {row[7]}")

                results.append({
                    "document": "Luna Tech Product Catalog",
                    "snippets": snippets,
                    "relevance_score": 100
                })

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
            # Find insurance section
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
                    # Extract large section (2000 chars) to get full list
                    snippet_start = max(0, idx - 100)
                    snippet_end = min(len(content), idx + 2000)
                    insurance_section = content[snippet_start:snippet_end]

                    # Clean up - stop at next major heading
                    lines = insurance_section.split('\n')
                    clean_lines = []
                    heading_count = 0
                    for line in lines:
                        # Stop at second numbered heading (e.g., "3. Medical Departments")
                        if re.match(r'^\d+\.', line.strip()):
                            heading_count += 1
                            if heading_count > 1:
                                break
                        clean_lines.append(line)

                    return ['\n'.join(clean_lines).strip()]

        # For doctor directory queries, prioritize Featured Providers
        if any(kw in ['doctor', 'doctors', 'directory', 'provider', 'providers'] for kw in keywords):
            if 'featured provider' in content_lower:
                idx = content_lower.find('featured provider')
                snippet_start = max(0, idx - 50)
                snippet_end = min(len(content), idx + 1500)  # Increased to 1500
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
                if len(relevant) >= 10:  # Increased to 10
                    break

        return relevant[:10] if relevant else sentences[:5]

    async def search_content(self, query: str, limit: int = 5, user_context: Dict = None) -> Dict[str, Any]:
        """Fast keyword search with name boosting"""
        query_lower = query.lower()

        # Extract keywords
        stopwords = {'show', 'me', 'your', 'the', 'a', 'an', 'what', 'how', 'can', 'do', 'you', 'tell', 'about', 'more'}
        keywords = [w for w in re.findall(r'\b\w{3,}\b', query_lower) if w not in stopwords]

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

        return {
            "query": query,
            "results": results[:limit],
            "result_count": len(results),
            "vertical": self.vertical_name
        }


class FastMCPServer:
    """Fast MCP server - keyword search only"""

    def __init__(self, webroot_dir: str = "."):
        self.server = Server("fast-multi-vertical")
        self.webroot_dir = Path(webroot_dir)

        print(f"\n⚡ FAST MCP SERVER", file=sys.stderr)

        self.providers = {
            "luna-tech": FastDatabaseProvider(
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
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_documents",
                    description="Fast keyword search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vertical": {"type": "string", "enum": ["luna-tech", "healthbridge", "enterprise"]},
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