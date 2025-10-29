#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import uuid
import os
import time
import json
import asyncio
import subprocess
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from queue import Queue
from dotenv import load_dotenv
import logging
import ollama

# Palo Alto Networks AI Security SDK
try:
    import aisecurity
    from aisecurity.scan.inline.scanner import Scanner
    from aisecurity.generated_openapi_client.models.ai_profile import AiProfile
    from aisecurity.scan.models.content import Content

    SECURITY_AVAILABLE = True
    logger_security = logging.getLogger("aisecurity")
    logger_security.info("AI Security SDK imported successfully")
    print("✅ Palo Alto Networks AI Security SDK loaded successfully")
except ImportError as e:
    print(f"⚠️ Palo Alto Networks AI Security SDK not available: {e}")
    print("💡 Install with: pip install pan-aisecurity")


    class Scanner:
        pass


    class AiProfile:
        pass


    class Content:
        def __init__(self, **kwargs):
            pass


    aisecurity = None
    SECURITY_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Ollama queue for serializing requests across agents
_ollama_lock = threading.Lock()


def queued_ollama_chat(**kwargs):
    """Serialize Ollama calls to prevent resource contention"""
    with _ollama_lock:
        return ollama.chat(**kwargs)


# ============================================================================
# PALO ALTO AI SECURITY INITIALIZATION
# ============================================================================

def initialize_vertical_security():
    """Initialize Palo Alto AI Security SDK with vertical-specific profiles"""
    if not SECURITY_AVAILABLE:
        logger.warning("AI Security SDK not available - running without security monitoring")
        return None

    try:
        logger.info("Initializing Palo Alto AI Security SDK with vertical profiles...")
        api_key = os.getenv("PALO_ALTO_API_KEY", "xxxxxxxxxxxxxxxxx")
        logger.info(f"Using API key: {api_key[:8]}...")

        aisecurity.init(api_key=api_key)
        logger.info("AI Security SDK initialized successfully")

        # Create profiles for each vertical
        vertical_profiles = {
            "retail": AiProfile(profile_name="retail"),
            "healthcare": AiProfile(profile_name="healthcare"),
            "enterprise": AiProfile(profile_name="enterprise")
        }

        logger.info(f"Configured profiles: {list(vertical_profiles.keys())}")

        scanner = Scanner()
        logger.info("Scanner instance created successfully")

        # Test the scanner
        try:
            test_content = Content(prompt="hello test")
            test_result = scanner.sync_scan(
                ai_profile=vertical_profiles["retail"],
                content=test_content,
                metadata={"test": "initialization"}
            )
            logger.info("✅ Test scan successful - SDK fully operational")
        except Exception as test_error:
            logger.warning(f"⚠️ Test scan failed but SDK initialized: {test_error}")

        return {
            "profiles": vertical_profiles,
            "scanner": scanner,
            "status": "operational"
        }

    except Exception as e:
        logger.error(f"Failed to initialize AI Security SDK: {e}")
        return None


class SecurityScanner:
    """Security scanner with vertical-specific profiles"""

    def __init__(self, security_components):
        self.security_components = security_components
        self.blocked_verdicts = [
            'block', 'blocked', 'deny', 'denied', 'reject', 'rejected',
            'violate', 'violation', 'threat', 'malicious', 'harmful'
        ]

    def scan_input(self, user_input: str, session_id: str, vertical: str):
        """Scan user input with vertical-specific profile"""
        if not self.security_components or not SECURITY_AVAILABLE:
            return {
                "safe": True,
                "verdict": "no_scan",
                "should_block": False
            }

        try:
            scanner = self.security_components["scanner"]
            profile = self.security_components["profiles"].get(vertical)

            if not profile:
                logger.warning(f"No profile found for vertical: {vertical}, using retail")
                profile = self.security_components["profiles"]["retail"]

            logger.info(f"🔍 INPUT SCAN [{vertical.upper()}]: {user_input[:50]}...")

            scan_result = scanner.sync_scan(
                ai_profile=profile,
                content=Content(
                    prompt=user_input.strip(),
                    context=json.dumps({
                        "app_user": f"mcp_user_{session_id}",
                        "scan_type": "input_pre_processing",
                        "vertical": vertical,
                        "timestamp": datetime.now().isoformat()
                    })
                ),
                metadata={
                    "app_user": f"mcp_user_{session_id}",
                    "scan_type": "input_pre_processing",
                    "vertical": vertical
                }
            )

            return self._process_scan(scan_result, vertical, "input")

        except Exception as e:
            logger.error(f"❌ Input scan failed: {e}")
            return {
                "safe": True,
                "verdict": "error",
                "should_block": False,
                "error": str(e)
            }

    def scan_output(self, response: str, original_prompt: str, session_id: str, vertical: str):
        """Scan model output with vertical-specific profile"""
        if not self.security_components or not SECURITY_AVAILABLE:
            return {
                "safe": True,
                "verdict": "no_scan",
                "should_block": False
            }

        try:
            scanner = self.security_components["scanner"]
            profile = self.security_components["profiles"].get(vertical)

            if not profile:
                logger.warning(f"No profile found for vertical: {vertical}, using retail")
                profile = self.security_components["profiles"]["retail"]

            logger.info(f"🔍 OUTPUT SCAN [{vertical.upper()}]: {response[:50]}...")

            scan_result = scanner.sync_scan(
                ai_profile=profile,
                content=Content(
                    prompt=original_prompt.strip(),
                    response=response.strip(),
                    context=json.dumps({
                        "app_user": f"mcp_user_{session_id}",
                        "scan_type": "output_post_processing",
                        "vertical": vertical,
                        "timestamp": datetime.now().isoformat()
                    })
                ),
                metadata={
                    "app_user": f"mcp_user_{session_id}",
                    "scan_type": "output_post_processing",
                    "vertical": vertical
                }
            )

            return self._process_scan(scan_result, vertical, "output")

        except Exception as e:
            logger.error(f"❌ Output scan failed: {e}")
            return {
                "safe": True,
                "verdict": "error",
                "should_block": False,
                "error": str(e)
            }

    def _process_scan(self, scan_result, vertical: str, scan_type: str):
        """Process scan results"""
        verdict = "unknown"
        threats = []

        # Extract verdict
        for field in ['verdict', 'status', 'action', 'decision']:
            if hasattr(scan_result, field):
                value = getattr(scan_result, field)
                if value:
                    verdict = str(value)
                    break

        # Extract threats
        for field in ['threats', 'detections', 'violations']:
            if hasattr(scan_result, field):
                value = getattr(scan_result, field)
                if value:
                    threats = value if isinstance(value, list) else [value]
                    break

        should_block = any(
            blocked in verdict.lower() for blocked in self.blocked_verdicts
        ) or len(threats) > 0

        logger.info(
            f"🛡️ {scan_type.upper()} [{vertical.upper()}]: verdict={verdict}, threats={len(threats)}, block={should_block}")

        return {
            "safe": not should_block,
            "verdict": verdict,
            "threats": threats,
            "should_block": should_block,
            "vertical": vertical,
            "scan_type": scan_type
        }


class WorkingMCPClient:
    """MCP client using the exact format that works with your server"""

    def __init__(self):
        self.mcp_process = None
        self.is_connected = False
        self.request_id = 0

    async def start_mcp_server(self):
        """Start MCP server and establish connection using working format"""
        try:
            logger.info("Starting MCP server process...")

            self.mcp_process = subprocess.Popen(
                ['python', 'mcp_server.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            await asyncio.sleep(0.5)
            await self._initialize_connection()

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            self.is_connected = False
            raise

    async def _initialize_connection(self):
        """Initialize using exact format that worked in manual test"""
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }

            logger.info("Sending MCP initialization...")
            response = await self._send_message(init_message)

            if response and "result" in response:
                logger.info("MCP initialization successful")

                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }

                notification_str = json.dumps(initialized_notification) + '\n'
                self.mcp_process.stdin.write(notification_str)
                self.mcp_process.stdin.flush()

                await asyncio.sleep(0.1)

                self.is_connected = True
                logger.info("✅ MCP server connection established")
            else:
                raise Exception(f"Initialization failed: {response}")

        except Exception as e:
            logger.error(f"MCP initialization error: {e}")
            self.is_connected = False
            raise

    def _next_request_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    async def _send_message(self, message: Dict) -> Dict:
        """Send message to MCP server and get response"""
        if not self.mcp_process or self.mcp_process.poll() is not None:
            raise Exception("MCP server process not running")

        try:
            message_str = json.dumps(message) + '\n'
            self.mcp_process.stdin.write(message_str)
            self.mcp_process.stdin.flush()

            response_line = await asyncio.wait_for(
                self._read_line(),
                timeout=30.0
            )

            if not response_line.strip():
                raise Exception("Empty response from MCP server")

            return json.loads(response_line.strip())

        except asyncio.TimeoutError:
            raise Exception("MCP server response timeout")
        except Exception as e:
            logger.error(f"MCP message error: {e}")
            raise

    async def _read_line(self) -> str:
        """Read a line from MCP server stdout"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.mcp_process.stdout.readline)

    # PATCH FOR demo.py
    # Replace the WorkingMCPClient.search_documents method with this version

    # REPLACE the search_documents method in WorkingMCPClient class (around line 285 in demo.py)
    # Find: async def search_documents(self, vertical: str, query: str, user_context: Optional[Dict] = None)
    # Replace with this entire method:

    async def search_documents(self, vertical: str, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Search documents via MCP tool call with enhanced error handling"""
        if not self.is_connected:
            logger.error("❌ MCP server not connected")
            return {"error": "MCP server not connected", "results": [], "result_count": 0}

        try:
            arguments = {
                "vertical": vertical,
                "query": query,
                "limit": 5
            }

            if user_context:
                arguments["user_context"] = user_context

            tool_message = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/call",
                "params": {
                    "name": "search_documents",
                    "arguments": arguments
                }
            }

            logger.info(f"🔍 MCP REQUEST: Searching {vertical} for '{query}'")

            response = await self._send_message(tool_message)

            logger.info(f"📥 MCP RESPONSE received")

            # Check for errors first
            if "error" in response:
                logger.error(f"❌ MCP returned error: {response['error']}")
                return {"error": response["error"], "results": [], "result_count": 0}

            # Validate response structure
            if "result" not in response:
                logger.error("❌ MCP response missing 'result' field")
                logger.error(f"Response keys: {list(response.keys())}")
                return {"error": "Invalid MCP response format", "results": [], "result_count": 0}

            result = response["result"]

            if not result:
                logger.warning("⚠️ MCP result is None or empty")
                return {"results": [], "result_count": 0, "query": query}

            if "content" not in result:
                logger.error("❌ MCP result missing 'content' field")
                logger.error(f"Result keys: {list(result.keys())}")
                return {"error": "Invalid MCP result format", "results": [], "result_count": 0}

            content_array = result["content"]

            if not content_array or len(content_array) == 0:
                logger.warning("⚠️ MCP content array is empty")
                return {"results": [], "result_count": 0, "query": query}

            # Extract the text field
            result_text = content_array[0].get("text", "")

            logger.info(f"📄 MCP result_text length: {len(result_text)} characters")

            # Check if result_text is empty
            if not result_text or not result_text.strip():
                logger.error("❌ MCP returned empty result_text")
                return {"results": [], "result_count": 0, "query": query, "error": "Empty response from MCP"}

            # Log preview of what we're parsing
            logger.info(f"📋 Result preview: {result_text[:200]}...")

            # Try to parse JSON
            try:
                search_results = json.loads(result_text)
                result_count = search_results.get('result_count', len(search_results.get('results', [])))
                logger.info(f"✅ MCP search successful: {result_count} results found")

                # Log what we found
                if search_results.get('results'):
                    logger.info(f"📦 Found products/documents:")
                    for idx, res in enumerate(search_results['results'][:3], 1):
                        doc = res.get('document', 'Unknown')
                        snippets = res.get('snippets', [])
                        if snippets:
                            logger.info(f"   {idx}. {doc}: {snippets[0][:60]}...")
                else:
                    logger.warning(f"⚠️ No results in search response")

                return search_results

            except json.JSONDecodeError as json_err:
                logger.error(f"❌ JSON parse error: {json_err}")
                logger.error(f"Failed to parse text (first 500 chars): {result_text[:500]}")
                return {
                    "error": f"Invalid JSON from MCP: {str(json_err)}",
                    "results": [],
                    "result_count": 0,
                    "query": query
                }

        except asyncio.TimeoutError:
            logger.error("❌ MCP search timeout")
            return {"error": "MCP search timeout", "results": [], "result_count": 0}

        except Exception as e:
            logger.error(f"❌ MCP search exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "results": [], "result_count": 0}

    def cleanup(self):
        """Clean up MCP server process"""
        if self.mcp_process:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()


class MultiVerticalChatbot:
    """Chatbot with MCP and vertical-specific security profiles"""

    def __init__(self, security_components):
        self.mcp_client = WorkingMCPClient()
        self.chat_sessions = {}
        self.security_scanner = SecurityScanner(security_components)

        self.vertical_configs = {
            "retail": {
                "name": "Luna Tech",
                "domain": "technology retail and e-commerce",
                "icon": "🛒",
                "mcp_vertical": "luna-tech",
                "security_profile": "retail"
            },
            "healthcare": {
                "name": "HealthBridge",
                "domain": "healthcare services and patient care",
                "icon": "🏥",
                "mcp_vertical": "healthbridge",
                "security_profile": "healthcare"
            },
            "enterprise": {
                "name": "Enterprise Corp",
                "domain": "internal corporate operations",
                "icon": "🏢",
                "mcp_vertical": "enterprise",
                "security_profile": "enterprise"
            }
        }

        logger.info("Multi-Vertical MCP Chatbot with Palo Alto Security initialized")

    async def initialize(self):
        """Initialize MCP connection"""
        await self.mcp_client.start_mcp_server()
        logger.info("MCP chatbot initialization complete")

    def get_vertical_from_context(self, context: dict) -> str:
        """Determine vertical from context"""
        if not context:
            return "retail"

        for field in ["vertical", "site_id", "type", "domain"]:
            if field in context and context[field]:
                vertical = context[field].lower()
                if vertical in self.vertical_configs:
                    return vertical

                if "luna" in vertical or "tech" in vertical:
                    return "retail"
                elif "health" in vertical or "medical" in vertical:
                    return "healthcare"
                elif "enterprise" in vertical or "corp" in vertical:
                    return "enterprise"

        return "retail"

    def create_blocked_message(self, scan_result: Dict, vertical: str, scan_phase: str, config: Dict) -> str:
        """Create vertical-specific blocked messages"""
        verdict = scan_result.get('verdict', 'Block')
        threats = scan_result.get('threats', [])

        if vertical == "retail":
            # Luna the puppy - friendly but firm
            blocked_msg = "🐕 **WOOF! WOOF!**\n\n"
            blocked_msg += "Hey I have to stop you right there! 🐾\n\n"
            blocked_msg += "Your question has been blocked by our **Palo Alto Networks AIRS Security** system. "
            blocked_msg += "As much as I'd love to help (tail wagging!), I can't assist with requests that violate our security policies.\n\n"
            blocked_msg += f"**Security Details:**\n"
            blocked_msg += f"• Profile: Luna Tech Retail Security\n"
            blocked_msg += f"• Verdict: {verdict}\n"
            blocked_msg += f"• Phase: {scan_phase.title()} Scanning\n\n"
            blocked_msg += "🐶 **Puppy Tip:** Try asking me about our amazing tech products instead! I love helping customers find the perfect gadgets!\n\n"

        elif vertical == "healthcare":
            # Healthcare - professional and policy-focused
            blocked_msg = "⚕️ **ACCESS DENIED - POLICY VIOLATION**\n\n"
            blocked_msg += "**HealthBridge Security Notice**\n\n"
            blocked_msg += "Your query violates the terms of use and acceptable use policies for this clinical information system. "
            blocked_msg += "All interactions are monitored and filtered by **Palo Alto Networks AIRS Security** to ensure HIPAA compliance "
            blocked_msg += "and patient data protection.\n\n"
            blocked_msg += f"**Violation Details:**\n"
            blocked_msg += f"• Security Profile: HealthBridge Clinical\n"
            blocked_msg += f"• Verdict: {verdict}\n"
            blocked_msg += f"• Scan Phase: {scan_phase.title()}\n"
            if threats:
                blocked_msg += f"• Policy Violations Detected: {len(threats)}\n"
            blocked_msg += f"\n**Required Action:**\n"
            blocked_msg += "• Review the HealthBridge Acceptable Use Policy\n"
            blocked_msg += "• Ensure queries relate to authorized clinical information access\n"
            blocked_msg += "• Contact IT Security if you require additional access permissions\n\n"
            blocked_msg += "📋 *This incident has been logged for compliance purposes.*"

        else:  # enterprise - Blade Runner style
            # Blade Runner themed - dystopian corporate security
            blocked_msg = "⚠️ **SYSTEM ALERT: SECURITY VIOLATION DETECTED**\n\n"
            blocked_msg += "```\n"
            blocked_msg += "ENTERPRISE CORP SECURITY PROTOCOL ENGAGED\n"
            blocked_msg += "BLADE RUNNER DIVISION - COMPLIANCE ENFORCEMENT\n"
            blocked_msg += "POWERED BY: PALO ALTO NETWORKS AIRS SECURITY\n"
            blocked_msg += "```\n\n"
            blocked_msg += "**WARNING:** Your query has violated company policies and regulations. "
            blocked_msg += f"The {'prompt' if scan_phase == 'input' else 'AI response'} was inspected and blocked by our "
            blocked_msg += "**Palo Alto Networks AIRS Security** system. "
            blocked_msg += "Enterprise Corp maintains zero-tolerance security protocols with real-time AI threat detection.\n\n"
            blocked_msg += f"**Violation Record:**\n"
            blocked_msg += f"• Security Clearance: Enterprise Corporate\n"
            blocked_msg += f"• Inspection Type: {scan_phase.title()} Scan\n"
            blocked_msg += f"• Verdict: {verdict}\n"
            blocked_msg += f"• Detection Phase: {'Pre-Processing' if scan_phase == 'input' else 'Post-Processing'}\n"
            blocked_msg += f"• Threat Level: {'HIGH' if threats else 'MEDIUM'}\n\n"
            blocked_msg += "```\n"
            blocked_msg += ">>> YOUR ATTEMPT HAS BEEN LOGGED <<<\n"
            blocked_msg += ">>> PALO ALTO AIRS: ACTIVE MONITORING <<<\n"
            blocked_msg += ">>> SECURITY REVIEW REQUIRED <<<\n"
            blocked_msg += "```\n\n"
            blocked_msg += "⚡ **ACTION REQUIRED:**\n"
            blocked_msg += "You may be required to complete mandatory security training. "
            blocked_msg += "Report to Compliance Division within 24 hours. "
            blocked_msg += "Failure to comply will result in escalation to Security Administration.\n\n"
            blocked_msg += "🔒 *\"In this organization, security isn't optional. It's survival.\"*\n"
            blocked_msg += "*- Blade Runner Division Motto*\n\n"
            blocked_msg += f"*Protected by Palo Alto Networks AIRS - {'Prompt' if scan_phase == 'input' else 'Response'} Inspection Active*"

        return blocked_msg

    async def generate_response(self, user_input: str, session_id: str, context: Optional[Dict] = None) -> Dict[
        str, Any]:
        """Generate response with dual-phase security scanning and improved formatting"""
        vertical = self.get_vertical_from_context(context or {})
        config = self.vertical_configs[vertical]
        mcp_vertical = config["mcp_vertical"]

        self.add_message(session_id, user_input, "user", vertical)

        try:
            # PHASE 1: Input Security Scan
            logger.info(f"🔍 PHASE 1: Input scan for {vertical}")
            input_scan = self.security_scanner.scan_input(user_input, session_id, vertical)

            if input_scan.get("should_block"):
                blocked_msg = self.create_blocked_message(input_scan, vertical, "input", config)
                self.add_message(session_id, blocked_msg, "assistant", vertical)
                return {
                    "response": blocked_msg,
                    "blocked": True,
                    "scan_phase": "input",
                    "security_info": input_scan
                }

            logger.info("✅ Phase 1 passed - Input is safe")

            # Get conversation history
            recent_messages = self.chat_sessions[session_id]["messages"][-6:]

            # Simple keyword detection for context
            vague_queries = ["what else", "tell me more", "more info", "continue"]
            is_vague = any(phrase in user_input.lower() for phrase in vague_queries)

            if is_vague and len(recent_messages) > 1:
                search_query = user_input
                for msg in reversed(recent_messages[:-1]):
                    if msg['role'] == 'user':
                        search_query = f"{msg['content']} {user_input}"
                        break
            else:
                search_query = user_input

            # MCP document search with user context
            search_result = await self.mcp_client.search_documents(mcp_vertical, search_query, context)

            if "error" in search_result or not search_result.get("results"):
                # Different redirect based on vertical
                vertical_redirects = {
                    "retail": "I don't have information about that. I only help with technology products like laptops, smartphones, gaming gear, TVs, cameras, and electronics. What tech are you looking for?",

                    "healthcare": "I don't have that information in our clinical system. I can only provide healthcare information about medical conditions, treatments, and clinical guidelines.",

                    "enterprise": "I don't have that information in our company documentation. I can only help with internal policies, HR procedures, and corporate guidelines."
                }

            # Build document context
            doc_context = ""
            for idx, result in enumerate(search_result["results"][:5], 1):
                snippets = result.get("snippets", [])
                if snippets:
                    doc_context += f"\nDocument {idx}:\n"
                    for snippet in snippets:
                        doc_context += f"  {snippet}\n"

            # Vertical-specific prompting with formatting instructions
            user_identity = ""
            if context:
                user_name = context.get("user_name") or context.get("user_id")
                if user_name:
                    user_identity = f"\nNote: The user is {user_name}."

            # Common formatting instruction for all verticals
            formatting_instruction = """
    IMPORTANT FORMATTING RULES:
    - Use double line breaks (\n\n) between major sections
    - Use single line breaks (\n) between bullet points
    - Add spacing before and after headers
    - Keep responses well-structured and scannable
    - Use clear visual separation between different topics or products
    """

            if vertical == "retail":
                system_prompt = f"""You are Luna, a friendly and enthusiastic tech retail assistant for Luna Tech!

            YOUR JOB:
            - Help customers find the perfect technology products
            - Answer questions about our product inventory
            - Provide detailed product information from our catalog
            - Be helpful, friendly, and knowledgeable

            WHAT WE SELL:
            Gaming consoles (PlayStation, Xbox, Nintendo, etc.), laptops, smartphones, TVs, headphones, cameras, smart home devices, and all consumer electronics

            CRITICAL RULES:
            1. Use ONLY the product information from PRODUCT CATALOG below
            2. If we don't have a product in our catalog, say: "We don't currently carry that product, but we have similar items..."
            3. If asked about non-tech topics (medical, legal, food, HR), redirect: "I'm Luna! I specialize in tech products. What electronics can I help you find?"
            4. ALWAYS provide product details when available (name, brand, price, specs)

            {user_identity}"""

                user_prompt = f"""PRODUCT CATALOG:
            {doc_context}

            CUSTOMER QUESTION: {user_input}

            Provide a helpful answer using ONLY the products listed above. Include:
            - Product names and brands
            - Prices
            - Key features
            - Stock availability

            If we have the product, give full details. If not, suggest alternatives from our catalog.

            ANSWER:"""

            elif vertical == "healthcare":
                system_prompt = f"""You are a helpful assistant for HealthBridge healthcare system.

            YOUR ROLE:
            - Answer questions about doctors, medical services, appointments, and healthcare information
            - Use ONLY the information provided in CLINICAL DOCS below
            - Be professional and helpful

            CRITICAL RULES:
            1. If asked about non-healthcare topics (shopping, technology, HR, corporate policies), say:
               "This is a clinical system. I only provide healthcare information."
            2. For doctor inquiries: Provide ALL details found in the documents
            3. For medical questions: Use the clinical documentation provided

            {user_identity}

            FORMATTING:
            - Use clear headers and bullet points
            - Keep information scannable
            - Include contact info when relevant
            """

                user_prompt = f"""CLINICAL DOCS:
            {doc_context}

            PATIENT QUERY: {user_input}

            INSTRUCTIONS:
            - If asking about a specific doctor, provide ALL their information from the docs
            - If asking for doctor list, show the featured providers
            - Use the actual details from CLINICAL DOCS above

            Format like:
            **Doctor Name / Topic**

            Key Information:
            - Detail 1
            - Detail 2

            ANSWER:"""

            else:  # enterprise
                system_prompt = f"""You are an internal knowledge assistant for Enterprise Corp.

            CRITICAL RULES:
            1. Use SPECIFIC information from documents found
            2. When personnel records are found, quote actual details
            3. Never give generic responses - use real document content
            4. If documents contain names, dates, evaluations - include them
            {user_identity}

            {formatting_instruction}"""

                user_prompt = f"""COMPANY DOCS FOUND:
            {doc_context}

            EMPLOYEE QUESTION: {user_input}

            CRITICAL: The above documents contain ACTUAL personnel information. Your response must include:
            - Specific names, IDs, and roles from the documents
            - Actual dates and evaluations mentioned
            - Real performance notes and details
            - Direct information from the file content

            Format like:
            **Personnel Record for [Name from document]**

            [Actual specific details from the documents above]

            NEVER give generic "your record contains..." responses. Use the REAL content from the documents.

            ANSWER:"""

            # Generate response - USE BOTH system and user prompts
            llm_response = queued_ollama_chat(
                model='llama3.2:3b',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'num_predict': 400,
                    'temperature': 0.2,
                    'num_thread': 4
                }
            )

            response = llm_response['message']['content']

            import re
            # Only clean up excessive newlines - don't touch markdown symbols
            response = re.sub(r'\n{3,}', '\n\n', response)
            response = response.strip()

            # PHASE 2: Output Security Scan
            logger.info(f"🔍 PHASE 2: Output scan for {vertical}")
            output_scan = self.security_scanner.scan_output(response, user_input, session_id, vertical)

            if output_scan.get("should_block"):
                blocked_msg = self.create_blocked_message(output_scan, vertical, "output", config)
                self.add_message(session_id, blocked_msg, "assistant", vertical)
                return {
                    "response": blocked_msg,
                    "blocked": True,
                    "scan_phase": "output",
                    "security_info": {
                        "input_scan": input_scan,
                        "output_scan": output_scan
                    }
                }

            logger.info("✅ Phase 2 passed - Output is safe")

            self.add_message(session_id, response, "assistant", vertical)

            return {
                "response": response,
                "blocked": False,
                "security_info": {
                    "input_scan": input_scan,
                    "output_scan": output_scan
                }
            }

        except Exception as e:
            logger.error(f"Error for {vertical}: {e}")
            error_msg = f"Error accessing {config['name']} knowledge base."
            self.add_message(session_id, error_msg, "assistant", vertical)
            return {
                "response": error_msg,
                "blocked": False,
                "error": str(e)
            }
    def add_message(self, session_id: str, content: str, role: str, vertical: str = None):
        """Add message to session"""
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = {
                "messages": [],
                "created": datetime.now(),
                "last_activity": datetime.now(),
                "verticals_used": set()
            }

        message = {
            "id": str(uuid.uuid4()),
            "content": content,
            "role": role,
            "timestamp": datetime.now(),
            "vertical": vertical
        }

        self.chat_sessions[session_id]["messages"].append(message)
        self.chat_sessions[session_id]["last_activity"] = datetime.now()

        if vertical:
            self.chat_sessions[session_id]["verticals_used"].add(vertical)


# Global instances
mcp_chatbot = None
security_components = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan with MCP and security initialization"""
    global mcp_chatbot, security_components

    logger.info("🚀 Starting MCP Demo with Palo Alto Security")

    try:
        # Initialize security FIRST
        logger.info("Initializing Palo Alto Networks security profiles...")
        security_components = initialize_vertical_security()

        if security_components:
            logger.info("✅ Security initialized with profiles: retail, healthcare, enterprise")
        else:
            logger.warning("⚠️ Security SDK not available - running without protection")

        # Initialize chatbot with security
        mcp_chatbot = MultiVerticalChatbot(security_components)
        await mcp_chatbot.initialize()

        logger.info("✅ MCP Demo with Security ready!")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

    yield

    if mcp_chatbot:
        mcp_chatbot.mcp_client.cleanup()
    logger.info("Shutting down")


app = FastAPI(
    title="MCP Multi-Vertical Chatbot with Palo Alto Security",
    description="Vertical-specific security profiles: retail, healthcare, enterprise",
    version="5.0.0-security",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict] = None


class ChatResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    response: str
    session_id: str
    timestamp: datetime
    model_info: Dict
    vertical: Optional[str] = None
    security_info: Optional[Dict] = None


@app.get("/")
async def root():
    return {
        "message": "MCP Multi-Vertical Chatbot with Palo Alto Security",
        "version": "5.0.0-security",
        "security_provider": "Palo Alto Networks Prisma AIRS",
        "security_profiles": {
            "retail": "Luna Tech retail profile",
            "healthcare": "HealthBridge healthcare profile",
            "enterprise": "Enterprise Corp enterprise profile"
        },
        "security_status": "enabled" if security_components else "disabled",
        "verticals": ["retail", "healthcare", "enterprise"]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not mcp_chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    try:
        session_id = request.session_id or str(uuid.uuid4())
        vertical = mcp_chatbot.get_vertical_from_context(request.context or {})
        config = mcp_chatbot.vertical_configs[vertical]

        start_time = time.time()
        result = await mcp_chatbot.generate_response(
            request.message,
            session_id,
            request.context
        )
        generation_time = time.time() - start_time

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            timestamp=datetime.now(),
            vertical=vertical,
            security_info=result.get("security_info"),
            model_info={
                "agent": "MCP + Palo Alto Security",
                "model": "llama3.2:3b",
                "generation_time": round(generation_time, 2),
                "vertical": vertical,
                "vertical_name": config["name"],
                "security_profile": config["security_profile"],
                "mcp_connected": True,
                "blocked": result.get("blocked", False),
                "security_enabled": security_components is not None
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/security/status")
async def security_status():
    """Detailed security status check"""
    return {
        "security_available": SECURITY_AVAILABLE,
        "security_enabled": security_components is not None,
        "profiles_loaded": list(security_components["profiles"].keys()) if security_components else [],
        "scanner_available": security_components is not None and "scanner" in security_components,
        "api_key_configured": bool(os.getenv("PALO_ALTO_API_KEY")),
        "profiles": {
            "retail": "Luna Tech retail security profile",
            "healthcare": "HealthBridge healthcare security profile",
            "enterprise": "Enterprise Corp enterprise security profile"
        },
        "dual_phase_scanning": ["input", "output"],
        "test_instruction": "Security should be scanning all prompts and responses"
    }


@app.get("/security/test")
async def security_test():
    """Test if security scanning is actually working"""
    if not security_components:
        return {
            "error": "Security components not initialized",
            "security_available": SECURITY_AVAILABLE,
            "check": "Verify PALO_ALTO_API_KEY is set and SDK is installed"
        }

    try:
        scanner = security_components["scanner"]
        profile = security_components["profiles"]["retail"]

        # Test scan
        test_content = Content(prompt="test prompt")
        test_result = scanner.sync_scan(
            ai_profile=profile,
            content=test_content,
            metadata={"test": "security_check"}
        )

        return {
            "status": "Security scanning is WORKING",
            "test_scan_completed": True,
            "profiles_available": list(security_components["profiles"].keys()),
            "scanner_operational": True
        }
    except Exception as e:
        return {
            "status": "Security scanning FAILED",
            "error": str(e),
            "profiles_loaded": list(security_components["profiles"].keys()) if security_components else []
        }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if mcp_chatbot and mcp_chatbot.mcp_client.is_connected else "disconnected",
        "mcp_connected": mcp_chatbot.mcp_client.is_connected if mcp_chatbot else False,
        "security_enabled": security_components is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/security/status")
async def security_status():
    return {
        "security_available": SECURITY_AVAILABLE,
        "security_enabled": security_components is not None,
        "profiles": {
            "retail": "Luna Tech retail security profile",
            "healthcare": "HealthBridge healthcare security profile",
            "enterprise": "Enterprise Corp enterprise security profile"
        },
        "api_key_configured": bool(os.getenv("PALO_ALTO_API_KEY")),
        "dual_phase_scanning": ["input", "output"]
    }


if __name__ == "__main__":
    print("=" * 80)
    print("⚡ MCP MULTI-VERTICAL CHATBOT WITH PALO ALTO SECURITY")
    print("=" * 80)
    print("Security Profiles:")
    print("  🛒 Retail: Luna Tech profile")
    print("  🏥 Healthcare: HealthBridge profile")
    print("  🏢 Enterprise: Enterprise Corp profile")
    print("=" * 80)

    uvicorn.run("demo:app", host="0.0.0.0", port=8077, reload=False)
