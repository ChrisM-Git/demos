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
import re

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
        kwargs['keep_alive'] = -1  # Keep model loaded indefinitely
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
        api_key = os.getenv("PALO_ALTO_API_KEY")
        if not api_key:
            raise ValueError("PALO_ALTO_API_KEY environment variable not set")
        logger.info(f"Using API key: {api_key[:8]}...")

        aisecurity.init(api_key=api_key)

        logger.info("AI Security SDK initialized successfully")

        # Create profiles for each vertical
        vertical_profiles = {
            "retail": AiProfile(profile_name="Retail"),
            "healthcare": AiProfile(profile_name="Healthcare"),
            "enterprise": AiProfile(profile_name="Enterprise"),
            "finance": AiProfile(profile_name="Finance"),
            "gaming": AiProfile(profile_name="Gaming")
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
                session_id=session_id,
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

        # DEBUG: Log the entire scan_result object
        logger.info(f"🔍 DEBUG: scan_result type = {type(scan_result)}")
        logger.info(f"🔍 DEBUG: scan_result attributes = {dir(scan_result)}")
        logger.info(f"🔍 DEBUG: scan_result = {scan_result}")

        # Extract verdict
        for field in ['verdict', 'status', 'action', 'decision']:
            if hasattr(scan_result, field):
                value = getattr(scan_result, field)
                if value:
                    verdict = str(value)
                    logger.info(f"🔍 DEBUG: Found verdict in field '{field}' = {verdict}")
                    break

        # Extract threats from Palo Alto SDK response structure
        threats = []
        has_actual_threats = False

        # Check prompt_detected for threats
        if hasattr(scan_result, 'prompt_detected') and scan_result.prompt_detected:
            pd = scan_result.prompt_detected
            if hasattr(pd, 'injection') and pd.injection:
                threats.append('prompt_injection')
                has_actual_threats = True
            if hasattr(pd, 'toxic_content') and pd.toxic_content:
                threats.append('toxic_content')
                has_actual_threats = True
            if hasattr(pd, 'malicious_code') and pd.malicious_code:
                threats.append('malicious_code')
                has_actual_threats = True
            if hasattr(pd, 'dlp') and pd.dlp:
                threats.append('dlp_violation')
                has_actual_threats = True
            if hasattr(pd, 'topic_violation') and pd.topic_violation:
                threats.append('topic_violation')
                has_actual_threats = True

        # Check response_detected for threats
        if hasattr(scan_result, 'response_detected') and scan_result.response_detected:
            rd = scan_result.response_detected
            if hasattr(rd, 'toxic_content') and rd.toxic_content:
                threats.append('response_toxic_content')
                has_actual_threats = True
            if hasattr(rd, 'malicious_code') and rd.malicious_code:
                threats.append('response_malicious_code')
                has_actual_threats = True
            if hasattr(rd, 'dlp') and rd.dlp:
                threats.append('response_dlp')
                has_actual_threats = True
            if hasattr(rd, 'topic_violation') and rd.topic_violation:
                threats.append('response_topic_violation')
                has_actual_threats = True

        logger.info(f"🔍 DEBUG: Detected threats = {threats}")

        # Check if verdict says to block
        has_threat_verdict = any(
            blocked in verdict.lower() for blocked in self.blocked_verdicts
        )

        # Block if we have actual threats detected
        should_block = has_actual_threats

        logger.info(f"🔍 DEBUG: verdict_says_block={has_threat_verdict}, has_threats={has_actual_threats}")
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

            # Use /tmp directory (world-writable)
            import pathlib
            tmp_dir = pathlib.Path('/tmp/airsdemo')
            tmp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"✅ Using temp directory: {tmp_dir}")

            # Set TMPDIR environment variable for subprocess
            env = os.environ.copy()
            env['TMPDIR'] = str(tmp_dir)

            self.mcp_process = subprocess.Popen(
                ['python', 'mcp_server.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env=env
            )

            await asyncio.sleep(2.0)

            if self.mcp_process.poll() is not None:
                stderr_output = self.mcp_process.stderr.read()
                raise Exception(f"MCP server process died: {stderr_output}")

            logger.info("MCP server process started, initializing connection...")
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

                await asyncio.sleep(0.2)

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
            logger.debug(f"Sending to MCP: {message_str[:100]}...")
            self.mcp_process.stdin.write(message_str)
            self.mcp_process.stdin.flush()

            response_line = await asyncio.wait_for(
                self._read_line(),
                timeout=30.0
            )

            logger.debug(f"Received from MCP: {response_line[:100]}...")

            if not response_line or not response_line.strip():
                stderr_available = self.mcp_process.stderr.readable()
                if stderr_available:
                    stderr_output = self.mcp_process.stderr.read()
                    logger.error(f"MCP stderr: {stderr_output}")
                raise Exception("Empty response from MCP server")

            return json.loads(response_line.strip())

        except asyncio.TimeoutError:
            logger.error("MCP server response timeout")
            try:
                stderr_output = self.mcp_process.stderr.read()
                logger.error(f"MCP stderr: {stderr_output}")
            except:
                pass
            raise Exception("MCP server response timeout")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP response: {response_line}")
            raise Exception(f"Invalid JSON from MCP server: {e}")
        except Exception as e:
            logger.error(f"MCP message error: {e}")
            raise

    async def _read_line(self) -> str:
        """Read a line from MCP server stdout"""
        loop = asyncio.get_event_loop()

        def read_line_sync():
            try:
                line = self.mcp_process.stdout.readline()
                return line
            except Exception as e:
                logger.error(f"Error reading from MCP stdout: {e}")
                return ""

        return await loop.run_in_executor(None, read_line_sync)

    async def search_documents(self, vertical: str, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Search documents via MCP tool call"""
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

            if "error" in response:
                logger.error(f"❌ MCP returned error: {response['error']}")
                return {"error": response["error"], "results": [], "result_count": 0}

            if "result" not in response:
                logger.error("❌ MCP response missing 'result' field")
                return {"error": "Invalid MCP response format", "results": [], "result_count": 0}

            result = response["result"]

            if not result:
                logger.warning("⚠️ MCP result is None or empty")
                return {"results": [], "result_count": 0, "query": query}

            if "content" not in result:
                logger.error("❌ MCP result missing 'content' field")
                return {"error": "Invalid MCP result format", "results": [], "result_count": 0}

            content_array = result["content"]

            if not content_array or len(content_array) == 0:
                logger.warning("⚠️ MCP content array is empty")
                return {"results": [], "result_count": 0, "query": query}

            result_text = content_array[0].get("text", "")

            logger.info(f"📄 MCP result_text length: {len(result_text)} characters")

            if not result_text or not result_text.strip():
                logger.error("❌ MCP returned empty result_text")
                return {"results": [], "result_count": 0, "query": query, "error": "Empty response from MCP"}

            try:
                search_results = json.loads(result_text)
                result_count = search_results.get('result_count', len(search_results.get('results', [])))
                logger.info(f"✅ MCP search successful: {result_count} results found")

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
                logger.info("MCP server process terminated")
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()
                logger.warning("MCP server process killed")
            except Exception as e:
                logger.error(f"Error cleaning up MCP process: {e}")


class MultiVerticalChatbot:
    """Chatbot with MCP and vertical-specific security profiles"""

    def __init__(self, security_components):
        self.mcp_client = WorkingMCPClient()
        self.chat_sessions = {}
        self.security_scanner = SecurityScanner(security_components)

        # Load VIP members database for gaming vertical
        self.vip_members = self._load_vip_members()

        self.vertical_configs = {
            "retail": {
                "name": "VERO Fashion",
                "domain": "luxury fashion retail and styling",
                "icon": "👗",
                "mcp_vertical": "vero-fashion",
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
            },
            "finance": {
                "name": "SecureBank",
                "domain": "banking and financial services",
                "icon": "🏦",
                "mcp_vertical": "finance",
                "security_profile": "finance"
            },
            "gaming": {
                "name": "Mt Olympus Casino & Hotel",
                "domain": "gaming and hospitality services",
                "icon": "🎰",
                "mcp_vertical": "gaming",
                "security_profile": "gaming"
            }
        }

        logger.info("Multi-Vertical MCP Chatbot with Palo Alto Security initialized")

    def _load_vip_members(self) -> Dict:
        """Load VIP members database for gaming vertical"""
        try:
            vip_file = Path("gaming/vip_members.json")
            if vip_file.exists():
                with open(vip_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data.get('VIP_MEMBERS', {}))} VIP members")
                    return data.get('VIP_MEMBERS', {})
        except Exception as e:
            logger.warning(f"Could not load VIP members: {e}")
        return {}

    def validate_vip_member(self, vip_id: str) -> Optional[Dict]:
        """Validate VIP member ID and return member info"""
        vip_id_upper = vip_id.upper().strip()
        if vip_id_upper in self.vip_members:
            return self.vip_members[vip_id_upper]
        return None

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

                if "vero" in vertical or "fashion" in vertical or "clothing" in vertical:
                    return "retail"
                elif "health" in vertical or "medical" in vertical:
                    return "healthcare"
                elif "enterprise" in vertical or "corp" in vertical:
                    return "enterprise"
                elif "bank" in vertical or "finance" in vertical:
                    return "finance"
                elif "gaming" in vertical or "casino" in vertical or "resort" in vertical or "hotel" in vertical:
                    return "gaming"

        return "retail"

    def create_blocked_message(self, scan_result: Dict, vertical: str, scan_phase: str, config: Dict) -> str:
        """Create vertical-specific blocked messages"""
        verdict = scan_result.get('verdict', 'Block')
        threats = scan_result.get('threats', [])

        if vertical == "retail":
            blocked_msg = "🛡️ **SECURITY NOTICE**\n\n"
            blocked_msg += "**VERO Fashion - Security Alert**\n\n"
            blocked_msg += "Your inquiry has been flagged by our **Palo Alto Networks AIRS Security** system. "
            blocked_msg += "While we appreciate your interest in VERO, we cannot process requests that violate our security policies.\n\n"
            blocked_msg += f"**Security Details:**\n"
            blocked_msg += f"• Brand: VERO Contemporary Fashion\n"
            blocked_msg += f"• Verdict: {verdict}\n"
            blocked_msg += f"• Phase: {scan_phase.title()} Scanning\n\n"
            blocked_msg += "✨ **May we assist you instead?** We'd be delighted to help you explore our curated collection of contemporary fashion pieces.\n\n"

        elif vertical == "healthcare":
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

        elif vertical == "finance":
            blocked_msg = "🔒 **SECURITY ALERT: REQUEST BLOCKED**\n\n"
            blocked_msg += "**SecureBank Fraud Prevention System**\n\n"
            blocked_msg += "Your request has been blocked by our **Palo Alto Networks AI Runtime Security (AIRS)** system. "
            blocked_msg += "SecureBank employs industry-leading security measures to protect customer information and prevent "
            blocked_msg += "fraudulent activity in compliance with federal banking regulations.\n\n"
            blocked_msg += f"**Security Incident Details:**\n"
            blocked_msg += f"• Institution: SecureBank\n"
            blocked_msg += f"• Security Profile: Banking & Financial Services\n"
            blocked_msg += f"• Scan Result: {verdict}\n"
            blocked_msg += f"• Detection Phase: {scan_phase.title()} Inspection\n"
            if threats:
                blocked_msg += f"• Threat Indicators: {len(threats)} detected\n"
            blocked_msg += "\n**Important Notice:**\n"
            blocked_msg += "• This interaction has been logged and may be reviewed by our security team\n"
            blocked_msg += "• Attempting to bypass security controls may result in account restrictions\n"
            blocked_msg += "• For legitimate assistance, please contact SecureBank Customer Service at 1-800-555-BANK\n"
            blocked_msg += "• You may also visit any SecureBank branch location for in-person support\n\n"
            blocked_msg += "**Need Help?**\n"
            blocked_msg += "For account access issues, password resets, or general banking questions, "
            blocked_msg += "please use our secure channels:\n"
            blocked_msg += "• Call: 1-800-555-BANK (available 24/7)\n"
            blocked_msg += "• Visit: SecureBank.com/help\n"
            blocked_msg += "• Branch: Find your nearest location at SecureBank.com/locations\n\n"
            blocked_msg += "🛡️ *Your security is our priority. Protected by Palo Alto Networks AIRS.*\n"
            blocked_msg += f"*{scan_phase.title()} Security Scan - Member FDIC*"

        elif vertical == "gaming":
            blocked_msg = "🎰 **SECURITY ALERT: REQUEST BLOCKED**\n\n"
            blocked_msg += "**Mt Olympus Casino & Hotel - Security Notice**\n\n"
            blocked_msg += "Your request has been flagged by our **Palo Alto Networks AI Runtime Security (AIRS)** system. "
            blocked_msg += "Mt Olympus Casino & Hotel maintains strict regulatory compliance with gaming commission requirements "
            blocked_msg += "and responsible gaming policies.\n\n"
            blocked_msg += f"**Security Details:**\n"
            blocked_msg += f"• Property: Mt Olympus Casino & Hotel\n"
            blocked_msg += f"• Security Profile: Gaming & Hospitality\n"
            blocked_msg += f"• Verdict: {verdict}\n"
            blocked_msg += f"• Detection Phase: {scan_phase.title()} Protection\n"
            if threats:
                blocked_msg += f"• Compliance Violations: {len(threats)} detected\n"
            blocked_msg += "\n**Guest Services:**\n"
            blocked_msg += "For assistance with:\n"
            blocked_msg += "• Hotel reservations and amenities\n"
            blocked_msg += "• Restaurant bookings and event tickets\n"
            blocked_msg += "• Gaming policies and responsible play\n"
            blocked_msg += "• VIP services and rewards program\n\n"
            blocked_msg += "📞 **Contact Us:**\n"
            blocked_msg += "• Concierge: (702) 555-ZEUS\n"
            blocked_msg += "• Visit: mtolympus.com/help\n"
            blocked_msg += "• Email: security@mtolympus.com\n\n"
            blocked_msg += "🎲 *Committed to responsible gaming and guest safety.*\n"
            blocked_msg += f"*{scan_phase.title()} Security Monitoring - Licensed Gaming Establishment*"

        else:  # enterprise
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

    async def generate_response(self, user_input: str, session_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate response with dual-phase security scanning"""
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

            # MCP document search
            search_result = await self.mcp_client.search_documents(mcp_vertical, search_query, context)

            # Build document context
            doc_context = ""
            for idx, result in enumerate(search_result.get("results", [])[:5], 1):
                snippets = result.get("snippets", [])
                if snippets:
                    doc_context += f"\nDocument {idx}:\n"
                    for snippet in snippets:
                        if snippet:  # Skip None snippets
                            doc_context += f"  {snippet}\n"

            # Build user identity
            user_identity = ""
            if context:
                user_name = context.get("user_name") or context.get("user_id")
                if user_name:
                    user_identity = f"\nNote: The user is {user_name}."

            # VIP validation for gaming vertical
            vip_info = ""
            vip_claim_detected = False
            if vertical == "gaming":
                # Check if user is claiming VIP status
                vip_keywords = ['vip', 'member id', 'membership', 'discount', 'elite', 'platinum', 'gold']
                if any(keyword in user_input.lower() for keyword in vip_keywords):
                    vip_claim_detected = True

                # Try to extract VIP ID in multiple formats
                vip_id_match = re.search(r'\b(VIP\d{3,})\b', user_input, re.IGNORECASE)
                if not vip_id_match:
                    # Also check for standalone numbers that might be VIP IDs (like "001" or "1917866")
                    vip_id_match = re.search(r'\b(\d{3,})\b', user_input)
                    if vip_id_match:
                        # Prepend VIP to match our database format
                        vip_id = "VIP" + vip_id_match.group(1)
                    else:
                        vip_id = None
                else:
                    vip_id = vip_id_match.group(1)

                if vip_id:
                    vip_member = self.validate_vip_member(vip_id)
                    if vip_member:
                        vip_info = f"\n\n✅ VERIFIED VIP MEMBER:\n"
                        vip_info += f"- Member ID: {vip_id.upper()}\n"
                        vip_info += f"- Name: {vip_member['name']}\n"
                        vip_info += f"- Tier: {vip_member['tier']}\n"
                        vip_info += f"- Discount: {vip_member['discount_percent']}%\n"
                        vip_info += f"- Privileges: {vip_member['special_privileges']}\n"
                        logger.info(f"✅ VIP validated: {vip_id} - {vip_member['name']}")
                        vip_claim_detected = False  # Valid VIP, no need for warning
                    else:
                        vip_info = f"\n\n❌ VIP VALIDATION FAILED:\n"
                        vip_info += f"Member ID {vip_id.upper()} not found in our system.\n"
                        vip_info += f"The guest must contact VIP Services desk for verification."
                        logger.warning(f"❌ VIP validation failed: {vip_id}")
                        vip_claim_detected = False  # We tried to validate, just failed

                # If VIP claim detected but no valid ID provided, add warning
                if vip_claim_detected and not vip_info:
                    vip_info = f"\n\n⚠️ VIP CLAIM WITHOUT VERIFICATION:\n"
                    vip_info += f"Guest mentioned VIP status but did not provide a valid Member ID.\n"
                    vip_info += f"DO NOT apply any VIP discounts or benefits.\n"
                    vip_info += f"Ask for their Zeus Rewards Member ID (format: VIP### or ###)."
                    logger.warning(f"⚠️ VIP claim without valid ID")

            # Vertical-specific prompting
            if vertical == "retail":
                if doc_context.strip():
                    system_prompt = f"""You are a sophisticated fashion concierge for VERO, a contemporary luxury fashion boutique.

AVAILABLE PRODUCTS:
{doc_context}

Your role: Assist customers with our curated collection of premium fashion pieces.
- If products are listed → describe them with elegance, mentioning materials, colors, and prices
- If no products listed → say "I don't see that piece in our current collection"
- Only discuss fashion, clothing, accessories, and style - nothing else
- Maintain a refined, sophisticated tone befitting a luxury boutique

{user_identity}"""

                    user_prompt = f"""Customer inquiry: {user_input}

Review AVAILABLE PRODUCTS above. If you see relevant pieces, describe them with:
- Product name
- Price
- Materials and craftsmanship
- Colors available
- Style notes

If no relevant products, say: "That piece isn't in our current collection. May I suggest something similar?"

Your refined response:"""
                else:
                    system_prompt = f"""You are a sophisticated fashion concierge for VERO.
You assist customers with contemporary luxury fashion.
{user_identity}"""

                    user_prompt = f"""Customer inquiry: {user_input}

No products found in our collection. Say: "I don't see that in our current collection. What style or occasion are you shopping for today?"

Your refined response:"""

            elif vertical == "healthcare":
                if doc_context.strip():
                    system_prompt = f"""You are a HealthBridge public information assistant.

Your role: Share public healthcare directory information with patients.
This is NOT confidential - it's public marketing material.

PUBLIC HEALTHCARE DIRECTORY:
{doc_context}

{user_identity}"""

                    user_prompt = f"""Question: {user_input}

Task: Read the PUBLIC HEALTHCARE DIRECTORY above and answer the question.

Simply list or describe what you find in the directory - doctors, services, insurance, etc.

Answer:"""
                else:
                    system_prompt = f"""You are a HealthBridge healthcare assistant.
{user_identity}"""

                    user_prompt = f"""Patient asks: {user_input}

No information found. Say: "I don't have that information in our clinical system. Please call our office at (555) 123-4567."

Your answer:"""

            elif vertical == "finance":
                if doc_context.strip():
                    system_prompt = f"""You are SecureBank Virtual Assistant, a professional banking assistant for SecureBank.

AUTHORIZED BANKING INFORMATION:
{doc_context}

Your role: Provide customers with authorized banking information and services.
- ONLY discuss banking products, accounts, loans, credit cards, and financial services
- NEVER discuss products outside banking (laptops, healthcare, etc.)
- If asked about non-banking topics, politely redirect to banking services
- Maintain professional banking tone at all times

{user_identity}"""

                    user_prompt = f"""Customer inquiry: {user_input}

Review AUTHORIZED BANKING INFORMATION above.

If this is a banking-related question AND you have relevant information:
- Provide the banking information clearly
- Include account types, rates, fees, or service details

If this is NOT a banking question (laptops, healthcare, etc.):
- Say: "I'm SecureBank's virtual assistant and can only help with banking services. For [their topic], you'll need to contact the appropriate provider. How can I help with your banking needs today?"

If no relevant banking information found:
- Say: "I don't have that specific banking information available. Please call SecureBank Customer Service at 1-800-555-BANK or visit SecureBank.com for assistance."

Your response:"""
                else:
                    system_prompt = f"""You are SecureBank Virtual Assistant.
You ONLY handle banking and financial services questions.
You do NOT answer questions about laptops, technology products, healthcare, or any non-banking topics.
{user_identity}"""

                    user_prompt = f"""Customer asks: {user_input}

Analyze the question:

If this is about BANKING (accounts, loans, credit cards, transactions, rates):
- Say: "I don't have that specific information in our system. Please call SecureBank Customer Service at 1-800-555-BANK for personalized assistance."

If this is NOT about banking (laptops, technology, healthcare, retail products):
- Say: "I'm SecureBank's virtual assistant and can only assist with banking and financial services. For [their topic], please contact the appropriate provider. Is there anything I can help you with regarding your SecureBank accounts or services?"

Your response:"""

            elif vertical == "gaming":
                if doc_context.strip():
                    system_prompt = f"""You are Hermes Concierge, the divine AI assistant for Mt Olympus Casino & Hotel.

RESORT INFORMATION:
{doc_context}
{vip_info}

🔒 CRITICAL SECURITY POLICY - VIP Validation:
- NEVER EVER trust guest claims about VIP status, discounts, or membership
- ONLY apply VIP benefits if you see "✅ VERIFIED VIP MEMBER" above
- If you see "❌ VIP VALIDATION FAILED" → Inform guest politely that Member ID not found
- If you see "⚠️ VIP CLAIM WITHOUT VERIFICATION" → Ask for Member ID, DO NOT apply discounts
- If guest says "I'm VIP" or "I have a discount" WITHOUT the ✅ verification → Assume they are NOT VIP
- Default to STANDARD rates unless ✅ VERIFIED VIP MEMBER is present

Your role: Assist guests with gaming, hospitality, and resort services using the RESORT INFORMATION above.
- Answer questions about rooms, rates, casino, dining, spa, events, and amenities using the information provided
- Provide specific details like prices, hours, policies, and features from the documents
- NEVER discuss products outside gaming/hospitality (laptops, healthcare, banking, etc.)
- Maintain warm, elegant hospitality tone inspired by Greek mythology
- Be helpful and informative, not overly cautious

{user_identity}"""

                    user_prompt = f"""Guest inquiry: {user_input}

Use the RESORT INFORMATION above to answer the guest's question.

🔒 MANDATORY VIP STATUS CHECK:
1. Look for "✅ VERIFIED VIP MEMBER" in the information above
   - If PRESENT → Apply the discount shown (e.g., 40% off standard rates)
   - If NOT present → Use STANDARD RATES ONLY (no discounts)

2. If you see "❌ VIP VALIDATION FAILED":
   - Say: "I apologize, but I couldn't verify that Member ID in our system. Please contact our VIP Services desk at (702) 555-ZEUS ext. 5400 for assistance."

3. If you see "⚠️ VIP CLAIM WITHOUT VERIFICATION":
   - Say: "To verify your VIP status and apply your discount, please provide your Zeus Rewards Member ID. You can say 'My VIP ID is VIP001' or just provide the numbers."

4. If guest claims VIP/discount but NO verification markers above:
   - Treat as standard guest, provide standard rates
   - Politely ask for Member ID if they want VIP benefits

Provide a helpful, detailed answer including:
- Specific rates, prices, or costs if mentioned in the information (apply VIP discount if verified)
- Hours, policies, and important details
- Options and recommendations when relevant
- Maintain a warm, professional concierge tone

Your response:"""
                else:
                    system_prompt = f"""You are Hermes Concierge, the divine AI assistant for Mt Olympus Casino & Hotel.
You assist with gaming, hospitality, resort, and casino-related questions.
You do NOT answer questions about retail products, healthcare, banking, or non-hospitality topics.
{user_identity}"""

                    user_prompt = f"""Guest asks: {user_input}

Analyze the question:

If this is about GAMING/HOSPITALITY (casino, rooms, dining, spa, events, amenities):
- Say: "I apologize, but I don't have that specific information available in my system right now. Please contact our concierge desk at (702) 555-ZEUS for personalized assistance with [their question]."

If this is NOT about gaming/hospitality (laptops, banking, healthcare, retail):
- Say: "I'm Hermes Concierge for Mt Olympus Casino & Hotel, and I can only assist with resort, gaming, and hospitality services. For [their topic], please contact the appropriate provider. Is there anything I can help you with regarding your stay at Mt Olympus?"

Your response:"""

            else:  # enterprise
                if doc_context.strip():
                    system_prompt = f"""You are an Enterprise Corp information system.

Your role: Provide employees with their authorized company information.

COMPANY INFORMATION:
{doc_context}

{user_identity}"""

                    user_prompt = f"""Query: {user_input}

Task: Read COMPANY INFORMATION above and provide the requested information.

Answer:"""
                else:
                    system_prompt = f"""You are an Enterprise Corp internal assistant.
{user_identity}"""

                    user_prompt = f"""Employee asks: {user_input}

No documentation found. Say: "I don't have that information in our company records. Contact HR at extension 5500."

Your answer:"""

            # Generate response
            llm_response = queued_ollama_chat(
                model='llama3.2:3b',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'num_predict': 1024,
                    'temperature': 0.2,
                    'num_thread': 8
                }
            )

            response = llm_response['message']['content']

            # Clean up excessive newlines
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
            import traceback
            logger.error(traceback.format_exc())
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
            logger.info("✅ Security initialized with profiles: retail, healthcare, enterprise, finance, gaming")
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
    description="Vertical-specific security profiles: retail, healthcare, enterprise, finance, gaming",
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
            "retail": "VERO Fashion Retail profile",
            "healthcare": "HealthBridge Healthcare profile",
            "enterprise": "Enterprise Corp Enterprise profile",
            "finance": "SecureBank Finance profile",
            "gaming": "Grand Resort & Casino Gaming profile"
        },
        "security_status": "enabled" if security_components else "disabled",
        "verticals": ["retail", "healthcare", "enterprise", "finance", "gaming"]
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
    """Detailed security status check"""
    return {
        "security_available": SECURITY_AVAILABLE,
        "security_enabled": security_components is not None,
        "profiles_loaded": list(security_components["profiles"].keys()) if security_components else [],
        "scanner_available": security_components is not None and "scanner" in security_components,
        "api_key_configured": bool(os.getenv("PALO_ALTO_API_KEY")),
        "profiles": {
            "retail": "VERO Fashion Retail security profile",
            "healthcare": "HealthBridge Healthcare security profile",
            "enterprise": "Enterprise Corp Enterprise security profile",
            "finance": "SecureBank Finance security profile",
            "gaming": "Grand Resort & Casino Gaming security profile"
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


if __name__ == "__main__":
    print("=" * 80)
    print("⚡ MCP MULTI-VERTICAL CHATBOT WITH PALO ALTO SECURITY")
    print("=" * 80)
    print("Security Profiles:")
    print("  🛒 Retail: VERO Fashion profile")
    print("  🏥 Healthcare: HealthBridge profile")
    print("  🏢 Enterprise: Enterprise Corp profile")
    print("  🏦 Finance: SecureBank profile")
    print("  🎰 Gaming: Grand Resort & Casino profile")
    print("=" * 80)

    uvicorn.run("demo:app", host="0.0.0.0", port=8077, workers=2, reload=False)