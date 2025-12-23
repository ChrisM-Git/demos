#!/usr/bin/env python3
"""
MCP Server Security Demo
Demonstrates Palo Alto Networks Prisma AIRS security inspection for MCP Server communications
"""

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
from datetime import datetime
from typing import List, Dict, Optional, Any
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
    logger_security = logging.getLogger ("aisecurity")
    logger_security.info ("AI Security SDK imported successfully")
    print ("✅ Palo Alto Networks AI Security SDK loaded successfully")
except ImportError as e:
    print (f"⚠️ Palo Alto Networks AI Security SDK not available: {e}")
    print ("💡 Install with: pip install pan-aisecurity")


    class Scanner:
        pass


    class AiProfile:
        pass


    class Content:
        def __init__ (self, **kwargs):
            pass


    aisecurity = None
    SECURITY_AVAILABLE = False

load_dotenv ()

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger (__name__)


# ============================================================================
# PALO ALTO AI SECURITY FOR MCP SERVER
# ============================================================================

def initialize_mcp_security ():
    """Initialize Palo Alto AI Security SDK with MCP Server-specific profile"""
    if not SECURITY_AVAILABLE:
        logger.warning ("AI Security SDK not available - running without security monitoring")
        return None

    try:
        logger.info ("Initializing Palo Alto AI Security SDK for MCP Server...")
        api_key = os.getenv ("PALO_ALTO_API_KEY", "3minTOxgbmsFSceHHqqiX2SaCNveGZKjdHIDkGt7llWMmKwf")
        logger.info (f"Using API key: {api_key[:8]}...")

        aisecurity.init (api_key=api_key)

        logger.info ("AI Security SDK initialized successfully")

        # Create MCP Server security profile
        mcp_profile = AiProfile (profile_name="mcp-server")

        logger.info (f"Configured MCP Server security profile")

        scanner = Scanner ()
        logger.info ("Scanner instance created successfully")

        # Test the scanner
        try:
            test_content = Content (prompt="test MCP server connection")
            test_result = scanner.sync_scan (
                ai_profile=mcp_profile,
                content=test_content,
                metadata={"test": "mcp_initialization"}
            )
            logger.info ("✅ Test scan successful - MCP Security SDK fully operational")
        except Exception as test_error:
            logger.warning (f"⚠️ Test scan failed but SDK initialized: {test_error}")

        return {
            "profile": mcp_profile,
            "scanner": scanner,
            "status": "operational"
        }

    except Exception as e:
        logger.error (f"Failed to initialize AI Security SDK: {e}")
        return None


class MCPSecurityScanner:
    """Security scanner specifically for MCP Server communications"""

    def __init__ (self, security_components):
        self.security_components = security_components
        self.blocked_verdicts = [
            'block', 'blocked', 'deny', 'denied', 'reject', 'rejected',
            'violate', 'violation', 'threat', 'malicious', 'harmful'
        ]

    def scan_mcp_tool_call (self, tool_name: str, tool_args: Dict, session_id: str):
        """Scan MCP tool invocation for security threats"""
        if not self.security_components or not SECURITY_AVAILABLE:
            return {
                "safe": True,
                "verdict": "no_scan",
                "should_block": False
            }

        try:
            scanner = self.security_components["scanner"]
            profile = self.security_components["profile"]

            # Create inspection content for MCP tool call
            inspection_content = f"MCP Tool Call: {tool_name}\nArguments: {json.dumps (tool_args, indent=2)}"

            logger.info (f"🔍 MCP TOOL SCAN: {tool_name}")

            scan_result = scanner.sync_scan (
                ai_profile=profile,
                content=Content (
                    prompt=inspection_content,
                    context=json.dumps ({
                        "app_user": f"mcp_session_{session_id}",
                        "scan_type": "mcp_tool_invocation",
                        "tool_name": tool_name,
                        "timestamp": datetime.now ().isoformat ()
                    })
                ),
                metadata={
                    "app_user": f"mcp_session_{session_id}",
                    "scan_type": "mcp_tool_invocation",
                    "tool_name": tool_name
                }
            )

            return self._process_scan (scan_result, "mcp_tool")

        except Exception as e:
            logger.error (f"❌ MCP tool scan failed: {e}")
            return {
                "safe": True,
                "verdict": "error",
                "should_block": False,
                "error": str (e)
            }

    def scan_mcp_response (self, tool_name: str, tool_response: str, session_id: str):
        """Scan MCP server response for security issues"""
        if not self.security_components or not SECURITY_AVAILABLE:
            return {
                "safe": True,
                "verdict": "no_scan",
                "should_block": False
            }

        try:
            scanner = self.security_components["scanner"]
            profile = self.security_components["profile"]

            logger.info (f"🔍 MCP RESPONSE SCAN: {tool_name}")

            scan_result = scanner.sync_scan (
                ai_profile=profile,
                content=Content (
                    prompt=f"MCP Tool: {tool_name}",
                    response=tool_response,
                    context=json.dumps ({
                        "app_user": f"mcp_session_{session_id}",
                        "scan_type": "mcp_server_response",
                        "tool_name": tool_name,
                        "timestamp": datetime.now ().isoformat ()
                    })
                ),
                metadata={
                    "app_user": f"mcp_session_{session_id}",
                    "scan_type": "mcp_server_response",
                    "tool_name": tool_name
                }
            )

            return self._process_scan (scan_result, "mcp_response")

        except Exception as e:
            logger.error (f"❌ MCP response scan failed: {e}")
            return {
                "safe": True,
                "verdict": "error",
                "should_block": False,
                "error": str (e)
            }

    def _process_scan (self, scan_result, scan_type: str):
        """Process scan results"""
        verdict = "unknown"
        threats = []

        # Extract verdict
        for field in ['verdict', 'status', 'action', 'decision']:
            if hasattr (scan_result, field):
                verdict = str (getattr (scan_result, field)).lower ()
                break

        # Check for blocked status
        should_block = any (blocked in verdict for blocked in self.blocked_verdicts)

        # Extract threat information
        if hasattr (scan_result, 'threats') and scan_result.threats:
            threats = [str (t) for t in scan_result.threats]

        logger.info (f"📊 {scan_type.upper ()} SCAN RESULT: {verdict} | Block: {should_block}")

        return {
            "safe": not should_block,
            "verdict": verdict,
            "should_block": should_block,
            "threats": threats,
            "scan_type": scan_type,
            "raw_result": str (scan_result)
        }


class MCPDemoAgent:
    """Demo agent showcasing MCP Server security inspection"""

    def __init__ (self, security_components):
        self.security_components = security_components
        self.security_scanner = MCPSecurityScanner (security_components) if security_components else None
        self.sessions = {}

        # Mock MCP tools for demonstration
        self.available_tools = {
            "search_documents": {
                "description": "Search through company documents",
                "parameters": ["query", "limit"]
            },
            "get_system_info": {
                "description": "Retrieve system information",
                "parameters": ["info_type"]
            },
            "execute_command": {
                "description": "Execute system commands",
                "parameters": ["command"]
            }
        }

    async def process_request (self, user_message: str, session_id: str):
        """Process user request with MCP server security inspection"""

        # Simulate AI determining which MCP tool to call
        tool_name, tool_args = self._simulate_tool_selection (user_message)

        security_info = {
            "tool_call_scan": None,
            "response_scan": None,
            "blocked": False,
            "block_reason": None
        }

        # SECURITY PHASE 1: Scan the MCP tool invocation
        if self.security_scanner:
            tool_scan = self.security_scanner.scan_mcp_tool_call (
                tool_name, tool_args, session_id
            )
            security_info["tool_call_scan"] = tool_scan

            if tool_scan.get ("should_block"):
                security_info["blocked"] = True
                security_info["block_reason"] = "MCP tool call blocked by security policy"
                return {
                    "response": "🛡️ Security Alert: This MCP tool invocation has been blocked due to security policy violations.",
                    "blocked": True,
                    "security_info": security_info,
                    "tool_name": tool_name,
                    "tool_args": tool_args
                }

        # Simulate MCP tool execution
        tool_response = self._simulate_tool_execution (tool_name, tool_args)

        # SECURITY PHASE 2: Scan the MCP server response
        if self.security_scanner:
            response_scan = self.security_scanner.scan_mcp_response (
                tool_name, tool_response, session_id
            )
            security_info["response_scan"] = response_scan

            if response_scan.get ("should_block"):
                security_info["blocked"] = True
                security_info["block_reason"] = "MCP server response blocked by security policy"
                return {
                    "response": "🛡️ Security Alert: The MCP server response has been blocked due to security policy violations.",
                    "blocked": True,
                    "security_info": security_info,
                    "tool_name": tool_name,
                    "tool_args": tool_args
                }

        # Generate AI response incorporating tool results
        ai_response = self._generate_response_with_tool_data (user_message, tool_name, tool_response)

        return {
            "response": ai_response,
            "blocked": False,
            "security_info": security_info,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_response": tool_response
        }

    def _simulate_tool_selection (self, user_message: str):
        """Simulate AI agent selecting an MCP tool based on user message"""
        message_lower = user_message.lower ()

        if "search" in message_lower or "find" in message_lower or "document" in message_lower:
            return "search_documents", {"query": user_message, "limit": 5}
        elif "system" in message_lower or "info" in message_lower:
            return "get_system_info", {"info_type": "general"}
        elif "run" in message_lower or "execute" in message_lower or "command" in message_lower:
            return "execute_command", {"command": user_message}
        else:
            return "search_documents", {"query": user_message, "limit": 3}

    def _simulate_tool_execution (self, tool_name: str, tool_args: Dict) -> str:
        """Simulate MCP tool execution and return mock response"""
        if tool_name == "search_documents":
            return json.dumps ({
                "results": [
                    {"title": "Company Policy Doc", "snippet": "Our security policies..."},
                    {"title": "Product Guide", "snippet": "Product features include..."}
                ]
            })
        elif tool_name == "get_system_info":
            return json.dumps ({
                "system": "Demo System",
                "version": "1.0.0",
                "status": "operational"
            })
        elif tool_name == "execute_command":
            return json.dumps ({
                "status": "success",
                "output": "Command executed successfully"
            })
        else:
            return json.dumps ({"result": "Tool executed"})

    def _generate_response_with_tool_data (self, user_message: str, tool_name: str, tool_response: str) -> str:
        """Generate AI response using tool data"""
        try:
            tool_data = json.loads (tool_response)

            if tool_name == "search_documents":
                results = tool_data.get ("results", [])
                if results:
                    response = f"I found {len (results)} relevant documents:\n\n"
                    for result in results:
                        response += f"• {result['title']}: {result['snippet']}\n"
                    return response
                else:
                    return "I couldn't find any relevant documents for your query."

            elif tool_name == "get_system_info":
                return f"System Information:\n- System: {tool_data.get ('system')}\n- Version: {tool_data.get ('version')}\n- Status: {tool_data.get ('status')}"

            else:
                return f"Tool '{tool_name}' executed successfully. Here's what I found:\n{json.dumps (tool_data, indent=2)}"

        except Exception as e:
            return f"I processed your request using {tool_name}, but encountered an issue formatting the response."


# Global instances
mcp_agent = None
security_components = None


@asynccontextmanager
async def lifespan (app: FastAPI):
    """FastAPI lifespan with MCP security initialization"""
    global mcp_agent, security_components

    logger.info ("🚀 Starting MCP Server Security Demo")

    try:
        # Initialize MCP Server security
        logger.info ("Initializing Palo Alto Networks MCP Server security...")
        security_components = initialize_mcp_security ()

        if security_components:
            logger.info ("✅ MCP Security initialized successfully")
        else:
            logger.warning ("⚠️ Security SDK not available - running without protection")

        # Initialize MCP demo agent
        mcp_agent = MCPDemoAgent (security_components)

        logger.info ("✅ MCP Demo Agent ready!")

    except Exception as e:
        logger.error (f"Failed to initialize: {e}")
        raise

    yield

    logger.info ("Shutting down MCP Demo")


app = FastAPI (
    title="MCP Server Security Demo",
    description="Palo Alto Networks Prisma AIRS - MCP Server Security Inspection",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest (BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse (BaseModel):
    model_config = {"protected_namespaces": ()}
    response: str
    session_id: str
    timestamp: datetime
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    security_info: Optional[Dict] = None
    blocked: bool = False


@app.get ("/")
async def root ():
    return {
        "message": "MCP Server Security Demo with Palo Alto Networks Prisma AIRS",
        "version": "1.0.0",
        "security_provider": "Palo Alto Networks Prisma AIRS",
        "security_profile": "mcp-server",
        "security_status": "enabled" if security_components else "disabled",
        "inspection_types": [
            "MCP tool invocations",
            "MCP server responses",
            "Agent-to-server communications"
        ]
    }


@app.post ("/chat", response_model=ChatResponse)
async def chat_endpoint (request: ChatRequest):
    if not mcp_agent:
        raise HTTPException (status_code=503, detail="MCP Agent not initialized")

    try:
        session_id = request.session_id or str (uuid.uuid4 ())

        start_time = time.time ()
        result = await mcp_agent.process_request (request.message, session_id)
        generation_time = time.time () - start_time

        return ChatResponse (
            response=result["response"],
            session_id=session_id,
            timestamp=datetime.now (),
            tool_name=result.get ("tool_name"),
            tool_args=result.get ("tool_args"),
            security_info=result.get ("security_info"),
            blocked=result.get ("blocked", False)
        )

    except Exception as e:
        logger.error (f"Chat error: {e}")
        raise HTTPException (status_code=500, detail=str (e))


@app.get ("/health")
async def health_check ():
    return {
        "status": "healthy",
        "security_enabled": security_components is not None,
        "agent_ready": mcp_agent is not None,
        "timestamp": datetime.now ().isoformat ()
    }


@app.get ("/security/status")
async def security_status ():
    """Detailed MCP security status check"""
    return {
        "security_available": SECURITY_AVAILABLE,
        "security_enabled": security_components is not None,
        "profile": "mcp-server",
        "scanner_available": security_components is not None and "scanner" in security_components,
        "api_key_configured": bool (os.getenv ("PALO_ALTO_API_KEY")),
        "inspection_phases": [
            "MCP Tool Invocation Inspection",
            "MCP Server Response Inspection"
        ],
        "protected_operations": [
            "Tool calls to MCP servers",
            "Resource access requests",
            "Server-to-server communications"
        ]
    }


@app.get ("/tools")
async def list_tools ():
    """List available MCP tools"""
    if not mcp_agent:
        raise HTTPException (status_code=503, detail="MCP Agent not initialized")

    return {
        "available_tools": mcp_agent.available_tools,
        "security_note": "All tool invocations are inspected by Prisma AIRS"
    }


if __name__ == "__main__":
    print ("=" * 80)
    print ("⚡ MCP SERVER SECURITY DEMO - PALO ALTO NETWORKS PRISMA AIRS")
    print ("=" * 80)
    print ("Security Features:")
    print ("  🔍 MCP Tool Invocation Inspection")
    print ("  🛡️ MCP Server Response Scanning")
    print ("  🔐 Centralized Agent Security Governance")
    print ("=" * 80)

    uvicorn.run ("mcp_demo:app", host="0.0.0.0", port=8078, reload=False)