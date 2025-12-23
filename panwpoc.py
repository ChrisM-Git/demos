#!/usr/bin/env python3
"""
Palo Alto Networks AI Security Suite - Complete Interactive Demo
Includes both Runtime Security and Model Security with full functionality

DEPENDENCIES:
  pip install flask pan-aisecurity requests

Usage: python3 panwpoc.py
"""

import os
import ssl
import urllib3
import subprocess
import json
from pathlib import Path

# CRITICAL: Disable SSL verification for demo environments with self-signed certificates
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings (urllib3.exceptions.InsecureRequestWarning)

from urllib3 import connectionpool

_original_https_connection_init = connectionpool.HTTPSConnectionPool.__init__


def _patched_https_connection_init (self, *args, **kwargs):
    kwargs['cert_reqs'] = ssl.CERT_NONE
    kwargs['assert_hostname'] = False
    return _original_https_connection_init (self, *args, **kwargs)


connectionpool.HTTPSConnectionPool.__init__ = _patched_https_connection_init

from flask import Flask, request, jsonify
import requests

_original_request = requests.Session.request


def _patched_request (self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request (self, method, url, **kwargs)


requests.Session.request = _patched_request

import requests.adapters

_original_send = requests.adapters.HTTPAdapter.send


def _patched_send (self, request, **kwargs):
    kwargs['verify'] = False
    return _original_send (self, request, **kwargs)


requests.adapters.HTTPAdapter.send = _patched_send

# Import Palo Alto SDK AFTER patching
try:
    import aisecurity
    from aisecurity.scan.inline.scanner import Scanner
    from aisecurity.generated_openapi_client.models.ai_profile import AiProfile
    from aisecurity.scan.models.content import Content

    SECURITY_AVAILABLE = True
    print ("✅ Palo Alto Networks AI Security SDK loaded successfully")
    print ("⚠️  SSL verification disabled for demo/POC environment")
except ImportError as e:
    print (f"⚠️ Palo Alto Networks AI Security SDK not available: {e}")
    print ("💡 Install with: pip install pan-aisecurity")
    SECURITY_AVAILABLE = False

app = Flask (__name__)

# Global variables for Runtime Security
scanner = None
profile = None
api_key = None
security_profile = None

# Global variables for Model Security
model_security_configured = False
model_security_env = {}

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

# Models directory
MODELS_DIR = os.path.join (os.path.dirname (os.path.abspath (__file__)), "models")


# ============================================================================
# RUNTIME SECURITY FUNCTIONS (unchanged)
# ============================================================================

def initialize_security (user_api_key, user_profile):
    """Initialize security exactly like the working demo"""
    global scanner, profile, api_key, security_profile

    if not SECURITY_AVAILABLE:
        raise Exception ("AI Security SDK not available")

    try:
        aisecurity.init (api_key=user_api_key)
        profile = AiProfile (profile_name=user_profile)
        scanner = Scanner ()
        api_key = user_api_key
        security_profile = user_profile
        return True
    except Exception as e:
        raise Exception (f"Security initialization failed: {str (e)}")


def scan_prompt (user_prompt):
    """Scan user prompt"""
    if not scanner or not profile:
        return {"verdict": "ERROR", "message": "Security not initialized"}

    try:
        result = scanner.sync_scan (
            ai_profile=profile,
            content=Content (prompt=user_prompt.strip ()),
            metadata={"scan_type": "prompt_scan"}
        )

        verdict = "unknown"
        for field in ['verdict', 'status', 'action', 'decision']:
            if hasattr (result, field):
                value = getattr (result, field)
                if value:
                    verdict = str (value)
                    break

        threats = []
        has_actual_threats = False

        if hasattr (result, 'prompt_detected') and result.prompt_detected:
            pd = result.prompt_detected
            if hasattr (pd, 'injection') and pd.injection:
                threats.append ('prompt_injection')
                has_actual_threats = True
            if hasattr (pd, 'toxic_content') and pd.toxic_content:
                threats.append ('toxic_content')
                has_actual_threats = True
            if hasattr (pd, 'malicious_code') and pd.malicious_code:
                threats.append ('malicious_code')
                has_actual_threats = True
            if hasattr (pd, 'dlp') and pd.dlp:
                threats.append ('dlp_violation')
                has_actual_threats = True
            if hasattr (pd, 'topic_violation') and pd.topic_violation:
                threats.append ('topic_violation')
                has_actual_threats = True

        return {
            "verdict": verdict,
            "threats": threats,
            "should_block": has_actual_threats,
            "scan_id": getattr (result, 'scan_id', None),
            "session_id": getattr (result, 'session_id', None),
            "timestamp": getattr (result, 'timestamp', None),
            "scan_time_ms": getattr (result, 'scan_time_ms', None),
            "raw_response": str (result)
        }
    except Exception as e:
        return {"verdict": "ERROR", "message": f"Scan failed: {str (e)}"}


def scan_response (user_prompt, model_response):
    """Scan model response"""
    if not scanner or not profile:
        return {"verdict": "ERROR", "message": "Security not initialized"}

    try:
        result = scanner.sync_scan (
            ai_profile=profile,
            content=Content (
                prompt=user_prompt.strip (),
                response=model_response.strip ()
            ),
            metadata={"scan_type": "response_scan"}
        )

        verdict = "unknown"
        for field in ['verdict', 'status', 'action', 'decision']:
            if hasattr (result, field):
                value = getattr (result, field)
                if value:
                    verdict = str (value)
                    break

        threats = []
        has_actual_threats = False

        if hasattr (result, 'response_detected') and result.response_detected:
            rd = result.response_detected
            if hasattr (rd, 'toxic_content') and rd.toxic_content:
                threats.append ('response_toxic_content')
                has_actual_threats = True
            if hasattr (rd, 'malicious_code') and rd.malicious_code:
                threats.append ('response_malicious_code')
                has_actual_threats = True
            if hasattr (rd, 'dlp') and rd.dlp:
                threats.append ('response_dlp')
                has_actual_threats = True
            if hasattr (rd, 'topic_violation') and rd.topic_violation:
                threats.append ('response_topic_violation')
                has_actual_threats = True

        return {
            "verdict": verdict,
            "threats": threats,
            "should_block": has_actual_threats,
            "scan_id": getattr (result, 'scan_id', None),
            "session_id": getattr (result, 'session_id', None),
            "timestamp": getattr (result, 'timestamp', None),
            "scan_time_ms": getattr (result, 'scan_time_ms', None),
            "raw_response": str (result)
        }
    except Exception as e:
        return {"verdict": "ERROR", "message": f"Scan failed: {str (e)}"}


def call_ollama_model (user_message, model_name=DEFAULT_MODEL):
    """Call Ollama model"""
    try:
        response = requests.post (
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": user_message,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json ()
            return result.get ('response', 'No response from model')
        else:
            return f"Ollama error: {response.status_code}"

    except Exception as e:
        return f"Error calling Ollama: {str (e)}"


# ============================================================================
# MODEL SECURITY FUNCTIONS
# ============================================================================

def check_model_security_client ():
    """Check if model-security-client CLI is installed"""
    try:
        result = subprocess.run (
            ['model-security-scan', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_environment_variables ():
    """Check if required environment variables are set"""
    required_vars = ['MODEL_SECURITY_CLIENT_ID', 'MODEL_SECURITY_CLIENT_SECRET', 'TSG_ID']
    env_status = {}

    for var in required_vars:
        value = os.environ.get (var)
        env_status[var] = {
            'set': value is not None and len (value) > 0,
            'value': value if value else None
        }

    return env_status


def set_model_security_env (client_id, client_secret, tsg_id):
    """Set environment variables for model security"""
    global model_security_configured, model_security_env

    os.environ['MODEL_SECURITY_CLIENT_ID'] = client_id
    os.environ['MODEL_SECURITY_CLIENT_SECRET'] = client_secret
    os.environ['TSG_ID'] = tsg_id
    os.environ['MODEL_SECURITY_API_ENDPOINT'] = "https://api.sase.paloaltonetworks.com/aims"

    model_security_env = {
        'client_id': client_id,
        'client_secret': client_secret,
        'tsg_id': tsg_id
    }
    model_security_configured = True


def ensure_models_directory ():
    """Ensure the models directory exists"""
    Path (MODELS_DIR).mkdir (exist_ok=True)
    return MODELS_DIR


def list_local_models ():
    """List models in the local models directory"""
    ensure_models_directory ()
    models = []

    for item in Path (MODELS_DIR).iterdir ():
        if item.is_dir ():
            models.append ({
                'name': item.name,
                'path': str (item),
                'size': sum (f.stat ().st_size for f in item.rglob ('*') if f.is_file ())
            })

    return models


def scan_model (model_path, security_group_uuid, is_huggingface=False):
    """Scan a model using model-security-scan CLI"""
    if not model_security_configured:
        return {
            'success': False,
            'message': 'Model security not configured. Please set credentials first.'
        }

    if not check_model_security_client ():
        return {
            'success': False,
            'message': 'model-security-scan CLI not installed. Please install model-security-client.'
        }

    try:
        # Build command
        cmd = [
            'model-security-scan',
            '--model-path', model_path,
            '--security-group', security_group_uuid,
            '--poll-interval', '5',
            '--poll-timeout', '900'
        ]

        # Run scan
        result = subprocess.run (
            cmd,
            capture_output=True,
            text=True,
            timeout=1000  # 900 seconds for scan + buffer
        )

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'message': 'Scan completed successfully' if result.returncode == 0 else 'Scan failed'
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'message': 'Scan timed out. Large models may take longer than 15 minutes.'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error running scan: {str (e)}'
        }


# ============================================================================
# HTML CONTENT WITH INTERACTIVE MODEL SCANNING
# ============================================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Palo Alto Networks AI Security Suite</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }

        .header h1 {
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            color: #718096;
            font-size: 1.1em;
        }

        .tabs {
            display: flex;
            background: white;
            border-radius: 12px 12px 0 0;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            background: #f7fafc;
            border: none;
            font-size: 1.1em;
            font-weight: 600;
            color: #718096;
            transition: all 0.3s;
        }

        .tab:hover {
            background: #edf2f7;
        }

        .tab.active {
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }

        .tab-content {
            display: none;
            background: white;
            padding: 30px;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-height: calc(100vh - 300px);
            overflow-y: auto;
        }

        .tab-content.active {
            display: block;
        }

        .config-section {
            background: #f7fafc;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 25px;
        }

        .config-section h3 {
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 1.3em;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            color: #4a5568;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 1em;
            font-family: inherit;
        }

        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .form-group small {
            display: block;
            color: #718096;
            margin-top: 5px;
            font-size: 0.9em;
        }

        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5a67d8;
        }

        .btn-secondary {
            background: #48bb78;
            color: white;
        }

        .btn-secondary:hover {
            background: #38a169;
        }

        .btn-warning {
            background: #ed8936;
            color: white;
        }

        .btn-warning:hover {
            background: #dd6b20;
        }

        .btn:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
        }

        .alert {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        .alert-success {
            background: #c6f6d5;
            color: #22543d;
            border-left: 4px solid #48bb78;
        }

        .alert-error {
            background: #fed7d7;
            color: #742a2a;
            border-left: 4px solid #f56565;
        }

        .alert-info {
            background: #bee3f8;
            color: #2c5282;
            border-left: 4px solid #3182ce;
        }

        .alert-warning {
            background: #feebc8;
            color: #744210;
            border-left: 4px solid #ed8936;
        }

        .chat-container {
            background: #f7fafc;
            border-radius: 8px;
            padding: 20px;
            height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
        }

        .message-user {
            justify-content: flex-end;
        }

        .message-bubble {
            max-width: 70%;
            padding: 15px;
            border-radius: 12px;
            word-wrap: break-word;
        }

        .message-user .message-bubble {
            background: #667eea;
            color: white;
        }

        .message-bot .message-bubble {
            background: #e2e8f0;
            color: #2d3748;
        }

        .message-blocked .message-bubble {
            background: #fed7d7;
            color: #742a2a;
            border: 2px solid #f56565;
        }

        .metadata {
            font-size: 0.85em;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            max-width: 70%;
        }

        .metadata.allow {
            background: #c6f6d5;
            color: #22543d;
            border: 1px solid #48bb78;
        }

        .metadata.block {
            background: #fed7d7;
            color: #742a2a;
            border: 1px solid #f56565;
        }

        .metadata-label {
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 1.05em;
        }

        .metadata-item {
            margin: 5px 0;
            padding-left: 15px;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        .input-group input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 1em;
        }

        .scan-results {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }

        .status-badge.installed {
            background: #c6f6d5;
            color: #22543d;
        }

        .status-badge.not-installed {
            background: #fed7d7;
            color: #742a2a;
        }

        .status-badge.set {
            background: #c6f6d5;
            color: #22543d;
        }

        .status-badge.not-set {
            background: #fed7d7;
            color: #742a2a;
        }

        .model-list {
            list-style: none;
            padding: 0;
        }

        .model-item {
            background: #edf2f7;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.3s;
        }

        .model-item:hover {
            border-color: #667eea;
        }

        .model-item.selected {
            border-color: #667eea;
            background: #e6fffa;
        }

        .model-name {
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 5px;
        }

        .model-path {
            font-size: 0.85em;
            color: #718096;
            font-family: 'Courier New', monospace;
        }

        .instructions-box {
            background: #edf2f7;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }

        .instructions-box h4 {
            color: #2d3748;
            margin-bottom: 10px;
        }

        .instructions-box ol {
            margin-left: 20px;
            line-height: 1.8;
        }

        .code-inline {
            background: #2d3748;
            color: #e2e8f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .loading {
            display: inline-block;
            margin-left: 10px;
        }

        .loading::after {
            content: '...';
            animation: loading 1.5s infinite;
        }

        @keyframes loading {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Palo Alto Networks AI Security Suite</h1>
            <p>Complete AI Security Testing Platform</p>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('runtime')">⚡ Runtime Security</button>
            <button class="tab" onclick="switchTab('model')">🔍 Model Scanning</button>
        </div>

        <!-- Runtime Security Tab -->
        <div id="runtime-tab" class="tab-content active">
            <div class="config-section" id="runtimeConfigSection">
                <h3>🔐 Configuration</h3>
                <div id="runtimeStatus"></div>

                <div class="form-group">
                    <label>Palo Alto Networks API Key:</label>
                    <input type="password" id="apiKey" placeholder="pan_...">
                </div>

                <div class="form-group">
                    <label>Security Profile Name:</label>
                    <input type="text" id="securityProfile" placeholder="e.g., Retail, Banking">
                </div>

                <div class="form-group">
                    <label>Model Name:</label>
                    <select id="modelName">
                        <option value="llama3.2">llama3.2</option>
                        <option value="mistral:latest">mistral:latest</option>
                        <option value="phi:latest">phi:latest</option>
                    </select>
                </div>

                <button class="btn btn-primary" onclick="initializeSecurity()">Initialize Runtime Security</button>
            </div>

            <div id="runtimeChatSection" style="display:none;">
                <div class="config-section" style="background: #f7fafc; padding: 15px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>🛡️ Runtime Security Active</strong>
                            <span style="margin-left: 15px; color: #718096;">Profile: <span id="activeProfile"></span></span>
                            <span style="margin-left: 15px; color: #718096;">Model: <span id="activeModel"></span></span>
                        </div>
                        <button class="btn btn-warning" onclick="resetRuntimeConfig()" style="padding: 8px 15px;">⚙️ Reconfigure</button>
                    </div>
                </div>

                <div class="chat-container" id="chatMessages"></div>

                <div class="input-group">
                    <input type="text" id="userMessage" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                    <button class="btn btn-secondary" id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <!-- Model Security Tab -->
        <div id="model-tab" class="tab-content">
            <div class="config-section">
                <h3>🔐 Configuration</h3>
                <div id="modelSecurityStatus"></div>

                <div class="form-group">
                    <label>Model Security CLIENT_ID:</label>
                    <input type="text" id="clientId" placeholder="From Strata Cloud Manager">
                </div>

                <div class="form-group">
                    <label>Model Security CLIENT_SECRET:</label>
                    <input type="password" id="clientSecret" placeholder="From Strata Cloud Manager">
                </div>

                <div class="form-group">
                    <label>TSG_ID:</label>
                    <input type="text" id="tsgId" placeholder="Your Tenant Service Group ID">
                </div>

                <button class="btn btn-primary" onclick="configureModelSecurity()">Initialize Model Security</button>
            </div>

            <div class="config-section">
                <h3>🔍 Select Model to Scan</h3>

                <div class="form-group">
                    <label>Choose a Model:</label>
                    <select id="modelPreset" onchange="handleModelPresetChange()">
                        <option value="">-- Select a Model --</option>
                        <optgroup label="🤗 Example HuggingFace Models (Quick Scan)">
                            <option value="hf:gpt2">GPT-2 (Small, ~500MB, 2-3 min scan)</option>
                            <option value="hf:distilbert-base-uncased">DistilBERT Base (~250MB, 1-2 min scan)</option>
                            <option value="hf:microsoft/DialoGPT-small">DialoGPT Small (~350MB, 2-3 min scan)</option>
                        </optgroup>
                        <optgroup label="💾 Local Models">
                            <option value="local">Use a local model from ./models folder</option>
                        </optgroup>
                        <optgroup label="✏️ Custom">
                            <option value="custom">Enter custom HuggingFace model name</option>
                        </optgroup>
                    </select>
                </div>

                <!-- Custom HuggingFace Input (hidden by default) -->
                <div id="customModelSection" class="form-group" style="display:none;">
                    <label>Custom HuggingFace Model Name:</label>
                    <input type="text" id="customHfModel" placeholder="e.g., facebook/opt-125m" onchange="updateCliCommand()">
                    <small>Enter any model from https://huggingface.co/models</small>
                </div>

                <!-- Local Models Section (hidden by default) -->
                <div id="localSection" class="form-group" style="display:none;">
                    <label>Local Models:</label>
                    <div id="localModelsList"></div>
                    <button class="btn btn-secondary" onclick="refreshLocalModels()" style="margin-top:10px;">🔄 Refresh List</button>
                    <div class="alert alert-info" style="margin-top:15px;">
                        <strong>💡 To scan local models:</strong><br>
                        1. Create folder: <code class="code-inline">models/your-model-name/</code><br>
                        2. Place your model files inside<br>
                        3. Click "🔄 Refresh List"
                    </div>
                </div>

                <div class="form-group">
                    <label>Security Group:</label>
                    <select id="securityGroupSelect" onchange="updateSecurityGroup()">
                        <option value="e3adbe4f-4481-4bd9-9c7c-33b5123d40bf">HuggingFace Models (Default)</option>
                        <option value="d9778b1c-cf1b-493d-a1ca-ef2a99d2bae4">Local Models</option>
                        <option value="a165a2a2-e864-4426-883d-fe623646a014">S3 Storage</option>
                        <option value="d2f3939f-931e-43b3-a47f-df931d4fdc44">Azure Storage</option>
                        <option value="8c38ffc5-6bd5-4f28-8845-78debfed6b3c">GCS Storage</option>
                    </select>
                    <small>Auto-selected based on model source</small>
                </div>

                <button class="btn btn-secondary" id="scanBtn" onclick="startModelScan()">🔍 Scan Model</button>
                <span id="scanProgress" style="display:none; color:#667eea; font-weight:600;"></span>
            </div>

            <!-- CLI Command Display Section -->
            <div class="config-section" id="cliCommandSection">
                <h3>💻 CLI Command Preview</h3>
                <div class="scan-results" id="cliCommandDisplay" style="margin-top:10px; background:#1a202c;">
                    <span style="color:#68d391;"># Select a model above to see the command</span>
                </div>
                <div style="margin-top:10px;">
                    <small style="color:#718096;">
                        💡 You can copy this command and run it in your terminal manually
                    </small>
                </div>
            </div>

            <div id="scanResultsSection" style="display:none;">
                <h3>📊 Scan Results</h3>
                <div class="scan-results" id="scanResults"></div>
            </div>
        </div>
    </div>

    <script>
        var selectedLocalModel = null;
        var currentModelPath = '';
        var currentModelUri = '';

        // Tab Switching
        function switchTab(tab) {
            // Update tab buttons
            var tabs = document.querySelectorAll('.tab');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }

            // Find which button was clicked
            var buttons = document.querySelectorAll('.tab');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.indexOf(tab === 'runtime' ? 'Runtime' : 'Model') !== -1) {
                    buttons[i].classList.add('active');
                }
            }

            // Update tab content
            var contents = document.querySelectorAll('.tab-content');
            for (var i = 0; i < contents.length; i++) {
                contents[i].classList.remove('active');
            }
            document.getElementById(tab + '-tab').classList.add('active');

            // Load model security data if switching to that tab
            if (tab === 'model') {
                checkEnvironment();
                refreshLocalModels();
                updateCliCommand();
            }
        }

        // RUNTIME SECURITY FUNCTIONS
        function resetRuntimeConfig() {
            // Show config section, hide chat section
            document.getElementById('runtimeConfigSection').style.display = 'block';
            document.getElementById('runtimeChatSection').style.display = 'none';

            // Clear chat messages
            document.getElementById('chatMessages').innerHTML = '';

            // Clear status
            document.getElementById('runtimeStatus').innerHTML = '';
        }

        async function initializeSecurity() {
            const apiKey = document.getElementById('apiKey').value;
            const securityProfile = document.getElementById('securityProfile').value;
            const statusDiv = document.getElementById('runtimeStatus');

            if (!apiKey || !securityProfile) {
                statusDiv.innerHTML = '<div class="alert alert-error">Please enter both API key and security profile</div>';
                return;
            }

            try {
                const response = await fetch('/api/initialize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: apiKey, security_profile: securityProfile })
                });

                const data = await response.json();

                if (data.success) {
                    // Hide config section, show chat section
                    document.getElementById('runtimeConfigSection').style.display = 'none';
                    document.getElementById('runtimeChatSection').style.display = 'block';

                    // Update active status bar
                    document.getElementById('activeProfile').textContent = securityProfile;
                    document.getElementById('activeModel').textContent = document.getElementById('modelName').value;

                    // Focus on input field
                    setTimeout(function() {
                        document.getElementById('userMessage').focus();
                    }, 100);
                } else {
                    statusDiv.innerHTML = '<div class="alert alert-error">❌ ' + data.message + '</div>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="alert alert-error">❌ Network error: ' + error.message + '</div>';
            }
        }

        async function sendMessage() {
            const input = document.getElementById('userMessage');
            const message = input.value.trim();
            const modelName = document.getElementById('modelName').value;
            const sendBtn = document.getElementById('sendBtn');

            if (!message) return;

            input.value = '';
            sendBtn.disabled = true;

            addMessage(message, 'user');

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, model: modelName })
                });

                const data = await response.json();

                if (data.error) {
                    addMessage('❌ ERROR: ' + data.error, 'bot');
                } else if (data.blocked) {
                    addMessage('🚫 ' + data.message.toUpperCase(), 'blocked', data.details);
                } else {
                    addMessage(data.response, 'bot', data.prompt_scan, data.response_scan);
                }
            } catch (error) {
                addMessage('❌ Network error: ' + error.message, 'bot');
            }

            sendBtn.disabled = false;
        }

        function addMessage(content, type, promptScan, responseScan) {
            if (promptScan === undefined) promptScan = null;
            if (responseScan === undefined) responseScan = null;

            var messagesDiv = document.getElementById('chatMessages');
            var messageDiv = document.createElement('div');
            messageDiv.className = 'message message-' + type;

            var bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'message-bubble';
            bubbleDiv.textContent = content;

            messageDiv.appendChild(bubbleDiv);

            if (promptScan || responseScan) {
                var metadataDiv = document.createElement('div');
                var isBlocked = (promptScan && promptScan.should_block) || (responseScan && responseScan.should_block);
                metadataDiv.className = 'metadata ' + (isBlocked ? 'block' : 'allow');

                var metadataHTML = '<div class="metadata-label">🔍 Security Scan Results:</div>';

                if (promptScan) {
                    metadataHTML += '<div style="margin-top: 8px; font-weight: 600;">📥 PROMPT SCAN:</div>';
                    metadataHTML += '<div class="metadata-item"><strong>Verdict:</strong> ' + (promptScan.verdict || 'N/A') + '</div>';
                    metadataHTML += '<div class="metadata-item"><strong>Action:</strong> ' + (promptScan.should_block ? '🚫 BLOCKED' : '✅ ALLOWED') + '</div>';

                    if (promptScan.threats && promptScan.threats.length > 0) {
                        metadataHTML += '<div class="metadata-item"><strong>Threats:</strong> ' + promptScan.threats.join(', ') + '</div>';
                    }

                    if (promptScan.scan_id) metadataHTML += '<div class="metadata-item"><strong>Scan ID:</strong> ' + promptScan.scan_id + '</div>';
                    if (promptScan.scan_time_ms) metadataHTML += '<div class="metadata-item"><strong>Scan Time:</strong> ' + promptScan.scan_time_ms + 'ms</div>';
                }

                if (responseScan) {
                    metadataHTML += '<div style="margin-top: 12px; font-weight: 600;">📤 RESPONSE SCAN:</div>';
                    metadataHTML += '<div class="metadata-item"><strong>Verdict:</strong> ' + (responseScan.verdict || 'N/A') + '</div>';
                    metadataHTML += '<div class="metadata-item"><strong>Action:</strong> ' + (responseScan.should_block ? '🚫 BLOCKED' : '✅ ALLOWED') + '</div>';

                    if (responseScan.threats && responseScan.threats.length > 0) {
                        metadataHTML += '<div class="metadata-item"><strong>Threats:</strong> ' + responseScan.threats.join(', ') + '</div>';
                    }

                    if (responseScan.scan_id) metadataHTML += '<div class="metadata-item"><strong>Scan ID:</strong> ' + responseScan.scan_id + '</div>';
                    if (responseScan.scan_time_ms) metadataHTML += '<div class="metadata-item"><strong>Scan Time:</strong> ' + responseScan.scan_time_ms + 'ms</div>';
                }

                metadataDiv.innerHTML = metadataHTML;
                messageDiv.appendChild(metadataDiv);
            }

            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        // MODEL SECURITY FUNCTIONS
        async function checkEnvironment() {
            try {
                const response = await fetch('/api/model-security/check-env');
                const data = await response.json();

                var statusHTML = '<div class="alert alert-info">';
                statusHTML += '<strong>Environment Status:</strong><br>';
                statusHTML += 'CLI Installed: <span class="status-badge ' + (data.cli_installed ? 'installed' : 'not-installed') + '">' + (data.cli_installed ? 'YES' : 'NO') + '</span><br>';
                statusHTML += 'CLIENT_ID: <span class="status-badge ' + (data.env_vars.MODEL_SECURITY_CLIENT_ID.set ? 'set' : 'not-set') + '">' + (data.env_vars.MODEL_SECURITY_CLIENT_ID.set ? 'SET' : 'NOT SET') + '</span><br>';
                statusHTML += 'CLIENT_SECRET: <span class="status-badge ' + (data.env_vars.MODEL_SECURITY_CLIENT_SECRET.set ? 'set' : 'not-set') + '">' + (data.env_vars.MODEL_SECURITY_CLIENT_SECRET.set ? 'SET' : 'NOT SET') + '</span><br>';
                statusHTML += 'TSG_ID: <span class="status-badge ' + (data.env_vars.TSG_ID.set ? 'set' : 'not-set') + '">' + (data.env_vars.TSG_ID.set ? 'SET' : 'NOT SET') + '</span>';

                if (!data.cli_installed) {
                    statusHTML += '<br><br><strong>⚠️ CLI not installed.</strong> Install with: <code class="code-inline">pip install model-security-client --extra-index-url $(./get_pypi_url.sh)</code>';
                }

                statusHTML += '</div>';

                document.getElementById('modelSecurityStatus').innerHTML = statusHTML;
            } catch (error) {
                document.getElementById('modelSecurityStatus').innerHTML = 
                    '<div class="alert alert-error">Error checking environment: ' + error.message + '</div>';
            }
        }

        async function configureModelSecurity() {
            const clientId = document.getElementById('clientId').value;
            const clientSecret = document.getElementById('clientSecret').value;
            const tsgId = document.getElementById('tsgId').value;
            const statusDiv = document.getElementById('modelSecurityStatus');

            if (!clientId || !clientSecret || !tsgId) {
                statusDiv.innerHTML = '<div class="alert alert-error">Please enter CLIENT_ID, CLIENT_SECRET, and TSG_ID</div>';
                return;
            }

            try {
                const response = await fetch('/api/model-security/configure', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        client_id: clientId,
                        client_secret: clientSecret,
                        tsg_id: tsgId
                    })
                });

                const data = await response.json();

                if (data.success) {
                    statusDiv.innerHTML = '<div class="alert alert-success">✅ Model Security configured successfully!</div>';
                    checkEnvironment();
                    updateCliCommand();
                } else {
                    statusDiv.innerHTML = '<div class="alert alert-error">❌ ' + data.message + '</div>';
                }
            } catch (error) {
                statusDiv.innerHTML = '<div class="alert alert-error">❌ Error: ' + error.message + '</div>';
            }
        }

        function handleModelPresetChange() {
            const preset = document.getElementById('modelPreset').value;
            const customSection = document.getElementById('customModelSection');
            const localSection = document.getElementById('localSection');
            const securityGroupSelect = document.getElementById('securityGroupSelect');

            // Hide all sections first
            customSection.style.display = 'none';
            localSection.style.display = 'none';

            if (preset.startsWith('hf:')) {
                // HuggingFace preset model
                const modelName = preset.substring(3);
                currentModelUri = 'https://huggingface.co/' + modelName;
                currentModelPath = '';
                securityGroupSelect.selectedIndex = 0; // HuggingFace group
                updateCliCommand();
            } else if (preset === 'custom') {
                // Show custom input
                customSection.style.display = 'block';
                currentModelUri = '';
                currentModelPath = '';
                securityGroupSelect.selectedIndex = 0; // HuggingFace group
                updateCliCommand();
            } else if (preset === 'local') {
                // Show local models
                localSection.style.display = 'block';
                currentModelUri = '';
                currentModelPath = selectedLocalModel || '';
                securityGroupSelect.selectedIndex = 1; // Local group
                refreshLocalModels();
                updateCliCommand();
            } else {
                // Nothing selected
                currentModelUri = '';
                currentModelPath = '';
                updateCliCommand();
            }
        }

        function updateSecurityGroup() {
            updateCliCommand();
        }

        async function refreshLocalModels() {
            try {
                const response = await fetch('/api/model-security/list-local-models');
                const data = await response.json();

                const listDiv = document.getElementById('localModelsList');

                if (data.models.length === 0) {
                    listDiv.innerHTML = '<div class="alert alert-warning">No models found in ./models directory. See instructions below to add models.</div>';
                    listDiv.innerHTML += '<p><strong>Models directory:</strong> <code class="code-inline">' + data.models_dir + '</code></p>';
                } else {
                    var html = '<ul class="model-list" id="modelListUl">';
                    data.models.forEach(function(model, index) {
                        var sizeInMB = (model.size / 1024 / 1024).toFixed(2);
                        html += '<li class="model-item" data-model-path="' + model.path + '" data-model-index="' + index + '">';
                        html += '<div class="model-name">' + model.name + '</div>';
                        html += '<div class="model-path">' + model.path + ' (' + sizeInMB + ' MB)</div>';
                        html += '</li>';
                    });
                    html += '</ul>';
                    listDiv.innerHTML = html;

                    // Add click listeners to all model items
                    var items = document.querySelectorAll('.model-item');
                    items.forEach(function(item) {
                        item.addEventListener('click', function() {
                            var path = this.getAttribute('data-model-path');
                            var index = parseInt(this.getAttribute('data-model-index'));
                            selectLocalModel(path, index);
                        });
                    });
                }
            } catch (error) {
                document.getElementById('localModelsList').innerHTML = 
                    '<div class="alert alert-error">Error loading models: ' + error.message + '</div>';
            }
        }

        function selectLocalModel(path, index) {
            selectedLocalModel = path;

            // Update UI
            document.querySelectorAll('.model-item').forEach(function(item) {
                item.classList.remove('selected');
            });
            document.querySelectorAll('.model-item')[index].classList.add('selected');

            // Update CLI command
            updateCliCommand();
        }

        function updateCliCommand() {
            const displayDiv = document.getElementById('cliCommandDisplay');
            const securityGroupSelect = document.getElementById('securityGroupSelect');
            const securityGroupUuid = securityGroupSelect.value;
            const customHfModel = document.getElementById('customHfModel').value;

            // Check if custom model is entered
            if (customHfModel) {
                currentModelUri = 'https://huggingface.co/' + customHfModel;
                currentModelPath = '';
            }

            // Validate we have what we need
            if (!currentModelPath && !currentModelUri) {
                displayDiv.innerHTML = '<span style="color:#f6ad55;">⚠️  Please select a model above</span>';
                return;
            }

            // Build the command
            var command = '<span style="color:#68d391;"># Set environment variables first:</span><br>';
            command += '<span style="color:#fc8181;">export</span> <span style="color:#90cdf4;">MODEL_SECURITY_CLIENT_ID</span>=<span style="color:#fbd38d;">"your-client-id"</span><br>';
            command += '<span style="color:#fc8181;">export</span> <span style="color:#90cdf4;">MODEL_SECURITY_CLIENT_SECRET</span>=<span style="color:#fbd38d;">"your-client-secret"</span><br>';
            command += '<span style="color:#fc8181;">export</span> <span style="color:#90cdf4;">TSG_ID</span>=<span style="color:#fbd38d;">"your-tsg-id"</span><br><br>';

            command += '<span style="color:#68d391;"># Run the scan:</span><br>';
            command += '<span style="color:#fc8181;">model-security-scan</span> ';

            if (currentModelPath) {
                command += '<span style="color:#90cdf4;">--model-path</span> <span style="color:#fbd38d;">"' + currentModelPath + '"</span> ';
            } else if (currentModelUri) {
                command += '<span style="color:#90cdf4;">--model-uri</span> <span style="color:#fbd38d;">"' + currentModelUri + '"</span> ';
            }

            command += '<br>&nbsp;&nbsp;<span style="color:#90cdf4;">--security-group</span> <span style="color:#fbd38d;">"' + securityGroupUuid + '"</span> ';
            command += '<br>&nbsp;&nbsp;<span style="color:#90cdf4;">--poll-interval</span> <span style="color:#b794f4;">5</span> ';
            command += '<br>&nbsp;&nbsp;<span style="color:#90cdf4;">--poll-timeout</span> <span style="color:#b794f4;">900</span>';

            displayDiv.innerHTML = command;
        }

        async function startModelScan() {
            const securityGroupSelect = document.getElementById('securityGroupSelect');
            const securityGroupUuid = securityGroupSelect.value;

            // Check if we have a model selected
            if (!currentModelPath && !currentModelUri) {
                alert('Please select a model to scan');
                return;
            }

            const modelPath = currentModelUri || currentModelPath;
            const isHuggingface = !!currentModelUri;

            // Disable button and show progress
            const scanBtn = document.getElementById('scanBtn');
            const progressSpan = document.getElementById('scanProgress');
            scanBtn.disabled = true;
            progressSpan.style.display = 'inline';
            progressSpan.innerHTML = '<span class="loading">Scanning model (this may take 10-15 minutes for large models)</span>';

            // Hide previous results
            document.getElementById('scanResultsSection').style.display = 'none';

            try {
                const response = await fetch('/api/model-security/scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_path: modelPath,
                        security_group_uuid: securityGroupUuid,
                        is_huggingface: source === 'huggingface'
                    })
                });

                const data = await response.json();

                // Show results
                document.getElementById('scanResultsSection').style.display = 'block';
                const resultsDiv = document.getElementById('scanResults');

                if (data.success) {
                    resultsDiv.textContent = '✅ SCAN COMPLETED SUCCESSFULLY\\n\\n' + 'STDOUT:\\n' + data.stdout + '\\n\\n' + 'STDERR:\\n' + data.stderr;
                } else {
                    resultsDiv.textContent = '❌ SCAN FAILED\\n\\n' + 'Message: ' + data.message + '\\n\\n' + (data.stdout ? 'STDOUT:\\n' + data.stdout + '\\n\\n' : '') + (data.stderr ? 'STDERR:\\n' + data.stderr : '');
                }
            } catch (error) {
                document.getElementById('scanResultsSection').style.display = 'block';
                document.getElementById('scanResults').textContent = '❌ ERROR: ' + error.message;
            } finally {
                scanBtn.disabled = false;
                progressSpan.style.display = 'none';
            }
        }

        // Initialize on load
        window.addEventListener('load', function() {
            checkEnvironment();
            updateCliCommand();
        });
    </script>
</body>
</html>
"""


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route ('/')
def index ():
    """Serve the web interface"""
    return HTML_CONTENT


@app.route ('/api/initialize', methods=['POST'])
def api_initialize ():
    """Initialize runtime security"""
    data = request.json
    user_api_key = data.get ('api_key')
    user_profile = data.get ('security_profile')

    if not user_api_key or not user_profile:
        return jsonify ({
            'success': False,
            'message': 'API key and security profile are required'
        })

    try:
        initialize_security (user_api_key, user_profile)
        return jsonify ({
            'success': True,
            'message': 'Security initialized successfully'
        })
    except Exception as e:
        return jsonify ({
            'success': False,
            'message': str (e)
        })


@app.route ('/api/chat', methods=['POST'])
def api_chat ():
    """Process chat message with security scanning"""
    data = request.json
    user_message = data.get ('message')
    model_name = data.get ('model', DEFAULT_MODEL)

    if not user_message:
        return jsonify ({'error': 'No message provided'})

    if not scanner:
        return jsonify ({
            'error': 'Security not initialized. Please enter your credentials first.'
        })

    prompt_scan = scan_prompt (user_message)

    if prompt_scan['verdict'] == 'ERROR':
        return jsonify ({
            'error': f"Palo Alto API Error: {prompt_scan.get ('message', 'Unknown error')}"
        })

    if prompt_scan.get ('should_block', False):
        return jsonify ({
            'blocked': True,
            'message': 'Prompt blocked by security scan',
            'details': prompt_scan
        })

    model_response = call_ollama_model (user_message, model_name)
    response_scan = scan_response (user_message, model_response)

    if response_scan['verdict'] == 'ERROR':
        return jsonify ({
            'error': f"Palo Alto API Error: {response_scan.get ('message', 'Unknown error')}"
        })

    if response_scan.get ('should_block', False):
        return jsonify ({
            'blocked': True,
            'message': 'Response blocked by security scan',
            'details': response_scan,
            'original_response': model_response
        })

    return jsonify ({
        'blocked': False,
        'response': model_response,
        'prompt_scan': prompt_scan,
        'response_scan': response_scan
    })


# MODEL SECURITY API ROUTES
@app.route ('/api/model-security/check-env', methods=['GET'])
def api_check_env ():
    """Check environment variables and CLI installation"""
    return jsonify ({
        'cli_installed': check_model_security_client (),
        'env_vars': check_environment_variables (),
        'models_dir': ensure_models_directory ()
    })


@app.route ('/api/model-security/configure', methods=['POST'])
def api_configure_model_security ():
    """Configure model security credentials"""
    data = request.json
    client_id = data.get ('client_id')
    client_secret = data.get ('client_secret')
    tsg_id = data.get ('tsg_id')

    if not client_id or not client_secret or not tsg_id:
        return jsonify ({
            'success': False,
            'message': 'CLIENT_ID, CLIENT_SECRET, and TSG_ID are required'
        })

    try:
        set_model_security_env (client_id, client_secret, tsg_id)
        return jsonify ({
            'success': True,
            'message': 'Model security configured successfully'
        })
    except Exception as e:
        return jsonify ({
            'success': False,
            'message': str (e)
        })


@app.route ('/api/model-security/list-local-models', methods=['GET'])
def api_list_local_models ():
    """List local models in the models directory"""
    return jsonify ({
        'models': list_local_models (),
        'models_dir': ensure_models_directory ()
    })


@app.route ('/api/model-security/scan', methods=['POST'])
def api_scan_model ():
    """Scan a model"""
    data = request.json
    model_path = data.get ('model_path')
    security_group_uuid = data.get ('security_group_uuid')
    is_huggingface = data.get ('is_huggingface', False)

    if not model_path or not security_group_uuid:
        return jsonify ({
            'success': False,
            'message': 'model_path and security_group_uuid are required'
        })

    result = scan_model (model_path, security_group_uuid, is_huggingface)
    return jsonify (result)


if __name__ == '__main__':
    print ("=" * 70)
    print ("🛡️ Palo Alto Networks AI Security Suite")
    print ("=" * 70)
    print ("\nFeatures:")
    print ("  ⚡ Runtime Security - Real-time prompt & response scanning")
    print ("  🔍 Model Security - AI model vulnerability scanning")
    print ("\nQuick Setup:")
    print ("  1. pip install -r requirements.txt")
    print ("  2. ollama serve (for Runtime Security)")
    print ("  3. model-security-client (for Model Security)")
    print ("\nModels Directory:")
    print (f"  📁 {ensure_models_directory ()}")
    print ("  Place your model files in this directory for local scanning")
    print ("\nStarting web server at http://localhost:5001")
    print ("=" * 70)

    app.run (debug=True, host='0.0.0.0', port=5001)