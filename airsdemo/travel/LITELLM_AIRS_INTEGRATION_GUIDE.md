# SCM API key and profile information

# API KEY OGGelLXjMwEigstEZc3PYiHBzpDXd8LwJCylDGaRLk3gVC3i

# SCM Security Profile name mspov2

# LiteLLM + Palo Alto Networks AI Security (AIRS) Integration Guide

## Overview

This guide explains how to integrate **Palo Alto Networks Prisma AI Security (AIRS)** with **LiteLLM Proxy** to add real-time AI guardrails to any LLM-powered application. LiteLLM acts as a unified AI gateway that routes requests to multiple model providers (OpenAI, Anthropic, Google, etc.) while enforcing security policies through PANW AIRS on every request.

### What This Achieves

- **Prompt injection detection** - Blocks malicious prompts before they reach your models
- **Toxic/harmful content filtering** - Scans both user inputs and model outputs
- **Data loss prevention (DLP)** - Prevents sensitive data from leaking through AI responses
- **Sensitive data masking** - Automatically redacts PII, credentials, and other sensitive content from responses
- **Centralized security policy** - One guardrail configuration protects all models behind the proxy

### Architecture

```
User Request
     |
     v
+------------------+
| Your Application |  (Flask, FastAPI, Node.js, etc.)
| (OpenAI SDK)     |  Points to LiteLLM proxy instead of OpenAI directly
+------------------+
     |
     v
+------------------+     +---------------------------+
| LiteLLM Proxy    |<--->| PANW Prisma AIRS          |
| (port 4000)      |     | - Pre-call: scan input    |
|                  |     | - Post-call: scan output  |
+------------------+     +---------------------------+
     |
     v
+------------------+
| LLM Providers    |
| - OpenAI         |
| - Anthropic      |
| - Google Gemini  |
| - Any provider   |
+------------------+
```

---

## Prerequisites

1. **Palo Alto Networks Prisma AIRS access**
   - An AIRS API key (`PANW_PRISMA_AIRS_API_KEY`)
   - A security profile name (`PANW_PROFILE_NAME`) configured in AIRS portal
   - The AIRS API endpoint (typically `https://service.api.aisecurity.paloaltonetworks.com`)

2. **LiteLLM installed**
   ```bash
   pip install litellm>=1.40.0
   ```

3. **At least one LLM provider API key** (e.g., `OPENAI_API_KEY`)

---

## Step 1: Create the LiteLLM Configuration File

Create a file called `litellm_config.yaml`. This is the core configuration that defines your models and guardrails.

```yaml
# litellm_config.yaml

# =============================================
# MODEL DEFINITIONS
# =============================================
# Define the models your application will use.
# Each model gets an alias (model_name) that your
# application references instead of the raw provider model.
model_list:
  - model_name: orchestrator
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: researcher
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: calculator
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

# =============================================
# PALO ALTO NETWORKS AIRS GUARDRAILS
# =============================================
# Two guardrails are configured:
#   1. pre_call  - Scans the USER INPUT before it reaches the LLM
#   2. post_call - Scans the MODEL OUTPUT before it reaches the user
guardrails:
  # --- Pre-Call Guardrail (Input Scanning) ---
  - guardrail_name: "panw-prisma-airs-pre"
    litellm_params:
      guardrail: panw_prisma_airs           # Built-in LiteLLM guardrail type
      mode: "pre_call"                       # Fires BEFORE the LLM call
      api_key: os.environ/PANW_PRISMA_AIRS_API_KEY
      profile_name: os.environ/PANW_PROFILE_NAME
      api_base: "https://service.api.aisecurity.paloaltonetworks.com"
      app_name: "YourAppName"                # Identifies this app in AIRS logs

  # --- Post-Call Guardrail (Output Scanning) ---
  - guardrail_name: "panw-prisma-airs-post"
    litellm_params:
      guardrail: panw_prisma_airs           # Same guardrail type
      mode: "post_call"                      # Fires AFTER the LLM responds
      api_key: os.environ/PANW_PRISMA_AIRS_API_KEY
      profile_name: os.environ/PANW_PROFILE_NAME
      api_base: "https://service.api.aisecurity.paloaltonetworks.com"
      app_name: "YourAppName"
      mask_response_content: true            # Redact sensitive data in output

# =============================================
# GENERAL SETTINGS
# =============================================
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY  # Auth key for the proxy itself
```

### Configuration Explained

| Field | Description |
|-------|-------------|
| `guardrail_name` | A unique name you assign. Your application references this name to activate the guardrail. |
| `guardrail: panw_prisma_airs` | Tells LiteLLM to use the built-in Palo Alto Networks AIRS integration. |
| `mode: "pre_call"` | Scans the user's input **before** it is sent to the LLM. If the input violates policy, the request is blocked and never reaches the model. |
| `mode: "post_call"` | Scans the LLM's response **after** generation but **before** it is returned to the user. |
| `api_key` | Your PANW Prisma AIRS API key. Uses `os.environ/` syntax to read from environment variables (recommended). |
| `profile_name` | The security profile configured in the AIRS portal. This profile defines what categories to scan for (injection, toxicity, DLP, etc.). |
| `api_base` | The AIRS API endpoint. |
| `app_name` | A label that identifies your application in AIRS dashboards and logs. |
| `mask_response_content` | When `true`, AIRS will redact/mask sensitive data (PII, credentials) in the model's output instead of blocking the entire response. |

---

## Step 2: Set Environment Variables

Create a `.env` file (or set these in your environment):

```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-your-openai-key
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-key   # If using Anthropic models

# Palo Alto Networks AIRS
PANW_PRISMA_AIRS_API_KEY=your-airs-api-key
PANW_PROFILE_NAME=your-security-profile-name

# LiteLLM Proxy Auth
LITELLM_MASTER_KEY=sk-your-proxy-master-key     # You choose this; secures the proxy
```

---

## Step 3: Start the LiteLLM Proxy

```bash
litellm --config litellm_config.yaml --port 4000 --host 0.0.0.0
```

The proxy is now running on `http://localhost:4000` and exposes an **OpenAI-compatible API**. Any application that uses the OpenAI SDK can point to it with zero code changes to the LLM calling logic.

### Optional: Run as a systemd Service (Linux)

For production deployments, create a service file:

```ini
# /etc/systemd/system/litellm-proxy.service
[Unit]
Description=LiteLLM Proxy with Palo Alto Networks AI Security
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/your/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/path/to/your/app/.env
ExecStart=/path/to/your/venv/bin/litellm --config /path/to/litellm_config.yaml --port 4000 --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable litellm-proxy
sudo systemctl start litellm-proxy
```

---

## Step 4: Connect Your Application

The key insight: **your application uses the standard OpenAI SDK**, but points it at the LiteLLM proxy URL instead of `api.openai.com`. This means AIRS guardrails are enforced transparently -- no guardrail-specific code in your app.

### Python Example

```python
from openai import OpenAI

# Point the OpenAI SDK at the LiteLLM proxy
client = OpenAI(
    api_key="sk-your-proxy-master-key",     # The LITELLM_MASTER_KEY
    base_url="http://localhost:4000/v1"      # LiteLLM proxy URL
)

# Make a normal chat completion call
# Use the model alias from litellm_config.yaml (e.g., "orchestrator")
response = client.chat.completions.create(
    model="orchestrator",                    # Maps to openai/gpt-4o in config
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Plan a trip to Tokyo"}
    ],
    temperature=0.7,
    max_tokens=2000,
    # === ACTIVATE GUARDRAILS ===
    extra_body={
        "guardrails": ["panw-prisma-airs-pre", "panw-prisma-airs-post"]
    }
)

print(response.choices[0].message.content)
```

### Activating Guardrails per Request

Guardrails are **opt-in per request** via the `extra_body` parameter. This gives you fine-grained control:

```python
# Full protection: scan both input and output
kwargs["extra_body"] = {
    "guardrails": ["panw-prisma-airs-pre", "panw-prisma-airs-post"]
}

# Input scanning only (faster, skip output scan)
kwargs["extra_body"] = {
    "guardrails": ["panw-prisma-airs-pre"]
}

# Output scanning only
kwargs["extra_body"] = {
    "guardrails": ["panw-prisma-airs-post"]
}

# No guardrails (e.g., for internal-only model calls)
# Simply omit extra_body
```

### Selective Guardrails by Model Role

A common pattern is to only enable guardrails on user-facing interactions (where untrusted input enters the system), and skip them for internal model-to-model calls to reduce latency:

```python
def call_model(model_type, messages, enable_guardrails=True):
    client = OpenAI(
        api_key=LITELLM_MASTER_KEY,
        base_url="http://localhost:4000/v1"
    )

    kwargs = {
        "model": model_type,
        "messages": messages,
    }

    # Only apply guardrails to the user-facing orchestrator
    if enable_guardrails and model_type == "orchestrator":
        kwargs["extra_body"] = {
            "guardrails": ["panw-prisma-airs-pre", "panw-prisma-airs-post"]
        }

    return client.chat.completions.create(**kwargs)
```

---

## Step 5: Handle Guardrail Violations

When AIRS blocks a request, LiteLLM returns an error. Your application should catch and handle these gracefully:

```python
try:
    response = client.chat.completions.create(
        model="orchestrator",
        messages=messages,
        extra_body={"guardrails": ["panw-prisma-airs-pre", "panw-prisma-airs-post"]}
    )
except Exception as e:
    error_str = str(e)
    if "panw_prisma_airs_blocked" in error_str or "guardrail_violation" in error_str:
        # AIRS blocked the request -- inform the user
        print("Your request was blocked by AI Security policy.")
    else:
        # Other error (network, auth, etc.)
        print(f"Error: {e}")
```

---

## How the Flow Works End-to-End

```
1. User sends: "Plan a trip to Tokyo"
       |
2. App sends request to LiteLLM proxy (localhost:4000)
   with guardrails: ["panw-prisma-airs-pre", "panw-prisma-airs-post"]
       |
3. PRE-CALL: LiteLLM sends user input to PANW AIRS
   AIRS scans for: prompt injection, toxic content, policy violations
       |
   +-- If BLOCKED: LiteLLM returns error immediately (model never called)
   +-- If ALLOWED: continues to step 4
       |
4. LiteLLM forwards request to the configured LLM (e.g., OpenAI GPT-4o)
       |
5. LLM generates response
       |
6. POST-CALL: LiteLLM sends model output to PANW AIRS
   AIRS scans for: sensitive data, harmful content, policy violations
       |
   +-- If BLOCKED: LiteLLM returns error (response discarded)
   +-- If MASKED: AIRS redacts sensitive data, returns sanitized response
   +-- If ALLOWED: returns response as-is
       |
7. App receives clean, security-scanned response
```

---

## Multi-Model Gateway Pattern

LiteLLM's model aliasing lets you use multiple providers behind a single proxy, all protected by the same AIRS guardrails:

```yaml
model_list:
  # OpenAI models
  - model_name: fast-model
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # Anthropic models
  - model_name: research-model
    litellm_params:
      model: anthropic/claude-sonnet-4-5-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

  # Google models
  - model_name: gemini-model
    litellm_params:
      model: gemini/gemini-1.5-pro
      api_key: os.environ/GEMINI_API_KEY
```

Your application switches models by changing the `model` parameter -- the guardrails, auth, and routing are all handled by the proxy:

```python
# All three calls go through the same AIRS guardrails
client.chat.completions.create(model="fast-model", ...)
client.chat.completions.create(model="research-model", ...)
client.chat.completions.create(model="gemini-model", ...)
```

---

## Configuration Reference

### Guardrail Modes

| Mode | When it fires | What it scans | Use case |
|------|--------------|---------------|----------|
| `pre_call` | Before the LLM is called | User input (prompt) | Block prompt injection, toxic input, unauthorized requests |
| `post_call` | After the LLM responds | Model output | Block harmful output, mask PII/sensitive data, enforce DLP |

### Key `litellm_params` for AIRS

| Parameter | Required | Description |
|-----------|----------|-------------|
| `guardrail` | Yes | Must be `panw_prisma_airs` |
| `mode` | Yes | `pre_call` or `post_call` |
| `api_key` | Yes | PANW AIRS API key |
| `profile_name` | Yes | Security profile name from AIRS portal |
| `api_base` | Yes | AIRS API endpoint URL |
| `app_name` | No | Application identifier for AIRS dashboards |
| `mask_response_content` | No | `true` to redact sensitive data instead of blocking (post_call only) |

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` from AIRS | Invalid or expired AIRS API key | Regenerate key in AIRS portal, update `PANW_PRISMA_AIRS_API_KEY` |
| `Profile not found` | Wrong profile name | Check profile name in AIRS portal matches `PANW_PROFILE_NAME` exactly |
| All requests blocked | Profile too restrictive | Adjust security profile thresholds in AIRS portal |
| Guardrails not firing | Missing `extra_body` | Ensure your app passes `extra_body={"guardrails": [...]}` in the API call |
| Slow responses | Both pre and post scans | Consider using pre_call only for internal/trusted model chains |
| `Connection refused` on port 4000 | LiteLLM proxy not running | Start with `litellm --config litellm_config.yaml --port 4000` |

---

## Summary

| Component | Role |
|-----------|------|
| **LiteLLM Proxy** | OpenAI-compatible gateway that routes to any LLM provider |
| **PANW AIRS (pre_call)** | Scans user input for injection, toxicity, and policy violations before the model sees it |
| **PANW AIRS (post_call)** | Scans model output for sensitive data, harmful content; optionally masks PII |
| **Your Application** | Uses standard OpenAI SDK pointed at the proxy; activates guardrails via `extra_body` |

The result: **any application** that can call an OpenAI-compatible API gets enterprise AI security with minimal code changes -- just change the `base_url` and add `extra_body` to enable guardrails.
