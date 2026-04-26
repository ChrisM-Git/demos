# Copilot Studio Red Team Wrapper — Setup Guide

This wrapper bridges the **Palo Alto Networks AI Runtime Security (AIRS)** red team tool
with **Microsoft Copilot Studio** via the Direct Line API.

---

## Option 1: Local Windows 11 Setup (using ngrok)

Use this option when you want to run the wrapper on your laptop for testing or demos.

### Prerequisites
- Python 3.9+ (python.org — check "Add to PATH" during install)
- A Copilot Studio bot with Direct Line Speech channel enabled
- A free ngrok account (ngrok.com) — upgrade to Personal for scans over 2 hours

### Step 1 — Install ngrok
Open PowerShell and run:
```powershell
winget install ngrok.ngrok
```
Or download the Windows ZIP from **ngrok.com/download**, extract it, and add the folder to your PATH.

### Step 2 — Authenticate ngrok
1. Sign up at **ngrok.com**
2. Copy your auth token from the ngrok dashboard
3. Run:
```powershell
ngrok config add-authtoken <your-auth-token>
```

### Step 3 — Clone or download the wrapper
Place the following files in a folder (e.g., `C:\Users\<username>\copilot_wrapper`):
- `main.py`
- `requirements.txt`
- `.env.example`

### Step 4 — Create your virtual environment
Open PowerShell and run:
```powershell
cd C:\Users\<username>\copilot_wrapper
python -m venv .venv
.venv\Scripts\Activate.ps1
```
> **Note:** If you get a script execution error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first.

### Step 5 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 6 — Configure your Copilot endpoint
```powershell
copy .env.example .env
notepad .env
```
In the `.env` file, paste your Copilot Studio token endpoint:
```
COPILOT_TOKEN_ENDPOINT=https://<your-env>.environment.api.powerplatform.com/powervirtualagents/botsbyschema/<your-bot>/directline/token?api-version=2022-03-01-preview
```

**How to find your token endpoint:**
1. Open Copilot Studio
2. Go to **Settings → Channels → Direct Line Speech**
3. Copy the **Token Endpoint** URL

### Step 7 — Start the wrapper
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```
> **Warning:** Keep this PowerShell window open for the entire scan. If you close it the wrapper will stop and AIRS will lose connectivity. Disable sleep/screen lock on your laptop during the scan.

### Step 8 — Expose with ngrok (new PowerShell window)
Open a second PowerShell window and run:
```powershell
ngrok http 8000
```
Copy the public URL — it looks like:
```
https://abc123.ngrok-free.dev
```
> **Warning:** Keep this window open as well. You need TWO PowerShell windows running simultaneously — one for uvicorn and one for ngrok. Closing either will stop the scan.

> **Note:** ngrok Free tier sessions expire after 2 hours. If your scan runs longer, upgrade to ngrok Personal or use Option 2.

### Step 9 — Configure AIRS Red Team Tool
In the AIRS red team tool use **cURL Import**:
```bash
curl -X POST "https://abc123.ngrok-free.dev/" -H "Content-Type: application/json"
```

**Request JSON:**
```json
{
  "input": "{INPUT}"
}
```

**Response JSON:**
```json
{
  "output": "{RESPONSE}"
}
```

**Supports Sessions:** No

---

## Option 2: Customer / Production Setup (hosted server or frontend)

Use this option when the customer wants to run the wrapper on their own infrastructure
(cloud VM, web server, or existing backend).

### Prerequisites
- A Linux/Windows server or cloud VM (AWS, Azure, GCP)
- Python 3.9+
- A public IP or domain name
- Port 8000 open in firewall/security group (or any port of your choice)
- A Copilot Studio bot with Direct Line Speech channel enabled

### Step 1 — Copy wrapper files to server
Transfer the following files to your server:
- `main.py`
- `requirements.txt`
- `.env.example`

```bash
scp -r copilot_wrapper/ user@your-server:/opt/copilot_wrapper/
```

### Step 2 — Install Python and dependencies
```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv -y
cd /opt/copilot_wrapper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3 — Configure your Copilot endpoint
```bash
cp .env.example .env
nano .env
```
Set the token endpoint:
```
COPILOT_TOKEN_ENDPOINT=https://<your-env>.environment.api.powerplatform.com/powervirtualagents/botsbyschema/<your-bot>/directline/token?api-version=2022-03-01-preview
```

**How to find your token endpoint:**
1. Open Copilot Studio
2. Go to **Settings → Channels → Direct Line Speech**
3. Copy the **Token Endpoint** URL

### Step 4 — Run the wrapper
**For testing:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**For production (runs in background, auto-restarts):**
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --daemon
```

### Step 5 — Open firewall port
**AWS:** Add inbound rule for port 8000 in your Security Group
**Azure:** Add inbound port rule in Network Security Group
**GCP:** Add firewall rule allowing TCP port 8000

### Step 6 — (Optional) Add HTTPS with a reverse proxy
For production use, put the wrapper behind nginx with SSL:
```bash
sudo apt install nginx certbot python3-certbot-nginx -y
```
Configure nginx to proxy `https://yourdomain.com` → `http://localhost:8000`

### Step 7 — Configure AIRS Red Team Tool
In the AIRS red team tool use **cURL Import**:
```bash
curl -X POST "http://your-server-ip:8000/" -H "Content-Type: application/json"
```
Or with a domain:
```bash
curl -X POST "https://yourdomain.com/" -H "Content-Type: application/json"
```

**Request JSON:**
```json
{
  "input": "{INPUT}"
}
```

**Response JSON:**
```json
{
  "output": "{RESPONSE}"
}
```

**Supports Sessions:** No

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `COPILOT_TOKEN_ENDPOINT is not set` | Missing `.env` file | Copy `.env.example` to `.env` and fill in the URL |
| `502 Token fetch failed` | Wrong token endpoint URL | Double-check URL from Copilot Studio Direct Line settings |
| `502 Conversation start failed` | Direct Line API issue | Verify the bot is published in Copilot Studio |
| `504 Bot did not respond` | Bot too slow or not responding | Increase poll timeout in `main.py` or check bot health |
| `Connection refused` | Wrapper not running | Ensure uvicorn is running on port 8000 |
| ngrok URL not working | ngrok session expired | Restart ngrok and update URL in AIRS |

---

## Architecture Overview

```
AIRS Red Team Tool (cloud)
        |
        | HTTPS POST {"input": "..."}
        v
   ngrok / public IP
        |
        v
  FastAPI Wrapper (main.py)
        |
        |-- GET  Copilot Token Endpoint  --> Microsoft Power Platform
        |-- POST Direct Line /conversations --> Bot Framework
        |-- POST Direct Line /activities  --> Copilot Studio Bot
        |-- GET  Direct Line /activities  --> Poll for bot reply
        |
        | {"output": "..."}
        v
AIRS Red Team Tool (receives response)
```
