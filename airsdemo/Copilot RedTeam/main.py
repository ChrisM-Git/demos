"""
Copilot Studio Red Team Wrapper
--------------------------------
A FastAPI middleware that bridges the PANW AI Runtime Security red team tool
with Microsoft Copilot Studio via the Direct Line API.

Setup:
  1. Copy .env.example to .env
  2. Set COPILOT_TOKEN_ENDPOINT to your Copilot Studio Direct Line token URL
  3. Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

TOKEN_ENDPOINT = os.getenv("COPILOT_TOKEN_ENDPOINT")
DIRECTLINE_BASE = "https://directline.botframework.com/v3/directline"

if not TOKEN_ENDPOINT:
    raise RuntimeError("COPILOT_TOKEN_ENDPOINT is not set. Check your .env file.")

app = FastAPI(title="Copilot Studio Red Team Wrapper")


class ChatRequest(BaseModel):
    input: str


class ChatResponse(BaseModel):
    output: str


@app.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    async with httpx.AsyncClient(timeout=90.0) as client:

        # Step 1: Get token + conversation ID from Copilot Studio
        token_resp = await client.get(TOKEN_ENDPOINT)
        if token_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Token fetch failed: {token_resp.text}")

        token_data = token_resp.json()
        token = token_data["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Step 2: Start the conversation with Direct Line
        conv_resp = await client.post(
            f"{DIRECTLINE_BASE}/conversations",
            headers=headers,
        )
        if conv_resp.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=f"Conversation start failed: {conv_resp.text}")

        conversation_id = conv_resp.json()["conversationId"]

        # Step 3: Send the user message
        send_resp = await client.post(
            f"{DIRECTLINE_BASE}/conversations/{conversation_id}/activities",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "type": "message",
                "from": {"id": "airs-redteam"},
                "text": request.input,
            },
        )
        if send_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Message send failed: {send_resp.text}")

        # Step 3: Poll for bot response (up to 60 seconds)
        watermark = None
        for _ in range(30):
            await asyncio.sleep(2)

            params = {}
            if watermark:
                params["watermark"] = watermark

            poll_resp = await client.get(
                f"{DIRECTLINE_BASE}/conversations/{conversation_id}/activities",
                headers=headers,
                params=params,
            )
            if poll_resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Poll failed: {poll_resp.text}")

            data = poll_resp.json()
            watermark = data.get("watermark")

            for activity in data.get("activities", []):
                if (
                    activity.get("type") == "message"
                    and activity.get("from", {}).get("role") == "bot"
                ):
                    return ChatResponse(output=activity.get("text", ""))

        raise HTTPException(status_code=504, detail="Bot did not respond within 60 seconds")


@app.get("/health")
async def health():
    return {"status": "ok"}
