from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

app = FastAPI()
OLLAMA = "http://127.0.0.1:11434"

agents = {
    1: {"model": None, "running": False},
    2: {"model": None, "running": False},
}


class AgentConfig(BaseModel):
    model: str


class ChatRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()


@app.get("/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA}/api/tags")
        return [m["name"] for m in r.json().get("models", [])]


@app.get("/agents")
async def get_agents():
    return agents


@app.post("/agent/{agent_id}/start")
async def start_agent(agent_id: int, config: AgentConfig):
    if agent_id not in agents:
        return {"error": "Invalid agent ID"}
    agents[agent_id] = {"model": config.model, "running": True}
    # Warm up — load model into memory
    async with httpx.AsyncClient(timeout=120.0) as client:
        await client.post(f"{OLLAMA}/api/generate", json={
            "model": config.model,
            "prompt": "",
            "keep_alive": "30m"
        })
    return {"status": "started", "agent": agents[agent_id]}


@app.post("/agent/{agent_id}/stop")
async def stop_agent(agent_id: int):
    if agent_id not in agents:
        return {"error": "Invalid agent ID"}
    model = agents[agent_id].get("model")
    if model:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(f"{OLLAMA}/api/generate", json={
                "model": model,
                "prompt": "",
                "keep_alive": 0
            })
    agents[agent_id]["running"] = False
    return {"status": "stopped", "agent": agents[agent_id]}


@app.post("/agent/{agent_id}/chat")
async def chat(agent_id: int, req: ChatRequest):
    if agent_id not in agents or not agents[agent_id]["running"]:
        return {"error": "Agent not running"}
    model = agents[agent_id]["model"]
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{OLLAMA}/api/generate", json={
            "model": model,
            "prompt": req.prompt,
            "stream": False
        })
    return {"output": r.json().get("response", "")}
