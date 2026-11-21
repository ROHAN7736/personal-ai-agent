"""
REST API for the personal agent.

    uvicorn api:app --reload        then open http://127.0.0.1:8000/docs

Optional protection: set AGENT_API_KEY and send it in the X-API-Key header.
"""
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import agent

app = FastAPI(title="Personal AI Agent API", version="1.0.0")
API_KEY = os.getenv("AGENT_API_KEY", "")


def require_key(x_api_key: str = Header(default="")) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)


class ChatOut(BaseModel):
    reply: str
    provider: str
    model: str
    tokens_used_total: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": agent.PROVIDER, "model": agent.MODEL}


@app.post("/chat", response_model=ChatOut, dependencies=[Depends(require_key)])
def chat(body: ChatIn) -> ChatOut:
    try:
        reply = agent.chat(body.session_id, body.message)
    except Exception as e:
        agent.log.error("api chat failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Model error: {e}")
    return ChatOut(
        reply=reply,
        provider=agent.PROVIDER,
        model=agent.MODEL,
        tokens_used_total=agent.total_tokens,
    )


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_key)])
def clear_session(session_id: str) -> dict:
    agent.clear_history(session_id)
    return {"cleared": session_id}
