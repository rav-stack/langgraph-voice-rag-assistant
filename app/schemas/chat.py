from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # server generates one if not sent


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]
    grounding_note: str
    latency_ms: float

class VoiceChatResponse(BaseModel):
    session_id: str
    transcript: str
    answer: str
    sources: list[str]
    grounding_note: str
    latency_ms: float    