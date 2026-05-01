"""Pydantic request/response shapes for the public HTTP API."""
from typing import Optional
from pydantic import BaseModel


class FrameRequest(BaseModel):
    meeting_id: str
    student_id: str
    frame_base64: str


class FrameResponse(BaseModel):
    attention: float
    fatigue: float
    boredom: float
    face_detected: bool
    model_fallback: bool
    backend: Optional[str]
