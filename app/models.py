from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4

class Transcript(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the transcript")
    text: str = Field(..., description="Transcribed text from the audio")
    confidence: Optional[float] = Field(None, description="Confidence score of the transcription")

class Mood(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the mood analysis")
    label: str = Field(..., description="Detected mood label from the audio")
    confidence: float = Field(..., description="Confidence score of the mood detection")
    evidence: Optional[list[str]] = Field(..., description="Evidence supporting the mood detection")

class Response(BaseModel):
    transcript: Optional[Transcript] = Field(..., description="Transcript data")
    mood: Optional[Mood] = Field(None, description="Mood analysis data if available")