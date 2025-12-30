from typing import Optional
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4

class Transcript(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the transcript")
    text: str = Field(..., description="Transcribed text from the audio")
    confidence: Optional[float] = Field(None, description="Confidence score of the transcription")
    
    @field_validator("confidence")
    @classmethod
    def round_float(cls, v: float) -> float:
        """Round floats to 2 decimal places."""
        return round(v, 2)

class Mood(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the mood analysis")
    mood: str = Field(..., description="Detected mood label from the audio")
    confidence: float = Field(..., description="Confidence score of the mood detection")
    evidence: Optional[list[str]] = Field(..., description="Evidence supporting the mood detection")

    @field_validator("confidence")
    @classmethod
    def round_float(cls, v: float) -> float:
        """Round floats to 2 decimal places."""
        return round(v, 2)

class Response(BaseModel):
    transcript: Optional[Transcript] = Field(..., description="Transcript data")
    mood: Optional[Mood] = Field(None, description="Mood analysis data if available")
    