from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# model for output from google stt api
class Transcript(BaseModel):
    uid: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the transcript",
    )
    text: str = Field(..., min_length=1, description="Transcribed text from the audio")


# model for output from gemini api
class Mood(BaseModel):
    uid: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the mood analysis",
    )
    mood: list[tuple[str, float]] = Field(
        ..., min_length=1, description="Detected mood label from the audio"
    )
    confidence: float = Field(..., description="Confidence score of the mood detection")
    evidence: Optional[list[str]] = Field(
        ..., description="Evidence supporting the mood detection"
    )

    @field_validator("mood")
    @classmethod
    def check_mood_scores(cls, v):
        total = sum(score for _, score in v)
        if total == 0:
            raise ValueError("Sum of mood scores cannot be zero.")
        # Normalize scores so they sum to 1.0
        normalized = [(label, round(score / total, 4)) for label, score in v]
        for label, score in normalized:
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Mood score for '{label}' must be between 0.0 and 1.0"
                )
        return normalized

    @field_validator("confidence")
    @classmethod
    def round_float(cls, v: float) -> float:
        """Round floats to 2 decimal places."""
        return round(v, 2)
