from functools import cached_property
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator


# model for output from google stt api
class Transcript(BaseModel):
    uid: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the transcript",
    )
    text: str = Field(..., min_length=1, description="Transcribed text from the audio")


# model for output from gemini api
class MoodEntry(BaseModel):
    label: str
    score: float


class EvidenceEntry(BaseModel):
    label: str
    explanation: str


class Mood(BaseModel):
    uid: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the mood analysis",
    )
    mood: list[MoodEntry] = Field(
        ..., min_length=1, description="Detected mood label from the audio"
    )
    confidence: float = Field(
        ..., gt=0.0, lt=1.0, description="Confidence score of the mood detection"
    )
    evidence: list[EvidenceEntry] = Field(
        ..., description="Evidence supporting the mood and its score"
    )

    @computed_field
    @cached_property
    def top_50_moods(self) -> list[MoodEntry]:
        """Return top 50% moods based on scores."""
        sorted_moods = sorted(self.mood, key=lambda x: x.score, reverse=True)
        filtered = []
        total = 0.0
        for entry in sorted_moods:
            if total >= 0.5:
                break
            filtered.append(entry)
            total += entry.score
        return filtered

    @computed_field
    @cached_property
    def top_50_evidences(self) -> list[EvidenceEntry]:
        """Return evidences corresponding to top 50% moods."""
        top_labels = {entry.label for entry in self.top_50_moods}
        return [ev for ev in self.evidence if ev.label in top_labels]

    @field_validator("mood")
    @classmethod
    def check_mood_scores(cls, v):
        total = sum(entry.score for entry in v)
        if total == 0:
            raise ValueError("Sum of mood scores cannot be zero.")
        # Normalize scores so they sum to 1.0
        normalized = []
        for entry in v:
            norm_score = round(entry.score / total, 4)
            if not (0.0 <= norm_score <= 1.0):
                raise ValueError(
                    f"Mood score for '{entry.label}' must be between 0.0 and 1.0"
                )
            normalized.append(MoodEntry(label=entry.label, score=norm_score))
        return normalized

    @field_validator("confidence")
    @classmethod
    def round_float(cls, v: float) -> float:
        """Round floats to 2 decimal places."""
        return round(v, 2)
