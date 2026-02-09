from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QAMoodPair(BaseModel):
    question: str
    answer: str
    mood: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    negative_emotion_percentages: Optional[dict[str, float]] = Field(default=None)


class AgentSession(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    session_id: str
    created_at: datetime
    qa_pairs: list[QAMoodPair]
    final_mood: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    total_question_count: int = Field(ge=1)
    direct_question_count: int = Field(ge=0, le=5)
    audio_url: str


class MoodAnalysisResult(BaseModel):
    mood: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    negative_emotion_percentages: Optional[dict[str, float]]
    music_requested: bool


class NextQuestionResult(BaseModel):
    question: str = Field(min_length=1)
    is_direct: bool


class MusicRecommendationResult(BaseModel):
    song: str = Field(min_length=1)
