from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QAMoodPair(BaseModel):
    question: str
    answer: str
    mood: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")


class AgentSession(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    session_id: str
    created_at: datetime
    qa_pairs: list[QAMoodPair]
    final_mood: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    question_count: int = Field(ge=1, le=5)
    audio_url: str


class MoodAnalysisResult(BaseModel):
    mood: str
    confidence: float = Field(ge=0.0, le=1.0)


class NextQuestionResult(BaseModel):
    question: str = Field(min_length=1)
