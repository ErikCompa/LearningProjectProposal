from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QAEmotionPair(BaseModel):
    question: str
    answer: str
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    negative_emotion_percentages: Optional[dict[str, float]] = Field(default=None)
    is_direct: bool = Field(
        default=False, description="Whether the question was direct or indirect"
    )


class AgentSession(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    session_id: str
    created_at: datetime
    qa_pairs: list[QAEmotionPair]
    final_emotion: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    total_question_count: int = Field(ge=1)
    direct_question_count: int = Field(ge=0, le=5)
    audio_url: str


class EmotionAnalysisResult(BaseModel):
    emotion: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    negative_emotion_percentages: Optional[dict[str, float]]


class NextQuestionResult(BaseModel):
    question: str = Field(min_length=1)
    is_direct: bool
    music_requested: bool


class MusicRecommendationResult(BaseModel):
    song: str = Field(min_length=1)


class OrchestrationResult(BaseModel):
    emotion: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    negative_emotion_percentages: Optional[dict[str, float]] = None
    next_question: Optional[str] = None
    recommend_music: bool = False
    music_recommendation: Optional[dict[str, str]] = None
