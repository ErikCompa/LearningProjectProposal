from datetime import datetime
from typing import List, Optional

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


class ConversationAgentResult(BaseModel):
    question: str = Field(min_length=1)
    is_direct: bool
    emotion: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    negative_emotion_percentages: Optional[dict[str, float]]


class MusicAgentResult(BaseModel):
    song: str = Field(min_length=1)
