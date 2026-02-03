from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QAPair(BaseModel):
    """Single question-answer pair with detected emotion"""

    question: str
    answer: str
    mood: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    depth: int = Field(ge=1, le=3, description="Emotion depth level 1-3")


class AgentSession(BaseModel):
    """Complete agent session with all Q&A pairs and final result"""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    session_id: str
    created_at: datetime
    qa_pairs: list[QAPair]
    final_mood: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    final_depth: int = Field(ge=1, le=3)
    question_count: int = Field(ge=1, le=5)
    audio_url: str
