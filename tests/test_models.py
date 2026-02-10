from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models import (
    AgentSession,
    EmotionAnalysisResult,
    MusicRecommendationResult,
    NextQuestionResult,
    QAEmotionPair,
)


class TestQAPair:
    """Test QAPair pydantic model"""

    def test_valid_qapair(self):
        """Test valid QAPair creation"""
        qa = QAEmotionPair(
            question="How are you feeling?",
            answer="I feel happy",
            emotion="joyful",
            confidence=0.85,
        )
        assert qa.question == "How are you feeling?"
        assert qa.answer == "I feel happy"
        assert qa.emotion == "joyful"
        assert qa.confidence == 0.85

    def test_confidence_boundaries(self):
        """Test confidence must be 0-1"""
        # Valid boundaries
        qa_min = QAEmotionPair(
            question="Q", answer="A", emotion="happy", confidence=0.0
        )
        assert qa_min.confidence == 0.0

        qa_max = QAEmotionPair(
            question="Q", answer="A", emotion="happy", confidence=1.0
        )
        assert qa_max.confidence == 1.0

        # Invalid - too low
        with pytest.raises(ValidationError):
            QAEmotionPair(question="Q", answer="A", emotion="happy", confidence=-0.1)

        # Invalid - too high
        with pytest.raises(ValidationError):
            QAEmotionPair(question="Q", answer="A", emotion="happy", confidence=1.1)

    def test_required_fields(self):
        """Test all fields are required"""
        with pytest.raises(ValidationError):
            QAEmotionPair()

    def test_optional_negative_emotions(self):
        """Test optional negative_emotion_percentages field"""
        qa_without = QAEmotionPair(
            question="Q", answer="A", emotion="happy", confidence=0.5
        )
        assert qa_without.negative_emotion_percentages is None

        qa_with = QAEmotionPair(
            question="Q",
            answer="A",
            emotion="sad",
            confidence=0.8,
            negative_emotion_percentages={"sadness": 0.6, "anxiety": 0.2},
        )
        assert qa_with.negative_emotion_percentages == {"sadness": 0.6, "anxiety": 0.2}


class TestAgentSession:
    """Test AgentSession pydantic model"""

    def test_valid_agent_session(self):
        """Test valid AgentSession creation"""
        now = datetime.now()
        qa_pair = QAEmotionPair(
            question="How are you?",
            answer="Good",
            emotion="content",
            confidence=0.9,
        )
        session = AgentSession(
            session_id="test-123",
            created_at=now,
            qa_pairs=[qa_pair],
            final_emotion="content",
            final_confidence=0.9,
            total_question_count=1,
            direct_question_count=1,
            audio_url="https://example.com/audio.flac",
        )
        assert session.session_id == "test-123"
        assert session.created_at == now
        assert len(session.qa_pairs) == 1
        assert session.qa_pairs[0].emotion == "content"
        assert session.final_emotion == "content"
        assert session.final_confidence == 0.9
        assert session.total_question_count == 1
        assert session.direct_question_count == 1
        assert session.audio_url == "https://example.com/audio.flac"

    def test_final_confidence_boundaries(self):
        """Test final_confidence must be 0-1"""
        now = datetime.now()
        qa_pair = QAEmotionPair(
            question="Q",
            answer="A",
            emotion="happy",
            confidence=0.5,
        )

        # Valid boundaries
        session_min = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_emotion="happy",
            final_confidence=0.0,
            total_question_count=1,
            direct_question_count=0,
            audio_url="url",
        )
        assert session_min.final_confidence == 0.0

        session_max = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_emotion="happy",
            final_confidence=1.0,
            total_question_count=1,
            direct_question_count=0,
            audio_url="url",
        )
        assert session_max.final_confidence == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_emotion="happy",
                final_confidence=1.5,
                total_question_count=1,
                direct_question_count=0,
                audio_url="url",
            )

    def test_question_count_boundaries(self):
        """Test total_question_count must be >= 1 and direct_question_count must be 0-5"""
        now = datetime.now()
        qa_pair = QAEmotionPair(
            question="Q", answer="A", emotion="happy", confidence=0.5
        )

        # Valid boundaries
        session_min = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_emotion="happy",
            final_confidence=0.5,
            total_question_count=1,
            direct_question_count=0,
            audio_url="url",
        )
        assert session_min.total_question_count == 1
        assert session_min.direct_question_count == 0

        session_max = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_emotion="happy",
            final_confidence=0.5,
            total_question_count=10,
            direct_question_count=5,
            audio_url="url",
        )
        assert session_max.total_question_count == 10
        assert session_max.direct_question_count == 5

        # Invalid - total too low
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_emotion="happy",
                final_confidence=0.5,
                total_question_count=0,
                direct_question_count=0,
                audio_url="url",
            )

        # Invalid - direct too high
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_emotion="happy",
                final_confidence=0.5,
                total_question_count=10,
                direct_question_count=6,
                audio_url="url",
            )

    def test_multiple_qa_pairs(self):
        """Test session with multiple Q&A pairs"""
        now = datetime.now()
        qa_pairs = [
            QAEmotionPair(question="Q1", answer="A1", emotion="happy", confidence=0.5),
            QAEmotionPair(
                question="Q2", answer="A2", emotion="content", confidence=0.7
            ),
            QAEmotionPair(question="Q3", answer="A3", emotion="joyful", confidence=0.9),
        ]
        session = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=qa_pairs,
            final_emotion="joyful",
            final_confidence=0.9,
            total_question_count=3,
            direct_question_count=2,
            audio_url="url",
        )
        assert len(session.qa_pairs) == 3
        assert session.qa_pairs[0].question == "Q1"
        assert session.qa_pairs[1].emotion == "content"

    def test_json_serialization(self):
        """Test datetime JSON serialization"""
        now = datetime.now()
        qa_pair = QAEmotionPair(
            question="Q", answer="A", emotion="happy", confidence=0.5
        )
        session = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_emotion="happy",
            final_confidence=0.5,
            total_question_count=1,
            direct_question_count=0,
            audio_url="url",
        )
        json_data = session.model_dump()
        assert "created_at" in json_data
        assert isinstance(json_data["created_at"], datetime)

    def test_required_fields(self):
        """Test all fields are required"""
        with pytest.raises(ValidationError):
            AgentSession()


class TestMoodAnalysisResult:
    """Test MoodAnalysisResult model"""

    def test_valid_mood_analysis(self):
        """Test valid MoodAnalysisResult creation"""
        result = EmotionAnalysisResult(
            emotion="happy",
            confidence=0.85,
            negative_emotion_percentages=None,
            music_requested=False,
        )
        assert result.emotion == "happy"
        assert result.confidence == 0.85
        assert result.negative_emotion_percentages is None
        assert result.music_requested is False

    def test_with_negative_emotions(self):
        """Test with negative emotion percentages"""
        result = EmotionAnalysisResult(
            emotion="sad",
            confidence=0.9,
            negative_emotion_percentages={"sadness": 0.7, "anxiety": 0.3},
            music_requested=True,
        )
        assert result.negative_emotion_percentages == {"sadness": 0.7, "anxiety": 0.3}
        assert result.music_requested is True


class TestNextQuestionResult:
    """Test NextQuestionResult model"""

    def test_valid_next_question(self):
        """Test valid NextQuestionResult creation"""
        result = NextQuestionResult(
            question="How are you feeling today?", is_direct=True
        )
        assert result.question == "How are you feeling today?"
        assert result.is_direct is True

    def test_indirect_question(self):
        """Test indirect question"""
        result = NextQuestionResult(question="Tell me about your day", is_direct=False)
        assert result.is_direct is False


class TestMusicRecommendationResult:
    """Test MusicRecommendationResult model"""

    def test_valid_recommendation(self):
        """Test valid MusicRecommendationResult creation"""
        result = MusicRecommendationResult(song="Happy by Pharrell Williams")
        assert result.song == "Happy by Pharrell Williams"

    def test_empty_song_invalid(self):
        """Test empty song name is invalid"""
        with pytest.raises(ValidationError):
            MusicRecommendationResult(song="")
