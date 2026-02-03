from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models import AgentSession, QAPair


class TestQAPair:
    """Test QAPair pydantic model"""

    def test_valid_qapair(self):
        """Test valid QAPair creation"""
        qa = QAPair(
            question="How are you feeling?",
            answer="I feel happy",
            mood="joyful",
            confidence=0.85,
            depth=3,
        )
        assert qa.question == "How are you feeling?"
        assert qa.answer == "I feel happy"
        assert qa.mood == "joyful"
        assert qa.confidence == 0.85
        assert qa.depth == 3

    def test_confidence_boundaries(self):
        """Test confidence must be 0-1"""
        # Valid boundaries
        qa_min = QAPair(question="Q", answer="A", mood="happy", confidence=0.0, depth=1)
        assert qa_min.confidence == 0.0

        qa_max = QAPair(question="Q", answer="A", mood="happy", confidence=1.0, depth=1)
        assert qa_max.confidence == 1.0

        # Invalid - too low
        with pytest.raises(ValidationError):
            QAPair(question="Q", answer="A", mood="happy", confidence=-0.1, depth=1)

        # Invalid - too high
        with pytest.raises(ValidationError):
            QAPair(question="Q", answer="A", mood="happy", confidence=1.1, depth=1)

    def test_depth_boundaries(self):
        """Test depth must be 1-3"""
        # Valid boundaries
        qa_min = QAPair(question="Q", answer="A", mood="happy", confidence=0.5, depth=1)
        assert qa_min.depth == 1

        qa_max = QAPair(question="Q", answer="A", mood="happy", confidence=0.5, depth=3)
        assert qa_max.depth == 3

        # Invalid - too low
        with pytest.raises(ValidationError):
            QAPair(question="Q", answer="A", mood="happy", confidence=0.5, depth=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            QAPair(question="Q", answer="A", mood="happy", confidence=0.5, depth=4)

    def test_required_fields(self):
        """Test all fields are required"""
        with pytest.raises(ValidationError):
            QAPair()


class TestAgentSession:
    """Test AgentSession pydantic model"""

    def test_valid_agent_session(self):
        """Test valid AgentSession creation"""
        now = datetime.now()
        qa_pair = QAPair(
            question="How are you?",
            answer="Good",
            mood="content",
            confidence=0.9,
            depth=2,
        )
        session = AgentSession(
            session_id="test-123",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="content",
            final_confidence=0.9,
            final_depth=2,
            question_count=1,
            audio_url="https://example.com/audio.flac",
        )
        assert session.session_id == "test-123"
        assert session.created_at == now
        assert len(session.qa_pairs) == 1
        assert session.qa_pairs[0].mood == "content"
        assert session.final_mood == "content"
        assert session.final_confidence == 0.9
        assert session.final_depth == 2
        assert session.question_count == 1
        assert session.audio_url == "https://example.com/audio.flac"

    def test_final_confidence_boundaries(self):
        """Test final_confidence must be 0-1"""
        now = datetime.now()
        qa_pair = QAPair(
            question="Q", answer="A", mood="happy", confidence=0.5, depth=1
        )

        # Valid boundaries
        session_min = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=0.0,
            final_depth=1,
            question_count=1,
            audio_url="url",
        )
        assert session_min.final_confidence == 0.0

        session_max = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=1.0,
            final_depth=1,
            question_count=1,
            audio_url="url",
        )
        assert session_max.final_confidence == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_mood="happy",
                final_confidence=1.5,
                final_depth=1,
                question_count=1,
                audio_url="url",
            )

    def test_final_depth_boundaries(self):
        """Test final_depth must be 1-3"""
        now = datetime.now()
        qa_pair = QAPair(
            question="Q", answer="A", mood="happy", confidence=0.5, depth=1
        )

        # Valid boundaries
        session_min = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=0.5,
            final_depth=1,
            question_count=1,
            audio_url="url",
        )
        assert session_min.final_depth == 1

        session_max = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=0.5,
            final_depth=3,
            question_count=1,
            audio_url="url",
        )
        assert session_max.final_depth == 3

        # Invalid
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_mood="happy",
                final_confidence=0.5,
                final_depth=4,
                question_count=1,
                audio_url="url",
            )

    def test_question_count_boundaries(self):
        """Test question_count must be 1-5"""
        now = datetime.now()
        qa_pair = QAPair(
            question="Q", answer="A", mood="happy", confidence=0.5, depth=1
        )

        # Valid boundaries
        session_min = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=0.5,
            final_depth=1,
            question_count=1,
            audio_url="url",
        )
        assert session_min.question_count == 1

        session_max = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=0.5,
            final_depth=1,
            question_count=5,
            audio_url="url",
        )
        assert session_max.question_count == 5

        # Invalid - too low
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_mood="happy",
                final_confidence=0.5,
                final_depth=1,
                question_count=0,
                audio_url="url",
            )

        # Invalid - too high
        with pytest.raises(ValidationError):
            AgentSession(
                session_id="test",
                created_at=now,
                qa_pairs=[qa_pair],
                final_mood="happy",
                final_confidence=0.5,
                final_depth=1,
                question_count=6,
                audio_url="url",
            )

    def test_multiple_qa_pairs(self):
        """Test session with multiple Q&A pairs"""
        now = datetime.now()
        qa_pairs = [
            QAPair(question="Q1", answer="A1", mood="happy", confidence=0.5, depth=1),
            QAPair(question="Q2", answer="A2", mood="content", confidence=0.7, depth=2),
            QAPair(question="Q3", answer="A3", mood="joyful", confidence=0.9, depth=3),
        ]
        session = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=qa_pairs,
            final_mood="joyful",
            final_confidence=0.9,
            final_depth=3,
            question_count=3,
            audio_url="url",
        )
        assert len(session.qa_pairs) == 3
        assert session.qa_pairs[0].question == "Q1"
        assert session.qa_pairs[1].mood == "content"
        assert session.qa_pairs[2].depth == 3

    def test_json_serialization(self):
        """Test datetime JSON serialization"""
        now = datetime.now()
        qa_pair = QAPair(
            question="Q", answer="A", mood="happy", confidence=0.5, depth=1
        )
        session = AgentSession(
            session_id="test",
            created_at=now,
            qa_pairs=[qa_pair],
            final_mood="happy",
            final_confidence=0.5,
            final_depth=1,
            question_count=1,
            audio_url="url",
        )
        json_data = session.model_dump()
        assert "created_at" in json_data
        assert isinstance(json_data["created_at"], datetime)

    def test_required_fields(self):
        """Test all fields are required"""
        with pytest.raises(ValidationError):
            AgentSession()
