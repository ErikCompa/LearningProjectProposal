import pytest
from pydantic import ValidationError

from app.models import Mood


def test_mood_model_valid_data():
    data = {
        "mood": [{"label": "happy", "score": 0.92}],
        "confidence": 0.92,
        "evidence": [{"label": "happy", "explanation": "The user laughed"}],
    }
    mood = Mood(**data)
    assert len(mood.mood) == 1
    assert mood.mood[0].label == "happy"
    assert mood.mood[0].score == 1.0
    assert mood.confidence == 0.92
    assert len(mood.evidence) == 1
    assert mood.evidence[0].label == "happy"


def test_mood_model_invalid_data():
    data = {
        "mood": [{"label": "", "score": 0.0}],
        "confidence": "high",
        "evidence": "Not a list",
    }
    with pytest.raises(ValidationError) as exc_info:
        Mood(**data)
    errors = exc_info.value.errors()
    assert any(
        "mood" in error["loc"]
        and "Sum of mood scores cannot be zero" in str(error["msg"])
        for error in errors
    )
    assert any(
        error["loc"] == ("confidence",) and error["type"] == "float_parsing"
        for error in errors
    )
    assert any(
        error["loc"] == ("evidence",) and error["type"] == "list_type"
        for error in errors
    )


def test_confidence_must_be_float():
    data = {
        "mood": [{"label": "sad", "score": 0.5}],
        "confidence": "not_a_float",
        "evidence": [{"label": "sad", "explanation": "The user sighed"}],
    }
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        Mood(**data)


def test_confidence_rounding():
    data = {
        "mood": [{"label": "angry", "score": 0.8}],
        "confidence": 0.8765,
        "evidence": [{"label": "angry", "explanation": "The user raised their voice"}],
    }
    mood = Mood(**data)
    assert mood.confidence == 0.88
