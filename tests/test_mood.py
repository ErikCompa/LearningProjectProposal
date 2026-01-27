import pytest
from pydantic import ValidationError

from app.models import Mood


def test_mood_model_valid_data():
    data = {
        "mood": [("happy", 0.92)],
        "confidence": 0.92,
        "evidence": ["The user laughed", "Positive tone detected"],
    }
    mood = Mood(**data)
    assert mood.mood == [("happy", 1.0)]
    assert mood.confidence == round(data["confidence"], 2)
    assert mood.evidence == data["evidence"]


def test_mood_model_invalid_data():
    data = {
        "mood": [("", 0.0)],
        "confidence": "high",
        "evidence": "Not a list",
    }
    with pytest.raises(ValidationError) as exc_info:
        Mood(**data)
    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("mood",)
        and "Sum of mood scores cannot be zero" in error["msg"]
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
        "mood": [("sad", 0.5)],
        "confidence": "not_a_float",
        "evidence": ["The user sighed"],
    }
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        Mood(**data)


def test_confidence_rounding():
    data = {
        "mood": [("angry", 0.8)],
        "confidence": 0.8765,
        "evidence": ["The user raised their voice"],
    }
    mood = Mood(**data)
    assert mood.confidence == 0.88
