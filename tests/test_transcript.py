import pytest
from pydantic import ValidationError

from app.models import Transcript


def test_transcript_model_valid_data():
    data = {"text": "This is a test transcript.", "confidence": 0.85}
    transcript = Transcript(**data)
    assert transcript.text == data["text"]
    assert transcript.confidence == round(data["confidence"], 2)


def test_transcript_model_invalid_data():
    data = {
        "text": "",
        "confidence": "high",
    }
    with pytest.raises(ValidationError) as exc_info:
        Transcript(**data)
    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("text",) and error["type"] == "string_too_short"
        for error in errors
    )
    assert any(
        error["loc"] == ("confidence",) and error["type"] == "float_parsing"
        for error in errors
    )


def test_confidence_must_be_float():
    data = {"text": "Another test transcript.", "confidence": "not_a_float"}
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        Transcript(**data)


def test_confidence_rounding():
    data = {"text": "Rounding test transcript.", "confidence": 0.8567}
    transcript = Transcript(**data)
    assert transcript.confidence == 0.86
