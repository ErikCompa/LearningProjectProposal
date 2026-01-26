from fastapi import (
    HTTPException,
    UploadFile,
)
from google.cloud import firestore
from google.cloud.speech_v2.types import cloud_speech

from app.models import Mood, Transcript
from app.speech_config import get_batch_recognition_config

from .deps import (
    get_firestore_client,
    get_gemini_client,
    get_project_id,
    get_speech_v2_client,
)


# Get all from firestore endpoint
async def get_from_firestore():
    rows = get_firestore_client().collection("record").stream()
    records = []
    for row in rows:
        records.append(row.to_dict())
    return records


# Send entire audio file for batch transcription
async def batchTranscriptionStep(file: UploadFile) -> tuple[Transcript, bytes]:
    if file.filename is None or file.filename == "":
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.size is None or file.size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # raw audio bytes in data for mp3(?) conversion later and saving to bucket
    data = await file.read()

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{get_project_id()}/locations/us/recognizers/mood",
        config=get_batch_recognition_config(),
        content=data,
    )

    # gRPC call
    response = get_speech_v2_client().recognize(request=request)

    transcript = Transcript(
        text=response.results[0].alternatives[0].transcript,
    )
    # print(f"Transcript step done: {transcript}")
    return transcript, data


# Send transcript to gemini for mood analysis
async def moodAnalysisStep(transcript: Transcript) -> Mood:
    if not transcript.text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    propmt = (
        "Analyze the following transcript and determine the overall mood of the user. "
        "Give a confidence score between 0.0 and 1.0 and evidence with explanations."
    )

    transcript_data = transcript.model_dump()

    response = get_gemini_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=[propmt, str(transcript_data)],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Mood.model_json_schema(),
        },
    )

    if not response.text:
        raise HTTPException(status_code=400, detail="Mood analysis failed.")

    mood = Mood.model_validate_json(response.text)
    # print(f"Mood analysis step done: {mood}")
    return mood


# Upload transcript and mood to firestore
async def uploadToFirestoreStep(transcript: Transcript, mood: Mood):
    if not transcript or not mood:
        raise HTTPException(status_code=400, detail="No response data provided.")

    doc_ref = get_firestore_client().collection("record")
    write_res = doc_ref.document(transcript.uid).set(
        {
            "created_at": firestore.SERVER_TIMESTAMP,
            "mood": {
                "confidence": mood.confidence,
                "evidence": mood.evidence,
                "mood": mood.mood,
            },
            "transcript": transcript.text,
            "uid": transcript.uid,
        }
    )

    if not write_res.update_time:
        raise HTTPException(status_code=400, detail="Failed to upload to Firestore.")

    # print(f"Upload to Firestore step done: {transcript.uid}")
    return {"status": 200, "uid": transcript.uid}
