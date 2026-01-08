from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.auth import default
from google.cloud import firestore
from google.cloud import speech_v1 as speech

from .models import Mood, Transcript

# clients startup
credentials, project = default()
gemini_client = genai.Client(
    vertexai=True,  # vertex for ADC so there are no keys
    project=project,
    location="us-central1",
    credentials=credentials,
)

app = FastAPI()

speech_client = speech.SpeechClient()

db = firestore.Client()


# CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# single endpoint to transcribe, analyze mood, and upload to firestore
@app.post("/v1/process_audio/")
async def process_audio(file: Annotated[UploadFile, File(...)]):
    # Transcription step
    # file check
    if file.content_type != "audio/webm":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only audio/webm is supported."
        )

    if file.filename is None or file.filename == "":
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.size is None or file.size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # read file bytes
    data = await file.read()
    sample_rate = 48000

    if not data:
        raise HTTPException(status_code=400, detail="Failed to read file.")

    # send bytes to google stt
    audio = speech.RecognitionAudio(content=data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
    )
    response = speech_client.recognize(config=config, audio=audio)

    if not response.results:
        raise HTTPException(status_code=400, detail="Transcription failed.")

    # transcript in pydantic model
    transcript = Transcript(
        text=response.results[0].alternatives[0].transcript,
        confidence=response.results[0].alternatives[0].confidence,
    )
    print(f"Transcript step done: {transcript}")

    # Mood analysis step
    if not transcript.text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    propmt = "Analyze the following transcript and determine the overall mood of the user. Give a confidence score between 0.0 and 1.0 and evidence with explanations."

    # remove confidence to force gemini to gen one
    transcript_data = transcript.model_dump(exclude={"confidence"})

    response = gemini_client.models.generate_content(
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
    print(f"Mood analysis step done: {mood}")

    # Upload to Firestore step
    if not transcript or not mood:
        raise HTTPException(status_code=400, detail="No response data provided.")

    # insert response into firestore
    doc_ref = db.collection("record")
    write_res = doc_ref.document(transcript.uid).set(
        {
            "created_at": firestore.SERVER_TIMESTAMP,
            "mood": {
                "confidence": mood.confidence,
                "evidence": mood.evidence,
                "mood": mood.mood,
            },
            "transcript": transcript.text,
            "transcript_confidence": transcript.confidence,
            "uid": transcript.uid,
        }
    )

    if not write_res.update_time:
        raise HTTPException(status_code=400, detail="Failed to upload to Firestore.")

    return {"status": 200, "uid": transcript.uid}


# get all from firestore endpoint
@app.get("/v1/firestore_get/")
async def get_from_firestore():
    rows = db.collection("record").stream()
    records = []
    for row in rows:
        records.append(row.to_dict())
    return records


# Mount static files
# https://fastapi.tiangolo.com/tutorial/static-files/
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
