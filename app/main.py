from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .models import Mood, Transcript
from google.cloud import speech_v1 as speech
from google import genai
from google.auth import default
from google.cloud import firestore
import os
from pathlib import Path

# clients startup
credentials, project = default()
gemini_client = genai.Client(
    vertexai=True, # vertex for ADC so there are no keys
    project=project,
    location="us-central1",
    credentials=credentials
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
    ],    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# speech to text endpoint
# https://www.youtube.com/watch?v=n43Td-mU7oA
@app.post("/v1/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # file check
    if (file.content_type != "audio/webm"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio/webm is supported.")
        
    if (file.filename is None or file.filename == ""):
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    if (file.size is None or file.size == 0):
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

    # return transcript in pydantic model
    return Transcript(
        text=response.results[0].alternatives[0].transcript,
        confidence=response.results[0].alternatives[0].confidence
    )

# gemini mood analysis endpoint
# https://www.youtube.com/watch?v=qfWpPEgea2A
# https://ai.google.dev/gemini-api/docs/structured-output?example=recipe
@app.post("/v1/analyze_mood/")
async def analyze(transcript: Transcript):
    if not transcript.text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    propmt = "Analyze the following transcript and determine the overall mood of the user. Give a confidence score between 0.0 and 1.0 and evidence with explanations."

    # remove confidence to force gemini to gen one
    transcript_data = transcript.model_dump(exclude={'confidence'})

    response = gemini_client.models.generate_content(
        model = "gemini-2.5-flash",
        contents=[propmt, str(transcript_data)],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Mood.model_json_schema(),
        },
    )

    if not response.text:
        raise HTTPException(status_code=400, detail="Mood analysis failed.")

    mood = Mood.model_validate_json(response.text)
    return mood

# upload to firestore endpoint
@app.post("/v1/firestore_upload/")
async def upload_to_firestore(transcript: Transcript, mood: Mood):
    if not transcript or not mood:
        raise HTTPException(status_code=400, detail="No response data provided.")

    # insert response into firestore
    doc_ref = db.collection(u'record')
    write_res = doc_ref.document(transcript.uid).set({
        u'created_at': firestore.SERVER_TIMESTAMP,
        u'mood': {
            u'confidence': mood.confidence,
            u'evidence': mood.evidence,
            u'mood': mood.mood,
        },
        u'transcript': transcript.text,
        u'transcript_confidence': transcript.confidence,
        u'uid': transcript.uid,
    })

    if not write_res.update_time:
        raise HTTPException(status_code=400, detail="Failed to upload to Firestore.")

    return {"success": True, "uid": transcript.uid}

# get all from firestore endpoint
@app.get("/v1/firestore_get/")
async def get_from_firestore():
    rows = db.collection(u'record').stream()
    records = []
    for row in rows:
        records.append(row.to_dict())
    return records

# Mount static files
# https://fastapi.tiangolo.com/tutorial/static-files/
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")