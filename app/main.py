from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .models import Mood, Transcript, Response
from google.cloud import speech_v1 as speech

app = FastAPI()
speech_client = speech.SpeechClient()

# CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/v1/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # file check
    if (file.content_type != "audio/webm"):
        return Response(
            success=False,
            error="Invalid file type. Only audio/webm is supported."
        )
    
    if (file.filename is None or file.filename == ""):
        return Response(
            success=False,
            error="No file uploaded."
        )
    
    if (file.size is None or file.size == 0):
        return Response(
            success=False,
            error="Empty file uploaded."
        )
    
    # read file bytes
    data = await file.read()
    sample_rate = 48000

    # send bytes to google stt
    audio = speech.RecognitionAudio(content=data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
    )
    response = speech_client.recognize(config=config, audio=audio)

    # return transcript in pydantic model
    return Transcript(
        text=response.results[0].alternatives[0].transcript,
        confidence=response.results[0].alternatives[0].confidence
    )

@app.post("/v1/analyze_mood/")
async def analyze():
    return "Hi"