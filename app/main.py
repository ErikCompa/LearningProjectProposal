from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .models import Mood, Transcript, Response
from google.cloud import speech_v1 as speech
from google import genai
from google.auth import default

credentials, project = default()
gemini_client = genai.Client(
    vertexai=True,
    project=project,
    location="us-central1",  # or your preferred region
    credentials=credentials
)

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

# https://www.youtube.com/watch?v=n43Td-mU7oA
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

# https://www.youtube.com/watch?v=qfWpPEgea2A
# https://ai.google.dev/gemini-api/docs/structured-output?example=recipe
@app.post("/v1/analyze_mood/")
async def analyze(transcript: Transcript):
    propmt = "Analyze the mood of the following transcript:"

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

    mood = Mood.model_validate_json(response.text)
    return mood