import io
import subprocess
import wave

import numpy as np
import pyflac
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
    get_storage_client,
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
    propmt = (
        "Analyze the following transcript and give a normalized (0.0 to 1.0) probability to be each one of the following mood categories:"
        "depressed, unmotivated, tired, anxious, stressed, unfocused, hyperactive, angry, sad, numb, confused, happy, excited, motivated, active, calm, focused, clear headed"
        "Also give a overall confidence score between 0.0 and 1.0 and evidence with explanations."
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

    moods = Mood.model_validate_json(response.text)
    # print(f"Mood analysis step done: {moods}")
    mood = filterMoods(moods)

    return mood


# Function to pick top 50% moods only
def filterMoods(moods: Mood) -> Mood:
    sorted_moods = sorted(moods.mood, key=lambda x: x[1], reverse=True)
    filtered = []
    total = 0.0
    for label, score in sorted_moods:
        if total >= 0.5:
            break
        filtered.append((label, score))
        total += score
    moods.mood = filtered
    return moods


# Upload transcript and mood to firestore
async def uploadToFirestoreStep(transcript: Transcript, mood: Mood):
    if not transcript or not mood:
        raise HTTPException(status_code=400, detail="No response data provided.")

    doc_ref = get_firestore_client().collection("record")
    write_res = doc_ref.document(transcript.uid).set(
        {
            "created_at": firestore.SERVER_TIMESTAMP,
            "mood_confidence": mood.confidence,
            "mood_evidence": mood.evidence,
            "moods": [{"label": label, "score": score} for label, score in mood.mood],
            "transcript": transcript.text,
            "uid": transcript.uid,
        }
    )

    if not write_res.update_time:
        raise HTTPException(status_code=400, detail="Failed to upload to Firestore.")

    # print(f"Upload to Firestore step done: {transcript.uid}")
    return {"status": 200, "uid": transcript.uid}


# convert LINEAR16 audio to WAV
def lin16ToWav(audio_bytes: bytes) -> bytes:
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    try:
        byte_io = io.BytesIO()
        with wave.open(byte_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_array.tobytes())

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert to WAV: {e}")

    return byte_io.getvalue()


# convert WAV to MP3
# https://www.geeksforgeeks.org/python/convert-mp3-to-wav-using-python/
def wavToMp3(wav_bytes: bytes) -> bytes:
    mp3_bytes, _ = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-b:a",
            "192K",
            "-f",
            "mp3",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate(input=wav_bytes)
    return mp3_bytes


# convert LINEAR16 audio to FLAC
# https://pyflac.readthedocs.io/en/latest/#encoder
def lin16ToFlac(audio_bytes: bytes) -> bytes:
    flac_chunks = []

    def write_callback(
        buffer: bytes, num_bytes: int, num_samples: int, current_frame: int
    ):
        flac_chunks.append(buffer)

    encoder = pyflac.StreamEncoder(
        sample_rate=16000,
        write_callback=write_callback,
        compression_level=5,
    )

    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    encoder.process(audio_array)
    encoder.finish()

    return b"".join(flac_chunks)


# Upload audio file to bucket
async def uploadToBucketStep(audio_bytes: bytes, filename: str) -> str:
    if not audio_bytes or not filename:
        raise HTTPException(status_code=400, detail="No audio data provided.")

    # wav_bytes = lin16ToWav(audio_bytes)
    # mp3_bytes = wavToMp3(wav_bytes)
    flac_bytes = lin16ToFlac(audio_bytes)
    bucket = get_storage_client().bucket("speech_wayv_bucket")

    blob = bucket.blob(f"audio/{filename}.flac")
    try:
        blob.upload_from_string(flac_bytes, content_type="audio/flac")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload to Bucket: {e}")

    return f"gs://speech_wayv_bucket/audio/{filename}.flac"
