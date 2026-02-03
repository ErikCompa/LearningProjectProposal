import io
import os

from fastapi import (
    HTTPException,
    UploadFile,
)
from google.cloud import firestore
from pydub import AudioSegment

from app.deps import (
    get_firestore_client,
    get_gemini_client,
    get_speech_v2_client,
    get_storage_client,
)
from app.models import Mood, Transcript
from app.speech_config import get_batch_recognition_request


# Get all from firestore endpoint
async def get_from_firestore():
    rows = get_firestore_client().collection("record").stream()
    records = []
    for row in rows:
        records.append(row.to_dict())
    return records


# Send entire audio file for batch transcription
async def batch_transcription_step(file: UploadFile) -> tuple[Transcript, bytes]:
    if file.filename is None or file.filename == "":
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.size is None or file.size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # raw audio bytes in data for mp3(?) conversion later and saving to bucket
    data = await file.read()

    request = get_batch_recognition_request(data)

    # gRPC call
    response = get_speech_v2_client().recognize(request=request)

    transcript = Transcript(
        text=response.results[0].alternatives[0].transcript,
    )
    # print(f"Transcript step done: {transcript}")
    return transcript, data


# Send transcript to gemini for mood analysis
def mood_analysis_step(transcript: Transcript) -> Mood:
    prompt = """
        You are an emotional analysis agent for a mental-wellbeing music application.

        Your task is: 
        Analyze the user's spoken transcript and give a probability with evidence for each of the following mood categories.
        Normalize probabilities so they add up to 1.0.
        Also give an overall confidence score.

        IMPORTANT CONSTRAINTS:
        - You MUST output STRICT JSON only.
        - Do NOT invent new emotion labels.
        
        MOOD CATEGORIES:
        depressed
        insomnia
        unmotivated
        tired
        anxious
        stressed
        unfocused
        hyperactive
        angry
        sad
        numb
        confused
        happy
        excited
        motivated
        active
        calm
        focused
        clear headed
                
        CONFIDENCE DEFINITION:
        - Confidence is a number between 0 and 1.
        - It represents how confident you are that the chosen moods and scores are appropriate and helpful for the user.
        - This is NOT a probability and does NOT need to sum to anything.
        - Higher confidence means clearer emotional signal and clearer intervention fit.
        
        USER TRANSCRIPT:
        {transcript_text}
    """

    prompt_filled = prompt.format(transcript_text=transcript.text)

    response = get_gemini_client().models.generate_content(
        model="gemini-3-pro-preview",
        contents=[prompt_filled],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Mood.model_json_schema(),
        },
    )

    if not response.text:
        raise HTTPException(status_code=400, detail="Mood analysis failed.")

    moods = Mood.model_validate_json(response.text)
    # print(f"Mood analysis step done: {moods}")

    return moods


# Upload transcript and mood to firestore
def upload_to_firestore_step(transcript: Transcript, mood: Mood):
    if not transcript or not mood:
        raise HTTPException(status_code=400, detail="No response data provided.")

    doc_ref = get_firestore_client().collection("record")
    write_res = doc_ref.document(transcript.uid).set(
        {
            "created_at": firestore.SERVER_TIMESTAMP,
            "mood_confidence": mood.confidence,
            "mood_evidence": [ev.model_dump() for ev in mood.evidence],
            "moods": [entry.model_dump() for entry in mood.mood],
            "transcript": transcript.text,
            "uid": transcript.uid,
            "top_50_moods": [entry.model_dump() for entry in mood.top_50_moods],
            "top_50_evidences": [entry.model_dump() for entry in mood.top_50_evidences],
        }
    )

    if not write_res.update_time:
        raise HTTPException(status_code=400, detail="Failed to upload to Firestore.")

    # print(f"Upload to Firestore step done: {transcript.uid}")
    return {"status": 200, "uid": transcript.uid}


# convert LINEAR16 audio to WAV
def linear16_to_wav(audio_bytes: bytes) -> bytes:
    audio = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=16000,
        channels=1,
    )
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")
    return out_io.getvalue()


# convert WAV to MP3
def wav_to_mp3(wav_bytes: bytes) -> bytes:
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    out_io = io.BytesIO()
    audio.export(out_io, format="mp3", bitrate="192k")
    return out_io.getvalue()


# convert LINEAR16 audio to FLAC
def linear_16_to_flac(audio_bytes: bytes) -> bytes:
    audio = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=16000, channels=1)
    out_io = io.BytesIO()
    audio.export(out_io, format="flac")
    return out_io.getvalue()


# Upload audio file to bucket
def upload_to_bucket_step(audio_bytes: bytes, filename: str) -> str:
    if not audio_bytes or not filename:
        raise HTTPException(status_code=400, detail="No audio data provided.")

    # wav_bytes = linear16_to_wav(audio_bytes)
    # mp3_bytes = wav_to_mp3(wav_bytes)
    flac_bytes = linear_16_to_flac(audio_bytes)
    bucket = get_storage_client().bucket(os.getenv("BUCKET_NAME"))

    blob_flac = bucket.blob(f"audio/{filename}.flac")
    # blob_mp3 = bucket.blob(f"audio/{filename}.mp3")
    # blob_wav = bucket.blob(f"audio/{filename}.wav")
    try:
        # blob_wav.upload_from_string(wav_bytes, content_type="audio/wav")
        # blob_mp3.upload_from_string(mp3_bytes, content_type="audio/mpeg")
        blob_flac.upload_from_string(flac_bytes, content_type="audio/flac")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload to Bucket: {e}")

    return f"{os.getenv('BUCKET_URL')}{filename}.flac"
