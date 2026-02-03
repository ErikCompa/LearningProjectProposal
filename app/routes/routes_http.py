import os
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    UploadFile,
)

from app.models import Transcript
from app.services import (
    batch_transcription_step,
    elevenlabs_batch_transcription_step,
    get_from_firestore,
    mood_analysis_step,
    upload_to_bucket_step,
    upload_to_firestore_step,
)

router = APIRouter(tags=["http"])


# Process final transcript in background
def process_final_transcript(transcript: Transcript, audioBytes: bytes):
    try:
        mood = mood_analysis_step(transcript)
        res = upload_to_firestore_step(transcript, mood)
        if res["uid"]:
            upload_to_bucket_step(audioBytes, res["uid"])
        print("[BATCH] final transcript processing done:", res)
    except Exception as e:
        print("[BATCH] error during final transcript processing:", e)


# POST batch audio processing endpoint
@router.post(os.getenv("BATCH_PROCESS_AUDIO_URL"))
async def batch_process_audio(
    file: Annotated[UploadFile, File(...)],
    background_tasks: BackgroundTasks,
    provider: Literal["google", "elevenlabs"] = "google",
):
    if provider == "elevenlabs":
        print("[BATCH] Using ElevenLabs STT provider")
        transcript, data = await elevenlabs_batch_transcription_step(file)
    else:
        print("[BATCH] Using Google STT provider")
        transcript, data = await batch_transcription_step(file)
    background_tasks.add_task(process_final_transcript, transcript, data)


# GET records from firestore endpoint
@router.get(os.getenv("FIRESTORE_GET_URL"))
async def firestore_get_records():
    records = await get_from_firestore()
    return records
