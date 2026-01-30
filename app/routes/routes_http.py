import os
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from app.services import (
    batchTranscriptionStep,
    get_from_firestore,
    moodAnalysisStep,
    uploadToBucketStep,
    uploadToFirestoreStep,
)

router = APIRouter(tags=["http"])


# POST batch audio processing endpoint
@router.post(os.getenv("BATCH_PROCESS_AUDIO_URL"))
async def batch_process_audio(file: Annotated[UploadFile, File(...)]):
    transcript, data = await batchTranscriptionStep(file)
    mood = await moodAnalysisStep(transcript)
    res = await uploadToFirestoreStep(transcript, mood)
    if res["uid"]:
        await uploadToBucketStep(data, res["uid"])
    return res


# GET records from firestore endpoint
@router.get(os.getenv("FIRESTORE_GET_URL"))
async def firestore_get_records():
    records = await get_from_firestore()
    return records
