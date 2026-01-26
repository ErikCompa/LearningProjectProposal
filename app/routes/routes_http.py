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
    uploadToFirestoreStep,
)

router = APIRouter(tags=["http"])


# POST batch audio processing endpoint
@router.post("/v1/batch_process_audio/")
async def batch_process_audio(file: Annotated[UploadFile, File(...)]):
    transcript, data = await batchTranscriptionStep(file)
    mood = await moodAnalysisStep(transcript)
    uploadResult = await uploadToFirestoreStep(transcript, mood)
    return uploadResult


# GET records from firestore endpoint
@router.get("/v1/firestore_get/")
async def firestore_get_records():
    records = await get_from_firestore()
    return records
