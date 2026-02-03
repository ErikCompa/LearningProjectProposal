import io
import os

from fastapi import HTTPException
from pydub import AudioSegment

from app.deps import get_firestore_client, get_storage_client
from app.models import AgentSession


# Upload agent session to Firestore
def upload_agent_session(session: AgentSession):
    """Upload complete agent session to Firestore 'sessions' collection"""
    try:
        doc_ref = get_firestore_client().collection("sessions")
        write_res = doc_ref.document(session.session_id).set(session.model_dump())

        if not write_res.update_time:
            raise HTTPException(
                status_code=400, detail="Failed to upload session to Firestore."
            )

        print(f"[FIRESTORE] Uploaded session: {session.session_id}")
        return {"status": 200, "session_id": session.session_id}
    except Exception as e:
        print(f"[FIRESTORE] Error uploading session: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to upload to Firestore: {e}"
        )


# convert LINEAR16 audio to FLAC
def linear_16_to_flac(audio_bytes: bytes) -> bytes:
    audio = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=16000, channels=1)
    out_io = io.BytesIO()
    audio.export(out_io, format="flac")
    return out_io.getvalue()


# Upload audio file to bucket for agent session
def upload_agent_audio_to_bucket(
    audio_bytes: bytes, session_id: str, timestamp: str
) -> str:
    """Upload agent session audio to GCS bucket as FLAC"""
    if not audio_bytes or not session_id:
        raise HTTPException(status_code=400, detail="No audio data provided.")

    flac_bytes = linear_16_to_flac(audio_bytes)
    bucket = get_storage_client().bucket(os.getenv("BUCKET_NAME"))

    filename = f"{session_id}_{timestamp}"
    blob_flac = bucket.blob(f"audio/agent/{filename}.flac")

    try:
        blob_flac.upload_from_string(flac_bytes, content_type="audio/flac")
        print(f"[BUCKET] Uploaded audio: {filename}.flac")
    except Exception as e:
        print(f"[BUCKET] Error uploading audio: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to upload to Bucket: {e}")

    return f"{os.getenv('BUCKET_URL')}agent/{filename}.flac"
