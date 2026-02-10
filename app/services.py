import io
import os
from datetime import datetime

from fastapi import HTTPException
from pydub import AudioSegment

from app.deps import get_firestore_client, get_storage_client
from app.models import AgentSession, QAEmotionPair


# Background upload of agent session data and audio
def upload_session_in_background(
    audio_bytes: bytearray,
    session_id: str,
    session_timestamp: str,
    qa_pairs_with_emotions: list[QAEmotionPair],
    final_emotion: str,
    final_confidence: float,
    total_question_count: int,
    direct_question_count: int,
):
    try:
        if not audio_bytes:
            print(f"[UPLOAD] No audio data to upload for session: {session_id}")
            return

        # upload audio to bucket
        audio_url = upload_agent_audio_to_bucket(
            bytes(audio_bytes), session_id, session_timestamp
        )

        # create session object
        session = AgentSession(
            session_id=session_id,
            created_at=datetime.fromisoformat(session_timestamp),
            qa_pairs=qa_pairs_with_emotions,
            final_emotion=final_emotion,
            final_confidence=final_confidence,
            total_question_count=total_question_count,
            direct_question_count=direct_question_count,
            audio_url=audio_url,
        )

        # upload session to Firestore
        upload_agent_session(session)
        print(f"[UPLOAD] Background upload completed for session: {session_id}")
    except Exception as e:
        print(f"[UPLOAD] Background upload failed for session {session_id}: {e}")


# Upload agent session to Firestore
def upload_agent_session(session: AgentSession):
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
