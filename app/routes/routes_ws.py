import asyncio
import os
import threading
import time
from queue import Queue
from threading import Event

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from google.cloud.speech_v2.types import cloud_speech  # type: ignore

from app.deps import get_speech_v2_client
from app.models import Transcript
from app.services import (
    mood_analysis_step,
    upload_to_bucket_step,
    upload_to_firestore_step,
)
from app.speech_config import get_streaming_config_request

router = APIRouter(tags=["ws"])


def process_final_transcript(transcript: Transcript, audioBytes: bytes):
    try:
        mood = mood_analysis_step(transcript)
        res = upload_to_firestore_step(transcript, mood)
        if res["uid"]:
            upload_to_bucket_step(audioBytes, res["uid"])
        print("final transcript processing done:", res)
    except Exception as e:
        print("error during final transcript processing:", e)


# WebSocket for realtime audio transcription
@router.websocket(os.getenv("STREAM_PROCESS_AUDIO_URL"))
async def websocket_stream_process_audio(websocket: WebSocket):
    def stt_thread():
        try:
            while not stop.is_set():
                start = time.time()

                def requests():
                    yield get_streaming_config_request()
                    while not stop.is_set():
                        if time.time() - start > 240:
                            print("4 min limit reached, breaking request generator.")
                            break
                        chunk = audio_queue.get()
                        if chunk is None:
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

                try:
                    responses: cloud_speech.StreamingRecognizeResponse = (
                        get_speech_v2_client().streaming_recognize(requests=requests())
                    )

                    for response in responses:
                        for result in response.results:
                            transcript_text = result.alternatives[0].transcript
                            res_queue.put(
                                {
                                    "transcript": transcript_text,
                                    "is_final": result.is_final,
                                    "stability": result.stability,
                                }
                            )
                except Exception as e:
                    print(f"Exception in STT thread: {e}")
                break
        except Exception as e:
            print(f"Fatal error in STT thread: {e}")

    audio_queue = Queue()
    audioBytes = bytearray()
    res_queue = Queue()
    stop = Event()
    full_transcript = ""

    await websocket.accept()

    threading.Thread(target=stt_thread, daemon=True).start()

    try:
        while True:
            data = await websocket.receive_bytes()

            MAX_CHUNK_SIZE = 25600
            for i in range(0, len(data), MAX_CHUNK_SIZE):
                chunk = data[i : i + MAX_CHUNK_SIZE]
                audio_queue.put(chunk)
                audioBytes.extend(chunk)

            while not res_queue.empty():
                res = res_queue.get()
                if res["is_final"]:
                    full_transcript += res["transcript"] + ". "
                await websocket.send_json(res)
    except WebSocketDisconnect as e:
        print(f"websocket disconnected: {e}")
    except Exception as e:
        print(f"error during websocket communication: {e}")
    finally:
        stop.set()
        audio_queue.put(None)

        # make final transcript. Only process if not empty
        if full_transcript.strip():
            transcript = Transcript(
                text=full_transcript,
            )
            # offload final processing to background task
            asyncio.create_task(
                asyncio.to_thread(process_final_transcript, transcript, audioBytes)
            )
        else:
            print("No transcript to process, skipping final processing.")
