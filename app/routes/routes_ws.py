import asyncio
import base64
import os
import threading
import time
from queue import Queue
from threading import Event

from elevenlabs import RealtimeAudioOptions, RealtimeEvents
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from google.cloud.speech_v2.types import cloud_speech  # type: ignore

from app.deps import get_elevenlabs, get_speech_v2_client
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
        print("[STREAMING] final transcript processing done:", res)
    except Exception as e:
        print("[STREAMING] error during final transcript processing:", e)


async def stt_elevenlabs(audio_queue, res_queue, stop):
    print("[STREAMING] Starting ElevenLabs STT connection")
    elevenlabs = get_elevenlabs()
    connection = await elevenlabs.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=16000,
            include_timestamps=True,
            commit_strategy=CommitStrategy.VAD,
            vad_silence_threshold_secs=1.5,
            vad_threshold=0.4,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
        )
    )

    def on_session_started(data):
        print(f"[STREAMING] ElevenLabs session started: {data}")

    def on_partial_transcript(data):
        res_queue.put_nowait(
            {
                "transcript": data.get("text", ""),
                "is_final": False,
            }
        )

    def on_committed_transcript(data):
        res_queue.put_nowait(
            {
                "transcript": data.get("text", ""),
                "is_final": True,
            }
        )

    def on_error(error):
        print(f"[STREAMING] ElevenLabs error: {error}")
        stop.set()

    def on_close():
        print("[STREAMING] ElevenLabs connection closed")
        stop.set()

    connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial_transcript)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed_transcript)
    connection.on(RealtimeEvents.ERROR, on_error)
    connection.on(RealtimeEvents.CLOSE, on_close)

    async def send_audio():
        while not stop.is_set():
            chunk = await audio_queue.get()
            if chunk is None:
                break
            audio_base64 = base64.b64encode(chunk).decode("utf-8")
            await connection.send({"audio_base_64": audio_base64})

    sender = asyncio.create_task(send_audio())
    try:
        await stop.wait()
    finally:
        await connection.close()
        sender.cancel()


# WebSocket for realtime audio transcription
@router.websocket(os.getenv("STREAM_PROCESS_AUDIO_URL"))
async def websocket_stream_process_audio(websocket: WebSocket):
    def stt_thread():
        print("[STREAMING] Starting Google STT thread")
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

    await websocket.accept()

    config_data = await websocket.receive_json()
    provider = config_data.get("provider", "google")

    print(f"[STREAMING] Received config: {config_data}")
    print(f"[STREAMING] Provider value: {provider}")

    if provider not in ["google", "elevenlabs"]:
        await websocket.close(code=1003)
        return

    if provider == "elevenlabs":
        print("[STREAMING] Using ElevenLabs STT provider")
        audio_queue = asyncio.Queue()
        res_queue = asyncio.Queue()
        stop = asyncio.Event()
        stt_task = asyncio.create_task(stt_elevenlabs(audio_queue, res_queue, stop))
    else:
        print("[STREAMING] Using Google STT provider")
        audio_queue = Queue()
        res_queue = Queue()
        stop = Event()
        threading.Thread(target=stt_thread, daemon=True).start()

    audioBytes = bytearray()
    full_transcript = ""

    try:
        while True:
            data = await websocket.receive_bytes()

            if provider == "elevenlabs":
                # put audio data to audio q continously
                audio_queue.put_nowait(data)
                audioBytes.extend(data)
            else:
                MAX_CHUNK_SIZE = 25600
                for i in range(0, len(data), MAX_CHUNK_SIZE):
                    chunk = data[i : i + MAX_CHUNK_SIZE]
                    audio_queue.put(chunk)
                    audioBytes.extend(chunk)

            # Check for responses
            if provider == "elevenlabs":
                while not res_queue.empty():
                    res = await res_queue.get()
                    if res["is_final"]:
                        full_transcript += res["transcript"] + ". "
                    await websocket.send_json(res)
            else:
                while not res_queue.empty():
                    res = res_queue.get()
                    if res["is_final"]:
                        full_transcript += res["transcript"] + ". "
                    await websocket.send_json(res)
    except WebSocketDisconnect as e:
        print(f"[STREAMING] Websocket disconnected: {e}")
    except Exception as e:
        print(f"[STREAMING] error during websocket communication: {e}")
    finally:
        stop.set()
        if provider == "elevenlabs":
            await audio_queue.put(None)
            await stt_task
        else:
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
            print("[STREAMING] No transcript to process, skipping final processing.")
