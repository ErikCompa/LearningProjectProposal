import asyncio
import base64
import os

from elevenlabs import RealtimeAudioOptions, RealtimeEvents
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.deps import get_elevenlabs
from app.models import Transcript
from app.services import moodAnalysisStep, uploadToBucketStep, uploadToFirestoreStep

router = APIRouter(tags=["ws"])


async def stt_elevenlabs(audio_queue, res_queue, stop):
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
        print(f"Session started: {data}")

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
        print(f"Error: {error}")
        stop.set()

    def on_close():
        print("Connection closed")
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
    audio_queue = asyncio.Queue()
    res_queue = asyncio.Queue()
    stop = asyncio.Event()
    audioBytes = bytearray()
    full_transcript = ""

    # initialization
    await websocket.accept()
    stt_task = asyncio.create_task(stt_elevenlabs(audio_queue, res_queue, stop))

    try:
        while True:
            # receive audio data from websocket
            data = await websocket.receive_bytes()

            # put audio data to audio q continously
            audio_queue.put_nowait(data)
            audioBytes.extend(data)

            # read from results q
            while not res_queue.empty():
                res = await res_queue.get()
                if res["is_final"]:
                    full_transcript += res["transcript"]
                # send result back to client for realtime display
                await websocket.send_json(res)
    except WebSocketDisconnect as e:
        print("websocket disconnected:", e)
    except Exception as e:
        print("error during websocket communication:", e)
    finally:
        # cleanup
        stop.set()
        await audio_queue.put(None)
        await stt_task

        # process final transcript
        transcript = Transcript(
            text=full_transcript,
        )
        mood = await moodAnalysisStep(transcript)
        res = await uploadToFirestoreStep(transcript, mood)
        if res["uid"]:
            await uploadToBucketStep(audioBytes, res["uid"])
    return res
