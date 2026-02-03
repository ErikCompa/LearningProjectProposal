import asyncio
import base64
import os
import uuid

from elevenlabs import RealtimeAudioOptions, RealtimeEvents
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app.agent import analyze_mood, get_next_question
from app.deps import get_elevenlabs
from app.models import Transcript

router = APIRouter(tags=["agent"])


async def stt_elevenlabs(
    audio_queue, res_queue, stop, answer_transcript_container, answer_ready
):
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
        text = data.get("text", "")
        answer_transcript_container["current"] += text
        res_queue.put_nowait(
            {
                "transcript": text,
                "is_final": True,
            }
        )
        # Signal that answer is ready (VAD detected end of speech)
        answer_ready.set()

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


async def receive_audio(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    audioBytes: bytearray,
    res_queue: asyncio.Queue,
):
    while True:
        # get audio data
        data = await websocket.receive_bytes()
        audio_queue.put_nowait(data)
        audioBytes.extend(data)
        # Check for responses and send to client
        while not res_queue.empty():
            res = await res_queue.get()
            await websocket.send_json(res)


@router.websocket(os.getenv("AGENT_URL"))
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()

    session_id = uuid.uuid4()
    audio_queue = asyncio.Queue()
    res_queue = asyncio.Queue()
    stop = asyncio.Event()
    audioBytes = bytearray()
    answer_transcript_container = {"current": ""}
    answer_ready = asyncio.Event()

    stt_task = asyncio.create_task(
        stt_elevenlabs(
            audio_queue, res_queue, stop, answer_transcript_container, answer_ready
        )
    )

    try:
        mood_confidence = 0.0
        question_counter = 0
        # [question, answer]
        qa_pairs: list[tuple[str, str]] = []
        # [mood, confidence]
        moods: list[tuple[str, float]] = []

        # Start receiving audio
        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audioBytes, res_queue)
        )

        while mood_confidence < 0.9 and question_counter < 5:
            # get question from agent
            if question_counter == 0:
                question = "Hello! How are you feeling today?"
            else:
                question = await get_next_question(qa_pairs, moods)

            # ask question
            print(question)

            # Reset current answer and event
            answer_transcript_container["current"] = ""
            answer_ready.clear()

            # Wait for VAD to detect end of speech
            await answer_ready.wait()

            answer_transcript = answer_transcript_container["current"]

            # analyze response
            mood, mood_confidence = await analyze_mood(
                qa_pairs, moods, question, answer_transcript
            )

            # go to next question
            qa_pairs.append((question, answer_transcript))
            moods.append((mood, mood_confidence))
            question_counter += 1

        if mood_confidence >= 0.9:
            print("Mood detected")
        else:
            print("Max questions reached. Shutting down.")

    except WebSocketDisconnect as e:
        print(f"[STREAMING] Websocket disconnected: {e}")
    except Exception as e:
        print(f"[STREAMING] error during websocket communication: {e}")
    finally:
        # cleanup
        receive_task.cancel()
        stop.set()
        await audio_queue.put(None)
        await stt_task

        # TODO upload session_id, full audio, qa_pairs, moods
