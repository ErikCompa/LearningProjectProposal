import asyncio
import base64
import os

# import uuid
from elevenlabs import RealtimeAudioOptions, RealtimeEvents
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app.agent import analyze_mood, get_next_question
from app.deps import get_elevenlabs
from app.wheel_of_emotions import get_emotion_depth, get_wheel_of_emotions

router = APIRouter(tags=["agent"])


async def stt_elevenlabs_session(
    audio_queue, res_queue, answer_transcript_container, answer_ready
):
    """Start a single ElevenLabs STT session for one Q&A cycle"""
    print("[STT] Starting ElevenLabs STT session")
    elevenlabs = get_elevenlabs()
    stop = asyncio.Event()

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
        print(f"[STT] Session started: {data.get('session_id', 'unknown')}")

    def on_partial_transcript(data):
        res_queue.put_nowait(
            {
                "type": "transcript",
                "transcript": data.get("text", ""),
                "is_final": False,
            }
        )

    def on_committed_transcript(data):
        text = data.get("text", "")
        answer_transcript_container["current"] += text
        res_queue.put_nowait(
            {
                "type": "transcript",
                "transcript": text,
                "is_final": True,
            }
        )
        # Signal that answer is ready (VAD detected end of speech)
        print(f"[STT] VAD detected silence, answer complete: {text}")
        answer_ready.set()

    def on_error(error):
        print(f"[STT] Error: {error}")
        stop.set()

    def on_close():
        print("[STT] Connection closed by server")
        stop.set()

    connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial_transcript)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed_transcript)
    connection.on(RealtimeEvents.ERROR, on_error)
    connection.on(RealtimeEvents.CLOSE, on_close)

    async def send_audio():
        last_send_time = asyncio.get_event_loop().time()
        min_interval = 0.005  # 5ms between chunks

        while not stop.is_set() and not answer_ready.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                if chunk is None:
                    break

                # Rate limit to prevent queue overflow
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - last_send_time
                if time_since_last < min_interval:
                    await asyncio.sleep(min_interval - time_since_last)

                audio_base64 = base64.b64encode(chunk).decode("utf-8")
                await connection.send({"audio_base_64": audio_base64})
                last_send_time = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                continue

    sender = asyncio.create_task(send_audio())
    try:
        # Wait for either answer_ready or stop event
        await asyncio.wait(
            [
                asyncio.create_task(answer_ready.wait()),
                asyncio.create_task(stop.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        print("[STT] Closing session")
        await connection.close()
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass


async def receive_audio(
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    audioBytes: bytearray,
    res_queue: asyncio.Queue,
):
    try:
        while True:
            # get audio data
            data = await websocket.receive_bytes()
            audio_queue.put_nowait(data)
            audioBytes.extend(data)
            # Check for responses and send to client
            while not res_queue.empty():
                res = await res_queue.get()
                await websocket.send_json(res)
    except WebSocketDisconnect:
        print("[STREAMING] receive_audio: WebSocket disconnected")
        raise
    except Exception as e:
        print(f"[STREAMING] receive_audio error: {e}")
        raise


@router.websocket(os.getenv("AGENT_URL"))
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()

    # session_id = uuid.uuid4()
    audio_queue = asyncio.Queue()
    res_queue = asyncio.Queue()
    audioBytes = bytearray()

    try:
        mood_confidence = 0.0
        question_counter = 0
        current_depth = 0
        max_depth = 3
        max_questions = 5
        # [question, answer]
        qa_pairs: list[tuple[str, str]] = []
        # [mood, confidence]
        moods: list[tuple[str, float]] = []
        wheel = get_wheel_of_emotions()

        # Start receiving audio continuously
        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audioBytes, res_queue)
        )

        # Continue asking questions until:
        # 1. Confidence >= 0.9 AND depth == 3 (tertiary emotion with high confidence), OR
        # 2. Max questions reached
        while question_counter < max_questions:
            # Check if we should stop: high confidence AND max depth reached
            if mood_confidence >= 0.9 and current_depth >= max_depth:
                print(
                    f"[AGENT] Stopping: High confidence ({mood_confidence}) at depth {current_depth}"
                )
                break

            # get question from agent
            if question_counter == 0:
                question = "Hello! How are you feeling today?"
            else:
                question = await get_next_question(
                    qa_pairs, moods, current_depth, max_depth
                )

            # ask question
            print(f"[AGENT] Question {question_counter + 1}: {question}")
            await websocket.send_json({"type": "question", "text": question})

            # Wait for user to read the question (frontend shows grey ball during this time)
            print("[AGENT] Waiting 5 seconds for user to read question...")
            await asyncio.sleep(5.0)

            # NOW clear old audio from queue (discard the 5 seconds of silence/ambient noise)
            cleared = 0
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    cleared += 1
                except asyncio.QueueEmpty:
                    break
            if cleared > 0:
                print(f"[AGENT] Cleared {cleared} old audio chunks from queue")

            # Signal frontend that we're now listening for the answer (turn ball red)
            print("[AGENT] Now listening for user response...")
            await websocket.send_json({"type": "listening"})

            # Start new STT session for this answer
            answer_transcript_container = {"current": ""}
            answer_ready = asyncio.Event()

            print(f"[AGENT] Starting STT session for question {question_counter + 1}")
            stt_task = asyncio.create_task(
                stt_elevenlabs_session(
                    audio_queue, res_queue, answer_transcript_container, answer_ready
                )
            )

            # Wait for VAD to detect end of speech
            print(
                f"[AGENT] Waiting for user response to question {question_counter + 1}..."
            )
            await answer_ready.wait()

            # Wait for STT task to complete cleanup
            await stt_task

            answer_transcript = answer_transcript_container["current"]
            print(f"[AGENT] Received answer: {answer_transcript}")

            # analyze response
            await websocket.send_json({"type": "analyzing"})
            mood, mood_confidence = await analyze_mood(
                qa_pairs, moods, question, answer_transcript
            )

            # Determine depth of detected emotion
            current_depth = get_emotion_depth(mood, wheel)
            depth_name = (
                ["unknown", "primary", "secondary", "tertiary"][current_depth]
                if current_depth <= 3
                else "unknown"
            )

            print(
                f"[AGENT] Detected mood: {mood} ({depth_name} level), confidence: {mood_confidence}"
            )

            # go to next question
            qa_pairs.append((question, answer_transcript))
            moods.append((mood, mood_confidence))
            question_counter += 1

        # Determine final depth
        final_depth = get_emotion_depth(mood, wheel)
        depth_name = (
            ["unknown", "primary", "secondary", "tertiary"][final_depth]
            if final_depth <= 3
            else "unknown"
        )

        if mood_confidence >= 0.9 and final_depth >= max_depth:
            print(
                f"[AGENT] Mood detected with high confidence at maximum depth: {mood} ({depth_name})"
            )
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )
        elif mood_confidence >= 0.9:
            print(
                f"[AGENT] Mood detected with high confidence but not at max depth: {mood} ({depth_name}, depth {final_depth}/{max_depth})"
            )
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )
        else:
            print(
                f"[AGENT] Max questions reached. Best mood: {mood} ({depth_name}), confidence: {mood_confidence}"
            )
            # Send the best mood we found, even if not 0.9 confidence
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )

        # Give frontend time to receive final message before cleanup
        await asyncio.sleep(0.5)

    except WebSocketDisconnect as e:
        print(f"[AGENT] Websocket disconnected: {e}")
    except Exception as e:
        print(f"[AGENT] error during websocket communication: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # cleanup
        print("[AGENT] Cleaning up websocket session")
        if "receive_task" in locals():
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception as e:
                print(f"[AGENT] Error during receive_task cleanup: {e}")

        # TODO upload session_id, full audio, qa_pairs, moods
