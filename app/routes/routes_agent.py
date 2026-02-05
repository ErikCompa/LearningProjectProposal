import asyncio
import base64
import json
import os
import threading
import uuid
from datetime import datetime

from elevenlabs import RealtimeAudioOptions, RealtimeEvents, VoiceSettings
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.websockets import WebSocketState

from app.deps import get_elevenlabs
from app.models import AgentSession, QAMoodPair
from app.openai_agent import (
    openai_analyze_conversation_mood,
    openai_create_conversation,
    openai_get_conversation_next_question,
)
from app.services import upload_agent_audio_to_bucket, upload_agent_session

router = APIRouter(tags=["agent"])


async def stt_elevenlabs_session(
    audio_queue, res_queue, answer_transcript_container, answer_ready, websocket
):
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
        transcript_data = {
            "type": "transcript",
            "transcript": data.get("text", ""),
            "is_final": False,
        }
        res_queue.put_nowait(transcript_data)
        # send to frontend
        asyncio.create_task(websocket.send_json(transcript_data))

    def on_committed_transcript(data):
        text = data.get("text", "")
        answer_transcript_container["current"] += text
        transcript_data = {
            "type": "transcript",
            "transcript": text,
            "is_final": True,
        }
        res_queue.put_nowait(transcript_data)
        # send to frontend
        asyncio.create_task(websocket.send_json(transcript_data))
        # signal that answer is ready (VAD detected end of speech)
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

                # rate limit to prevent queue overflow
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
        # wait for either answer_ready or stop event
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
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                print("[AGENT] Client disconnected during audio reception")
                break

            # handle JSON messages (like audio_playback_finished)
            if message["type"] == "websocket.receive":
                if "text" in message:
                    try:
                        json_message = json.loads(message["text"])
                        if json_message.get("type") == "audio_playback_finished":
                            # Put signal in response queue
                            await res_queue.put({"type": "audio_playback_finished"})
                            continue
                    except json.JSONDecodeError:
                        pass

                # handle binary audio data
                if "bytes" in message:
                    audio_data = message["bytes"]
                    audioBytes.extend(audio_data)
                    await audio_queue.put(audio_data)
    except WebSocketDisconnect:
        print("[AGENT] WebSocket disconnected in receive_audio")
    except Exception as e:
        print(f"[AGENT] Error in receive_audio: {e}")


def upload_session_in_background(
    audio_bytes: bytearray,
    session_id: str,
    session_timestamp: str,
    qa_pairs_with_moods: list[QAMoodPair],
    final_mood: str,
    final_confidence: float,
    question_count: int,
):
    try:
        # only upload if we have audio data
        if not audio_bytes:
            print(f"[AGENT] No audio data to upload for session: {session_id}")
            return

        # upload audio to bucket
        audio_url = upload_agent_audio_to_bucket(
            bytes(audio_bytes), session_id, session_timestamp
        )

        # create session object
        session = AgentSession(
            session_id=session_id,
            created_at=datetime.fromisoformat(session_timestamp),
            qa_pairs=qa_pairs_with_moods,
            final_mood=final_mood,
            final_confidence=final_confidence,
            question_count=question_count,
            audio_url=audio_url,
        )

        # upload session to Firestore
        upload_agent_session(session)
        print(f"[AGENT] Background upload completed for session: {session_id}")
    except Exception as e:
        print(f"[AGENT] Background upload failed for session {session_id}: {e}")


async def tts_elevenlabs_session(text: str, websocket: WebSocket):
    response = get_elevenlabs().text_to_speech.stream(
        voice_id="I3MrSgiotopLY33bjEX7",  # Yaron: I3MrSgiotopLY33bjEX7, Erik: VWoIQlDpnFjY9kfJ11dz, Adam: pNInz6obpgDQGcFmaJgB
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    # send question audio to client
    for chunk in response:
        if chunk and websocket.application_state == WebSocketState.CONNECTED:
            audio_base64 = base64.b64encode(chunk).decode("utf-8")
            await websocket.send_json(
                {"type": "question_audio_base_64", "chunk": audio_base64}
            )
    await asyncio.sleep(0.1)


@router.websocket(os.getenv("AGENT_URL"))
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session_timestamp = datetime.now().isoformat()
    audio_queue = asyncio.Queue()
    res_queue = asyncio.Queue()
    audioBytes = bytearray()

    llm = websocket.query_params.get("llm", "openai").lower()
    print(f"[AGENT] Using LLM: {llm}")

    if llm == "openai":
        conversation_id = await openai_create_conversation(session_id)

    try:
        mood_confidence = 0.0
        question_counter = 0
        max_questions = 5
        # [question, answer]
        qa_pairs: list[tuple[str, str]] = []
        # [mood, confidence]
        moods: list[tuple[str, float]] = []
        # QAPair objects for upload
        qa_pairs_with_moods: list[QAMoodPair] = []

        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audioBytes, res_queue)
        )

        while question_counter < max_questions:
            # check if should stop: high confidence reached
            if mood_confidence >= 0.9:
                print(
                    f"[AGENT] Stopping: High confidence ({mood_confidence}) at confidence {mood_confidence}"
                )
                break

            # get question from agent
            if question_counter == 0:
                question = "Hello! How are you feeling today?"
            else:
                try:
                    question = await openai_get_conversation_next_question(
                        moods,
                        mood_confidence,
                        conversation_id,
                    )
                    print(f"[AGENT] Generated next question: {question}")
                    print(f"[AGENT] Question type: {type(question)}")
                except Exception as e:
                    print(f"[AGENT] Error generating next question: {e}")
                    raise

            # ask question
            print(f"[AGENT] Question {question_counter + 1}: {question}")
            await tts_elevenlabs_session(question, websocket)
            print("[AGENT] Sent all audio chunks, now sending question text")
            await websocket.send_json({"type": "question", "text": question})

            # wait for frontend to signal audio playback finished via res_queue
            print("[AGENT] Waiting for frontend audio playback to finish...")
            try:
                # keep reading from queue until we get the audio_playback_finished signal
                while True:
                    response = await asyncio.wait_for(res_queue.get(), timeout=30.0)
                    print(f"[AGENT] Received response from res_queue: {response}")
                    if response.get("type") == "audio_playback_finished":
                        print(
                            "[AGENT] Frontend audio playback finished, proceeding to listening"
                        )
                        break
                    else:
                        print(
                            f"[AGENT] Ignoring message type: {response.get('type')} (waiting for audio_playback_finished)"
                        )
            except asyncio.TimeoutError:
                print("[AGENT] Timeout waiting for audio playback finished signal")

            # clear old audio from queue
            cleared = 0
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    cleared += 1
                except asyncio.QueueEmpty:
                    break
            if cleared > 0:
                print(f"[AGENT] Cleared {cleared} old audio chunks from queue")

            print("[AGENT] Now listening for user response...")
            await websocket.send_json({"type": "listening"})

            answer_transcript_container = {"current": ""}
            answer_ready = asyncio.Event()

            print(f"[AGENT] Starting STT session for question {question_counter + 1}")
            stt_task = asyncio.create_task(
                stt_elevenlabs_session(
                    audio_queue,
                    res_queue,
                    answer_transcript_container,
                    answer_ready,
                    websocket,
                )
            )

            print(
                f"[AGENT] Waiting for user response to question {question_counter + 1}..."
            )
            await answer_ready.wait()
            await stt_task

            answer_transcript = answer_transcript_container["current"]
            print(f"[AGENT] Received answer: {answer_transcript}")

            # analyze response
            await websocket.send_json({"type": "analyzing"})

            mood, mood_confidence = await openai_analyze_conversation_mood(
                question,
                answer_transcript,
                conversation_id,
            )

            # go to next question
            qa_pairs.append((question, answer_transcript))
            moods.append((mood, mood_confidence))
            qa_pairs_with_moods.append(
                QAMoodPair(
                    question=question,
                    answer=answer_transcript,
                    mood=mood,
                    confidence=mood_confidence,
                )
            )
            question_counter += 1

        if mood_confidence >= 0.9:
            print(f"[AGENT] Mood detected with high confidence: {mood}")
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )
        else:
            print(
                f"[AGENT] Max questions reached. Best mood: {mood}, confidence: {mood_confidence}"
            )
            # send the best mood, even if not 0.9 confidence
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )

        # give frontend time to receive final message before cleanup
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

        # upload session data in background (works for complete and incomplete sessions)
        if "qa_pairs_with_moods" in locals():
            upload_thread = threading.Thread(
                target=upload_session_in_background,
                args=(
                    audioBytes,
                    session_id,
                    session_timestamp,
                    qa_pairs_with_moods,
                    mood if "mood" in locals() else "unknown",
                    mood_confidence if "mood_confidence" in locals() else 0.0,
                    question_counter if "question_counter" in locals() else 0,
                ),
                daemon=True,
            )
            upload_thread.start()
            print(f"[AGENT] Started background upload for session: {session_id}")
