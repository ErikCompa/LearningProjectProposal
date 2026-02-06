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
    openai_suggest_music,
)
from app.services import upload_agent_audio_to_bucket, upload_agent_session

router = APIRouter(tags=["agent"])


async def stt_elevenlabs_session(
    audio_queue, res_queue, answer_transcript_container, answer_ready, websocket
):
    print("[WEBSOCKET-STT] Starting ElevenLabs STT session")
    elevenlabs = get_elevenlabs()
    stop = asyncio.Event()

    connection = await elevenlabs.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=16000,
            include_timestamps=True,
            commit_strategy=CommitStrategy.VAD,
            vad_silence_threshold_secs=3.0,
            vad_threshold=0.4,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
        )
    )

    def on_session_started(data):
        print(f"[WEBSOCKET-STT] Session started: {data.get('session_id', 'unknown')}")

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
        print(f"[WEBSOCKET-STT] VAD detected silence, answer complete: {text}")
        answer_ready.set()

    def on_error(error):
        print(f"[WEBSOCKET-STT] Error: {error}")
        stop.set()

    def on_close():
        print("[WEBSOCKET-STT] Connection closed by server")
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
        print("[WEBSOCKET-STT] Closing session")
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
                print(
                    "[WEBSOCKET-RECEIVE_AUDIO] Client disconnected during audio reception"
                )
                break

            if message["type"] == "websocket.receive":
                if "text" in message:
                    try:
                        json_message = json.loads(message["text"])
                        if json_message.get("type") == "audio_playback_finished":
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
        print("[WEBSOCKET-RECEIVE_AUDIO] WebSocket disconnected in receive_audio")
    except Exception as e:
        print(f"[WEBSOCKET-RECEIVE_AUDIO] Error in receive_audio: {e}")


def upload_session_in_background(
    audio_bytes: bytearray,
    session_id: str,
    session_timestamp: str,
    qa_pairs_with_moods: list[QAMoodPair],
    final_mood: str,
    final_confidence: float,
    total_question_count: int,
    direct_question_count: int,
):
    try:
        if not audio_bytes:
            print(
                f"[WEBSOCKET-UPLOAD] No audio data to upload for session: {session_id}"
            )
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
            total_question_count=total_question_count,
            direct_question_count=direct_question_count,
            audio_url=audio_url,
        )

        # upload session to Firestore
        upload_agent_session(session)
        print(
            f"[WEBSOCKET-UPLOAD] Background upload completed for session: {session_id}"
        )
    except Exception as e:
        print(
            f"[WEBSOCKET-UPLOAD] Background upload failed for session {session_id}: {e}"
        )


async def tts_elevenlabs_session(text: str, websocket: WebSocket):
    print(f"[WEBSOCKET-TTS] Sending text to ElevenLabs TTS: {text}")
    try:
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
    except Exception as e:
        print(f"[WEBSOCKET-TTS] Error during ElevenLabs TTS: {e}")
    await asyncio.sleep(0.1)


@router.websocket(os.getenv("AGENT_URL"))
async def websocket_agent(websocket: WebSocket):
    print("[WEBSOCKET] Client connected")
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session_timestamp = datetime.now().isoformat()
    audio_queue = asyncio.Queue()
    res_queue = asyncio.Queue()
    audioBytes = bytearray()

    conversation_id = await openai_create_conversation(session_id)

    try:
        mood_confidence = 0.0
        direct_question_counter = 0
        total_question_counter = 0
        max_direct_questions = 5
        initial_question = True
        # [question, answer]
        qa_pairs: list[tuple[str, str]] = []
        # [mood, confidence]
        moods: list[tuple[str, float]] = []
        # QAPair objects for upload
        qa_pairs_with_moods: list[QAMoodPair] = []

        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audioBytes, res_queue)
        )

        while direct_question_counter < max_direct_questions:
            # check if should stop: high confidence reached
            if (
                mood_confidence >= 0.9
                or direct_question_counter >= max_direct_questions
            ):
                print(f"[WEBSOCKET] Stopping: confidence ({mood_confidence})")
                print(
                    f"[WEBSOCKET] Stopping: direct questions asked ({direct_question_counter})"
                )
                break

            # get question from agent
            if initial_question:
                question = "Hello! How are you feeling today?"
                print(f"[WEBSOCKET] Initial question: {question}")
                initial_question = False
            else:
                try:
                    question = await openai_get_conversation_next_question(
                        direct_question_counter,
                        moods,
                        mood_confidence,
                        conversation_id,
                    )
                    print(f"[WEBSOCKET] Generated next question: {question}")
                except Exception as e:
                    print(f"[WEBSOCKET] Error generating next question: {e}")
                    raise
                if question.is_direct:
                    direct_question_counter += 1
                question = question.question

            total_question_counter += 1
            # ask question
            await tts_elevenlabs_session(question, websocket)
            await websocket.send_json({"type": "question", "text": question})

            # wait for audio playback finished signal
            try:
                while True:
                    response = await asyncio.wait_for(res_queue.get(), timeout=30.0)
                    if response.get("type") == "audio_playback_finished":
                        print(
                            "[WEBSOCKET] Frontend audio playback finished, proceeding to listening"
                        )
                        break
            except asyncio.TimeoutError:
                print("[WEBSOCKET] Timeout waiting for audio playback finished signal")

            # clear old audio from queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # listen for answer
            print("[WEBSOCKET] Now listening for user response...")
            await websocket.send_json({"type": "listening"})

            answer_transcript_container = {"current": ""}
            answer_ready = asyncio.Event()
            stt_task = asyncio.create_task(
                stt_elevenlabs_session(
                    audio_queue,
                    res_queue,
                    answer_transcript_container,
                    answer_ready,
                    websocket,
                )
            )

            # wait for VAD signal
            await answer_ready.wait()
            await stt_task
            answer_transcript = answer_transcript_container["current"]
            print(f"[WEBSOCKET] Received answer: {answer_transcript}")

            # analyze response
            await websocket.send_json({"type": "analyzing"})
            (
                mood,
                mood_confidence,
                negative_emotion_percentages,
            ) = await openai_analyze_conversation_mood(
                question,
                answer_transcript,
                conversation_id,
            )

            # log mood analysis results
            print(f"[WEBSOCKET] Mood: {mood}, Confidence: {mood_confidence}")
            if negative_emotion_percentages:
                print(
                    f"[WEBSOCKET] Negative emotion percentages: {negative_emotion_percentages}"
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
                    negative_emotion_percentages=negative_emotion_percentages,
                )
            )

        if mood_confidence >= 0.9:
            print(
                f"[WEBSOCKET] High confidence reached: {mood}, confidence: {mood_confidence}"
            )
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )
        else:
            print(
                f"[WEBSOCKET] Max direct questions reached. Best mood: {mood}, confidence: {mood_confidence}"
            )
            await websocket.send_json(
                {"type": "result", "mood": mood, "confidence": mood_confidence}
            )

        # recommend music based on mood
        user_preferences = ["metal", "rock"]
        music = await openai_suggest_music(user_preferences, conversation_id)
        print(f"[WEBSOCKET] Music recommendation based on mood ({mood}): {music}")
        await websocket.send_json({"type": "music_recommendation", "music": music.song})

        # give frontend time to receive and display final message before cleanup
        await asyncio.sleep(0.5)

    except WebSocketDisconnect as e:
        print(f"[WEBSOCKET] Websocket disconnected: {e}")
    except Exception as e:
        print(f"[WEBSOCKET] error during websocket communication: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # cleanup
        print("[WEBSOCKET] Cleaning up websocket session")
        if "receive_task" in locals():
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception as e:
                print(f"[WEBSOCKET] Error during receive_task cleanup: {e}")

        # upload session data in background
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
                    total_question_counter
                    if "total_question_counter" in locals()
                    else 0,
                    direct_question_counter
                    if "direct_question_counter" in locals()
                    else 0,
                ),
                daemon=True,
            )
            upload_thread.start()
            print(f"[WEBSOCKET] Started background upload for session: {session_id}")
