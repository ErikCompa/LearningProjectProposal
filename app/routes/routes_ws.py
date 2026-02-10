import asyncio
import json
import os
import threading
import uuid
from datetime import datetime

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app.elevenlabs import (
    stt_elevenlabs_session,
    tts_elevenlabs_session,
)
from app.models import QAMoodPair
from app.openai_agent import (
    openai_analyze_conversation_mood,
    openai_create_conversation,
    openai_get_conversation_next_question,
    openai_suggest_music,
)
from app.services import (
    upload_session_in_background,
)

router = APIRouter(tags=["agent"])


# receive audio data from frontend
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


async def listen_for_answer(
    audio_queue: asyncio.Queue, res_queue: asyncio.Queue, websocket: WebSocket
):
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
    return answer_transcript


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
        mood = "neutral"
        mood_confidence = 0.0
        direct_question_counter = 0
        total_question_counter = 0
        initial_question = True
        reminder_asked = False
        # [question, answer]
        qa_pairs: list[tuple[str, str]] = []
        # [mood, confidence]
        moods: list[tuple[str, float]] = []
        # QAPair objects for upload
        qa_pairs_with_moods: list[QAMoodPair] = []

        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audioBytes, res_queue)
        )

        while True:
            # get question from agent
            if initial_question:
                question = 'Hello! How are you feeling today? If you say "Play me some music", I can play you a song.'
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

            # check for reminder suggestion
            music_reminder = f'It sounds like you\'re feeling {mood}, I can play you some music anytime that I think would suit your mood. Just say "Play me some music".'
            if mood_confidence > 0.8 and not reminder_asked and len(qa_pairs) > 4:
                question += f" {music_reminder}"
                reminder_asked = True

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
            answer_transcript = await listen_for_answer(
                audio_queue, res_queue, websocket
            )

            # if empty try again
            if not answer_transcript.strip():
                print("[WEBSOCKET] Empty transcript received, asking user to repeat")

                # Send TTS audio for the retry message
                retry_message = "Sorry, I didn't catch that. If you'd like me to play some music just say 'Play me some music'"
                await tts_elevenlabs_session(retry_message, websocket)
                await websocket.send_json(
                    {
                        "type": "empty_transcript",
                        "message": retry_message,
                    }
                )

                # Wait for audio playback finished signal
                try:
                    while True:
                        response = await asyncio.wait_for(res_queue.get(), timeout=30.0)
                        if response.get("type") == "audio_playback_finished":
                            print(
                                "[WEBSOCKET] Empty transcript audio playback finished"
                            )
                            break
                except asyncio.TimeoutError:
                    print(
                        "[WEBSOCKET] Timeout waiting for empty transcript audio playback"
                    )

                # Clear old audio from queue
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                answer_transcript = await listen_for_answer(
                    audio_queue, res_queue, websocket
                )
                if not answer_transcript.strip():
                    break  # if still empty, end session

            # if "play me some music"
            if "play me some music" in answer_transcript.lower():
                print("[WEBSOCKET] User requested music, breaking out of question loop")
                break

            # analyze response
            await websocket.send_json({"type": "analyzing"})
            (
                mood,
                mood_confidence,
                negative_emotion_percentages,
                music_requested,
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

            if music_requested:
                print("[WEBSOCKET] User requested music in mood analysis response")
                break

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

        # Send result only if we actually analyzed the mood
        if qa_pairs_with_moods:
            if mood_confidence >= 0.8:
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
