import asyncio
import json
import os
import threading
import uuid
from datetime import datetime

from agents import Runner
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

from app import main_agent
from app.elevenlabs import (
    stt_elevenlabs_session,
    tts_elevenlabs_session,
)
from app.models import QAEmotionPair
from app.services import (
    upload_session_in_background,
)

router = APIRouter(tags=["agent"])


# helper to send status updates to frontend
async def send_status(websocket: WebSocket, status_type: str, data: dict = None):
    payload = {"type": status_type}
    if data:
        payload.update(data)
    await websocket.send_json(payload)


# helper to empty audio q and not use old audio data in next STT session
def clear_audio_queue(audio_queue: asyncio.Queue):
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


# helper to wait for frontend signal
async def wait_for_playback_finished(res_queue: asyncio.Queue, timeout: float = 30.0):
    try:
        while True:
            response = await asyncio.wait_for(res_queue.get(), timeout=timeout)
            if response.get("type") == "audio_playback_finished":
                break
    except asyncio.TimeoutError:
        print("[WEBSOCKET] Timeout waiting for audio playback finished signal")


# receives constant stream of audio from frontend
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


# listens for user response using elevenlabs STT
async def listen_for_answer(
    audio_queue: asyncio.Queue, res_queue: asyncio.Queue, websocket: WebSocket
):
    print("[WEBSOCKET] Now listening for user response...")
    # update frontend
    await send_status(websocket, "listening")

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


# main function to ask and listen
async def ask_question_and_get_response(
    question: str,
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    res_queue: asyncio.Queue,
):
    await tts_elevenlabs_session(question, websocket)
    # update frontend
    await send_status(websocket, "question", {"text": question})

    await wait_for_playback_finished(res_queue)

    clear_audio_queue(audio_queue)

    answer_transcript = await listen_for_answer(audio_queue, res_queue, websocket)

    if not answer_transcript.strip():
        retry_message = "Sorry, I didn't catch that. If you'd like me to play some music just say 'Play me some music'"
        await tts_elevenlabs_session(retry_message, websocket)
        await send_status(websocket, "empty_transcript", {"message": retry_message})

        await wait_for_playback_finished(res_queue)
        clear_audio_queue(audio_queue)

        answer_transcript = await listen_for_answer(audio_queue, res_queue, websocket)

    return answer_transcript


@router.websocket(os.getenv("AGENT_URL"))
async def websocket_agent(websocket: WebSocket):
    print("[WEBSOCKET] Client connected")
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session_timestamp = datetime.now().isoformat()
    audio_queue = asyncio.Queue()
    res_queue = asyncio.Queue()
    audio_bytes = bytearray()
    qa_pairs: list[QAEmotionPair] = []
    receive_task = None

    try:
        current_question = None
        current_is_direct = False
        high_confidence_reached = False
        music_reminder_given = False
        direct_question_count = 0

        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audio_bytes, res_queue)
        )

        initial_message = 'Hello! How are you feeling today? If you say "Play me some music", I can play you a song.'
        current_question = initial_message
        user_input = await ask_question_and_get_response(
            initial_message, websocket, audio_queue, res_queue
        )

        while True:
            # empty response
            if not user_input.strip():
                print(
                    "[WEBSOCKET] Two consecutive empty responses - ending with music recommendation"
                )
                if qa_pairs:
                    last_qa = qa_pairs[-1]
                    final_emotion = last_qa.emotion
                    final_confidence = last_qa.confidence
                else:
                    final_emotion = "Calm"
                    final_confidence = 0.5

                await send_status(
                    websocket,
                    "result",
                    {"mood": final_emotion, "confidence": final_confidence},
                )

                # force music rec
                user_input = "play me some music"

            await send_status(websocket, "analyzing")

            # build existing context
            qa_pairs_json = json.dumps(
                [
                    {
                        "question": qa.question,
                        "answer": qa.answer,
                        "emotion": qa.emotion,
                        "confidence": qa.confidence,
                        "is_direct": qa.is_direct,
                    }
                    for qa in qa_pairs
                ]
            )

            main_agent_prompt = f"""
                User message: "{user_input}"

                Current context:
                - Questions asked so far: {len(qa_pairs)}
                - Direct questions used: {direct_question_count}/5
                - High confidence reached: {high_confidence_reached}
                - Music reminder already given: {music_reminder_given}

                Previous Q&A pairs:
                {qa_pairs_json}

                Process this user input according to your workflow.
            """

            agent_result = await Runner.run(main_agent, main_agent_prompt)

            final_output = agent_result.final_output
            print(f"[WEBSOCKET] Main agent output: {final_output}")

            if isinstance(final_output, dict):
                result_data = final_output
            elif hasattr(final_output, "model_dump"):
                result_data = final_output.model_dump()
            else:
                print(f"[WEBSOCKET] Unexpected output format: {type(final_output)}")
                break

            # handle question response
            if result_data.get("question") is not None:
                next_question = result_data["question"]
                emotion = result_data.get("emotion")
                confidence = result_data.get("confidence")

                qa_pairs.append(
                    QAEmotionPair(
                        question=current_question,
                        answer=user_input,
                        emotion=emotion,
                        confidence=confidence,
                        negative_emotion_percentages=result_data.get(
                            "negative_emotion_percentages"
                        ),
                        is_direct=current_is_direct,
                    )
                )

                if result_data.get("is_direct", False):
                    direct_question_count += 1

                if confidence >= 0.8 and not high_confidence_reached:
                    high_confidence_reached = True
                    print(
                        "[WEBSOCKET] High confidence (>= 0.8) reached for the first time."
                    )

                if "Play me some music" in next_question and not music_reminder_given:
                    music_reminder_given = True
                    print("[WEBSOCKET] Music reminder has been given to the user.")

                current_question = next_question
                current_is_direct = result_data.get("is_direct", False)

                # send emotion results to frontend
                await send_status(
                    websocket,
                    "intermediate_result",
                    {
                        "mood": emotion,
                        "confidence": confidence,
                        "negative_emotion_percentages": result_data.get(
                            "negative_emotion_percentages"
                        ),
                    },
                )

                # ask next question
                user_input = await ask_question_and_get_response(
                    next_question, websocket, audio_queue, res_queue
                )

            # handle music response
            elif result_data.get("song") is not None:
                music_song = result_data.get("song")
                emotion = result_data.get(
                    "emotion", qa_pairs[-1].emotion if qa_pairs else "Calm"
                )
                confidence = result_data.get(
                    "confidence", qa_pairs[-1].confidence if qa_pairs else 0.5
                )

                # final pair
                qa_pairs.append(
                    QAEmotionPair(
                        question=current_question,
                        answer=user_input,
                        emotion=emotion,
                        confidence=confidence,
                        negative_emotion_percentages=result_data.get(
                            "negative_emotion_percentages"
                        ),
                        is_direct=current_is_direct,
                    )
                )

                await send_status(
                    websocket, "result", {"mood": emotion, "confidence": confidence}
                )
                await send_status(
                    websocket, "music_recommendation", {"music": music_song}
                )
                print(f"[WEBSOCKET] Music recommendation: {music_song}")
                break

            # main agent returned something unexpected
            else:
                print(f"[WEBSOCKET] Unknown result format: {result_data}")
                break

    except WebSocketDisconnect as e:
        print(f"[WEBSOCKET] WebSocket disconnected: {e}")
    except Exception as e:
        print(f"[WEBSOCKET] Error during websocket communication: {e}")
        try:
            await send_status(websocket, "error", {"message": str(e)})
        except Exception:
            pass
    finally:
        print("[WEBSOCKET] Cleaning up websocket session")
        if receive_task:
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception as e:
                print(f"[WEBSOCKET] Error during receive_task cleanup: {e}")

        # upload session data in background thread
        if qa_pairs:
            print(f"[WEBSOCKET] Uploading {len(qa_pairs)} QA pairs")
            last_qa = qa_pairs[-1]
            upload_thread = threading.Thread(
                target=upload_session_in_background,
                args=(
                    audio_bytes,
                    session_id,
                    session_timestamp,
                    qa_pairs,
                    last_qa.emotion,
                    last_qa.confidence,
                    len(qa_pairs),
                    sum(1 for qa in qa_pairs if qa.is_direct),
                ),
                daemon=True,
            )
            upload_thread.start()
            print(f"[WEBSOCKET] Started background upload for session: {session_id}")
        else:
            print("[WEBSOCKET] No QA pairs to upload")
