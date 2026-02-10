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

from app import emotion_agent, orchestration_agent
from app.elevenlabs import (
    stt_elevenlabs_session,
    tts_elevenlabs_session,
)
from app.models import QAEmotionPair
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


async def ask_question_and_get_response(
    question: str,
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    res_queue: asyncio.Queue,
):
    # Send question via TTS
    await tts_elevenlabs_session(question, websocket)
    await websocket.send_json({"type": "question", "text": question})

    # Wait for audio playback finished
    try:
        while True:
            response = await asyncio.wait_for(res_queue.get(), timeout=30.0)
            if response.get("type") == "audio_playback_finished":
                break
    except asyncio.TimeoutError:
        print("[WEBSOCKET] Timeout waiting for audio playback finished signal")

    # Clear old audio from queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    # Listen for answer
    answer_transcript = await listen_for_answer(audio_queue, res_queue, websocket)

    # Handle empty response
    if not answer_transcript.strip():
        retry_message = "Sorry, I didn't catch that. If you'd like me to play some music just say 'Play me some music'"
        await tts_elevenlabs_session(retry_message, websocket)
        await websocket.send_json(
            {"type": "empty_transcript", "message": retry_message}
        )

        try:
            while True:
                response = await asyncio.wait_for(res_queue.get(), timeout=30.0)
                if response.get("type") == "audio_playback_finished":
                    break
        except asyncio.TimeoutError:
            pass

        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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
    audioBytes = bytearray()
    qa_pairs_with_emotions: list[
        QAEmotionPair
    ] = []  # Initialize here so finally block can access it
    receive_task = None  # Initialize here so finally block can access it

    # Context functions that orchestration agent can use
    async def speak_and_listen(message: str) -> str:
        """Send TTS message and get STT response"""
        return await ask_question_and_get_response(
            message, websocket, audio_queue, res_queue
        )

    async def send_status(status_type: str, data: dict = None):
        """Send status updates to frontend"""
        payload = {"type": status_type}
        if data:
            payload.update(data)
        await websocket.send_json(payload)

    try:
        current_question = None  # Track the current question being asked
        current_is_direct = False  # Track if current question is direct

        receive_task = asyncio.create_task(
            receive_audio(websocket, audio_queue, audioBytes, res_queue)
        )

        # Build context for orchestration agent
        context = {
            "session_id": session_id,
            "speak_and_listen": speak_and_listen,
            "send_status": send_status,
            "qa_pairs": qa_pairs_with_emotions,
        }

        # Let orchestration agent run the entire conversation
        initial_message = 'Hello! How are you feeling today? If you say "Play me some music", I can play you a song.'
        current_question = initial_message

        # Start conversation with orchestration agent
        user_input = await speak_and_listen(initial_message)
        high_confidence_reached = False  # Track if we've hit 80%+ confidence

        # Main conversation loop - two-step: emotion analysis then orchestration routing
        while True:
            emotion_data = None  # Initialize for proper scoping

            # Check for empty response - ask_question_and_get_response already retries once,
            # so if we get empty here, it's already 2 consecutive VAD timeouts
            if not user_input.strip():
                print(
                    "[WEBSOCKET] Two consecutive empty responses (VAD timeout) - ending with music recommendation"
                )
                # Use last known emotion if available, otherwise use neutral default
                if qa_pairs_with_emotions:
                    last_qa = qa_pairs_with_emotions[-1]
                    emotion_data = {
                        "emotion": last_qa.emotion,
                        "confidence": last_qa.confidence,
                        "negative_emotion_percentages": last_qa.negative_emotion_percentages,
                    }
                    print(
                        f"[WEBSOCKET] Using last emotion for final recommendation: {emotion_data}"
                    )
                else:
                    # Use neutral emotion as default
                    emotion_data = {
                        "emotion": "Calm",
                        "confidence": 0.5,
                        "negative_emotion_percentages": None,
                    }
                    print(
                        f"[WEBSOCKET] No previous emotions - using neutral default: {emotion_data}"
                    )

                # Send emotion result to frontend
                await send_status(
                    "result",
                    {
                        "mood": emotion_data["emotion"],
                        "confidence": emotion_data["confidence"],
                    },
                )

                # Get music recommendation using orchestration agent
                user_input = "play me some music"
                # Continue to music recommendation flow below

            await send_status("analyzing")

            # Check if user is requesting music - if so, skip emotion analysis and use last known emotion
            is_music_request = "play me some music" in user_input.lower()

            if is_music_request and (qa_pairs_with_emotions or emotion_data):
                # Use emotion_data that was set above (either from last QA or VAD timeout default)
                if not emotion_data:
                    # Fallback: use last QA emotion
                    last_qa = qa_pairs_with_emotions[-1]
                    emotion_data = {
                        "emotion": last_qa.emotion,
                        "confidence": last_qa.confidence,
                        "negative_emotion_percentages": last_qa.negative_emotion_percentages,
                    }
                print(
                    f"[WEBSOCKET] Music request detected - using emotion: {emotion_data}"
                )
            else:
                # Step 1: Analyze emotion with Emotion Agent
                emotion_prompt = f'User message: "{user_input}"\n\nAnalyze the emotion in this message.'
                emotion_result = await Runner.run(emotion_agent, emotion_prompt)
                emotion_output = emotion_result.final_output

                # Extract emotion data
                emotion_data = (
                    emotion_output
                    if isinstance(emotion_output, dict)
                    else emotion_output.model_dump()
                    if hasattr(emotion_output, "model_dump")
                    else None
                )

                print(f"[WEBSOCKET] Emotion analysis: {emotion_data}")

                # Check if confidence reached 80% or higher for the first time
                if (
                    emotion_data
                    and emotion_data.get("confidence", 0) >= 0.8
                    and not high_confidence_reached
                ):
                    high_confidence_reached = True
                    print(
                        "[WEBSOCKET] High confidence reached (≥80%) - will remind user about music option"
                    )

            # Step 2: Route with Orchestration Agent
            orchestration_prompt = f"""
            User message: "{user_input}"
            
            Context:
            - Questions asked so far: {len(qa_pairs_with_emotions)}
            - Recent emotions: {[qa.emotion for qa in qa_pairs_with_emotions[-3:]]}
            - High confidence reached: {high_confidence_reached}
            """

            agent_result = await Runner.run(orchestration_agent, orchestration_prompt)

            # Clear analyzing status now that we have results
            await send_status("idle")

            # Debug logging
            print(f"[WEBSOCKET] Orchestration result: {agent_result}")
            print(
                f"[WEBSOCKET] Last agent: {agent_result.last_agent.name if hasattr(agent_result, 'last_agent') else 'unknown'}"
            )
            print(f"[WEBSOCKET] Final output type: {type(agent_result.final_output)}")
            print(f"[WEBSOCKET] Final output: {agent_result.final_output}")

            # Store emotion data if we have it
            if emotion_data:
                print(
                    f"[WEBSOCKET] Storing QA pair with emotion: {emotion_data.get('emotion')}"
                )
                qa_pairs_with_emotions.append(
                    QAEmotionPair(
                        question=current_question if current_question else "unknown",
                        answer=user_input,
                        emotion=emotion_data.get("emotion", "unknown"),
                        confidence=emotion_data.get("confidence", 0.0),
                        negative_emotion_percentages=emotion_data.get(
                            "negative_emotion_percentages"
                        ),
                        is_direct=current_is_direct,
                    )
                )

                # Don't send result status here - only with music recommendation
                # await send_status(
                #     "result",
                #     {
                #         "mood": emotion_data["emotion"],
                #         "confidence": emotion_data["confidence"],
                #     },
                # )
            else:
                print("[WEBSOCKET] WARNING: No emotion data found in agent result!")

            # Determine which agent produced the final output (should be conversation or music agent)
            final_agent_name = (
                agent_result.last_agent.name
                if hasattr(agent_result, "last_agent")
                else None
            )
            final_output = agent_result.final_output

            # Handle based on which agent completed LAST (not emotion agent)
            if final_agent_name == "Conversation Agent":
                # Got next question
                question_data = (
                    final_output
                    if isinstance(final_output, dict)
                    else final_output.model_dump()
                )
                next_question = question_data.get("question")
                is_direct = question_data.get("is_direct", False)

                if next_question:
                    print(f"[WEBSOCKET] Asking next question: {next_question}")
                    current_question = next_question  # Track current question
                    current_is_direct = is_direct  # Track if question is direct
                    user_input = await speak_and_listen(next_question)

                    if not user_input.strip():
                        break
                else:
                    print("[WEBSOCKET] No question provided, ending")
                    break

            elif final_agent_name == "Music Agent":
                # Got music recommendation
                music_data = (
                    final_output
                    if isinstance(final_output, dict)
                    else final_output.model_dump()
                )
                music_song = music_data.get("song")

                if music_song:
                    print(f"[WEBSOCKET] Music recommendation: {music_song}")
                    # Send final emotion result with music recommendation
                    if emotion_data:
                        await send_status(
                            "result",
                            {
                                "mood": emotion_data["emotion"],
                                "confidence": emotion_data["confidence"],
                            },
                        )
                    await send_status("music_recommendation", {"music": music_song})
                break

            elif final_agent_name == "Emotion Agent":
                # Orchestration agent stopped after emotion analysis (shouldn't happen)
                # This means it didn't complete the second handoff
                print(
                    "[WEBSOCKET] WARNING: Orchestration agent only completed emotion analysis, no follow-up action"
                )
                print(
                    "[WEBSOCKET] This indicates the orchestration agent didn't perform the second handoff"
                )
                break

            else:
                print(f"[WEBSOCKET] Unknown agent result: {final_agent_name}")
                break

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
        if receive_task:
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, WebSocketDisconnect):
                pass
            except Exception as e:
                print(f"[WEBSOCKET] Error during receive_task cleanup: {e}")

        # upload session data in background
        if qa_pairs_with_emotions:
            print(f"[WEBSOCKET] Uploading {len(qa_pairs_with_emotions)} QA pairs")
            last_qa = qa_pairs_with_emotions[-1] if qa_pairs_with_emotions else None
            upload_thread = threading.Thread(
                target=upload_session_in_background,
                args=(
                    audioBytes,
                    session_id,
                    session_timestamp,
                    qa_pairs_with_emotions,
                    last_qa.emotion if last_qa else "unknown",
                    last_qa.confidence if last_qa else 0.0,
                    len(qa_pairs_with_emotions),
                    sum(1 for qa in qa_pairs_with_emotions if qa.is_direct),
                ),
                daemon=True,
            )
            upload_thread.start()
            print(f"[WEBSOCKET] Started background upload for session: {session_id}")
        else:
            print("[WEBSOCKET] No QA pairs to upload")
