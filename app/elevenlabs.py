import asyncio
import base64

from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    RealtimeAudioOptions,
    RealtimeEvents,
    VoiceSettings,
)
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from app.deps import get_elevenlabs

STT_MODEL_ID = "scribe_v2_realtime"
STT_AUDIO_FORMAT = AudioFormat.PCM_16000
STT_SAMPLE_RATE = 16000
VAD_SILENCE_THRESHOLD_SECS = 3.0
VAD_THRESHOLD = 0.4
MIN_SPEECH_DURATION_MS = 100
MIN_SILENCE_DURATION_MS = 100

TTS_VOICE_ID = "I3MrSgiotopLY33bjEX7"  # Yaron, Erik: "VWoIQlDpnFjY9kfJ11dz", Adam: "pNInz6obpgDQGcFmaJgB"
TTS_OUTPUT_FORMAT = "mp3_22050_32"
TTS_MODEL_ID = "eleven_multilingual_v2"


async def stt_elevenlabs_session(
    audio_queue, res_queue, answer_transcript_container, answer_ready, websocket
):
    print("[STT] Starting ElevenLabs STT session")
    elevenlabs = get_elevenlabs()
    stop = asyncio.Event()

    connection = await elevenlabs.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id=STT_MODEL_ID,
            audio_format=STT_AUDIO_FORMAT,
            sample_rate=STT_SAMPLE_RATE,
            include_timestamps=True,
            commit_strategy=CommitStrategy.VAD,
            vad_silence_threshold_secs=VAD_SILENCE_THRESHOLD_SECS,
            vad_threshold=VAD_THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
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

                # rate limit to prevent elevenlabs error
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


async def tts_elevenlabs_session(text: str, websocket: WebSocket):
    print(f"[TTS] Sending text to ElevenLabs TTS: {text}")
    try:
        response = get_elevenlabs().text_to_speech.stream(
            voice_id=TTS_VOICE_ID,
            output_format=TTS_OUTPUT_FORMAT,
            text=text,
            model_id=TTS_MODEL_ID,
            voice_settings=VoiceSettings(
                stability=1.0,
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
        print(f"[TTS] Error during ElevenLabs TTS: {e}")
    await asyncio.sleep(0.1)
