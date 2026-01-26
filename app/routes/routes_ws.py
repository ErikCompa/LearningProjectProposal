import queue
import threading
import time

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from google.cloud.speech_v2.types import cloud_speech

from app.deps import get_speech_v2_client
from app.models import Transcript
from app.services import (
    moodAnalysisStep,
    uploadToFirestoreStep,
)
from app.speech_config import get_streaming_config_request

router = APIRouter(tags=["ws"])


# WebSocket for realtime audio transcription
@router.websocket("/v1/ws/stream_process_audio/")
async def websocket_stream_process_audio(websocket: WebSocket):
    # queues for audio and results
    audio_queue = (
        queue.Queue()
    )  # audio chunks from websocket, get consumed by stt thread
    bucket_audio_queue = queue.Queue()  # full audio chunks for upload later
    res_queue = queue.Queue()
    final_results_queue = queue.Queue()
    stop = threading.Event()

    # open connection
    await websocket.accept()
    print("WebSocket connection accepted")

    def stt_thread():
        while not stop.is_set():
            start = time.time()

            def requests():
                yield get_streaming_config_request()
                while not stop.is_set():
                    if (
                        time.time() - start > 240
                    ):  # 4 min limit, then restart stream recognize
                        print("Restarting STT stream due to time limit\n\n\n\n\n\n\n")
                        break
                    chunk = audio_queue.get()
                    bucket_audio_queue.put(chunk)  # save for upload later
                    if chunk is None:
                        break
                    yield cloud_speech.StreamingRecognizeRequest(
                        audio=chunk
                    )  # yield audio chunks to google stt

            # call google stt
            responses: cloud_speech.StreamingRecognizeResponse = (
                get_speech_v2_client().streaming_recognize(  # returns iterator
                    requests=requests()
                )
            )

            # process responses
            try:
                for response in responses:  # iterator blocks thread if no response
                    for result in response.results:
                        print("STT Result received:", result)
                        transcript_text = result.alternatives[0].transcript
                        res_queue.put(
                            {
                                "transcript": transcript_text,
                                "is_final": result.is_final,
                                "stability": result.stability,
                            }
                        )
            except Exception as e:
                print("Error in STT thread:", e)

    # run blocking thread for stt
    threading.Thread(target=stt_thread, daemon=True).start()

    try:
        while True:
            # receive audio data from websocket
            data = await websocket.receive_bytes()
            # print(f"Received audio data of length: {len(data)}")

            # put audio into q
            MAX_CHUNK_SIZE = 25600
            for i in range(0, len(data), MAX_CHUNK_SIZE):
                chunk = data[i : i + MAX_CHUNK_SIZE]
                audio_queue.put(chunk)

            # read from results q
            while not res_queue.empty():
                res = res_queue.get()
                if res["is_final"]:
                    final_results_queue.put(res)
                await websocket.send_json(res)
    except WebSocketDisconnect as e:
        print("websocket disconnected:", e)
    except Exception as e:
        print("error during websocket communication:", e)
    finally:
        stop.set()
        audio_queue.put(None)
        # get full transcript from results q
        full_transcript = ""
        while not final_results_queue.empty():
            res = final_results_queue.get()
            full_transcript += res["transcript"] + ". "

        transcript = Transcript(
            text=full_transcript,
        )
        mood = await moodAnalysisStep(transcript)
        # TODO bucket_audio_queue to mp3(?) and upload to bucket later
        uploadResult = await uploadToFirestoreStep(transcript, mood)
    return uploadResult
