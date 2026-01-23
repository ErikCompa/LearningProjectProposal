import queue
import threading
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.auth import default
from google.cloud import firestore
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from .models import Mood, Transcript

# clients startup
credentials, project = default()
gemini_client = genai.Client(
    vertexai=True,  # vertex for ADC so there are no keys
    project=project,
    location="us-central1",
    credentials=credentials,
)

app = FastAPI()

speech_client = SpeechClient(
    client_options={"api_endpoint": "us-speech.googleapis.com"}
)

db = firestore.Client()


# CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# websocket for streaming audio processing
# https://docs.cloud.google.com/speech-to-text/docs/v1/transcribe-streaming-audio#speech-streaming-recognize-python
# https://docs.python.org/3/library/threading.html
# https://fastapi.tiangolo.com/advanced/websockets/#create-a-websocket
@app.websocket("/v1/ws/stream_process_audio/")
async def websocket_stream_process_audio(websocket: WebSocket):
    # configure google stt streaming
    config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            audio_channel_count=1,
        ),
        language_codes=["en-US"],
        model="long",  # chirp3 doesnt support interim results
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=False,
        ),
    )
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True,
        ),
    )

    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{project}/locations/us/recognizers/mood",
        streaming_config=streaming_config,
    )

    # queues for audio and results
    audio_queue = queue.Queue()
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
                yield config_request
                while not stop.is_set():
                    if (
                        time.time() - start > 240
                    ):  # 4 min limit, then restart stream recognize
                        print("Restarting STT stream due to time limit\n\n\n\n\n\n\n")
                        break
                    chunk = audio_queue.get()
                    if chunk is None:
                        break
                    yield cloud_speech.StreamingRecognizeRequest(
                        audio=chunk
                    )  # yield audio chunks to google stt

            # call google stt
            responses: cloud_speech.StreamingRecognizeResponse = (
                speech_client.streaming_recognize(  # returns iterator
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
        uploadResult = await uploadToFirestoreStep(transcript, mood)
    return uploadResult


# single endpoint to batch transcribe, analyze mood, and upload to firestore
@app.post("/v1/batch_process_audio/")
async def batch_process_audio(file: Annotated[UploadFile, File(...)]):
    transcript = await batchTranscriptionStep(file)
    mood = await moodAnalysisStep(transcript)
    uploadResult = await uploadToFirestoreStep(transcript, mood)
    return uploadResult


# get all from firestore endpoint
@app.get("/v1/firestore_get/")
async def get_from_firestore():
    rows = db.collection("record").stream()
    records = []
    for row in rows:
        records.append(row.to_dict())
    return records


# Mount static files
# https://fastapi.tiangolo.com/tutorial/static-files/
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


async def batchTranscriptionStep(file: UploadFile) -> Transcript:
    # Transcription step
    # file check
    if file.filename is None or file.filename == "":
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.size is None or file.size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    data = await file.read()

    config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            audio_channel_count=1,
        ),
        language_codes=["en-US"],
        model="chirp_3",
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project}/locations/us/recognizers/mood",
        config=config,
        content=data,
    )

    # Transcribes the audio into text
    response = speech_client.recognize(request=request)

    # transcript in pydantic model
    transcript = Transcript(
        text=response.results[0].alternatives[0].transcript,
    )
    print(f"Transcript step done: {transcript}")
    return transcript


async def moodAnalysisStep(transcript: Transcript) -> Mood:
    # Mood analysis step
    if not transcript.text:
        raise HTTPException(status_code=400, detail="Transcript text is empty.")

    propmt = (
        "Analyze the following transcript and determine the overall mood of the user. "
        "Give a confidence score between 0.0 and 1.0 and evidence with explanations."
    )

    transcript_data = transcript.model_dump()

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[propmt, str(transcript_data)],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Mood.model_json_schema(),
        },
    )

    if not response.text:
        raise HTTPException(status_code=400, detail="Mood analysis failed.")

    mood = Mood.model_validate_json(response.text)
    print(f"Mood analysis step done: {mood}")
    return mood


async def uploadToFirestoreStep(transcript: Transcript, mood: Mood):
    # Upload to Firestore step
    if not transcript or not mood:
        raise HTTPException(status_code=400, detail="No response data provided.")

    # insert response into firestore
    doc_ref = db.collection("record")
    write_res = doc_ref.document(transcript.uid).set(
        {
            "created_at": firestore.SERVER_TIMESTAMP,
            "mood": {
                "confidence": mood.confidence,
                "evidence": mood.evidence,
                "mood": mood.mood,
            },
            "transcript": transcript.text,
            "uid": transcript.uid,
        }
    )

    if not write_res.update_time:
        raise HTTPException(status_code=400, detail="Failed to upload to Firestore.")

    print(f"Upload to Firestore step done: {transcript.uid}")
    return {"status": 200, "uid": transcript.uid}
