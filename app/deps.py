import os

from elevenlabs import ElevenLabs
from google import genai
from google.auth import default
from google.cloud import firestore, storage
from google.cloud.speech_v2 import SpeechClient  # type: ignore

# Clients startup and config
credentials, project = default()
gemini_client = genai.Client(
    vertexai=True,  # vertex for ADC so there are no keys
    project=project,
    location="global",
    credentials=credentials,
)

speech_v2_client = SpeechClient(
    client_options={"api_endpoint": "us-speech.googleapis.com"}
)

firestore_client = firestore.Client()

storage_client = storage.Client()

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),  # TODO look for more secure way later
)


# Export all clients
def get_gemini_client():
    return gemini_client


def get_speech_v2_client():
    return speech_v2_client


def get_firestore_client():
    return firestore_client


def get_project_id():
    return project


def get_storage_client():
    return storage_client


def get_elevenlabs():
    return elevenlabs
