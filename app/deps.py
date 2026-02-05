import os

from elevenlabs import ElevenLabs
from google.auth import default
from google.cloud import firestore, storage
from openai import OpenAI

# Clients startup and config
credentials, project = default()


firestore_client = firestore.Client()

storage_client = storage.Client()

elevenlabs_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),  # TODO look for more secure way later
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # TODO look for more secure way later
)


def get_firestore_client():
    return firestore_client


def get_storage_client():
    return storage_client


def get_elevenlabs():
    return elevenlabs_client


def get_openai_client():
    return openai_client
