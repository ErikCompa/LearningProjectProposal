# Learning Project - Speech to Text

Minimal web application that records microphone audio, transcribes speech using Google Speech_v2, and analyzes the mood of the transcript using a Google LLM (Gemini), then saves transcript and analysis Firestore and audio on Bucket.

---

## What the application does

1. Records microphone audio in the browser.
2. Sends/streams the audio to Python backend.
3. Transcribes speech using Google Speech_v2.
4. Sends the transcript to Gemini for mood analysis.
5. Stores the transcript and mood result in Firestore.
6. Stores the audio file in a Google Bucket.
7. Returns the transcript and mood to the browser for display.

---

## Mood analysis

The mood analysis step uses an LLM to classify the overall emotional tone of the transcript.

The model returns structured JSON that includes:

- mood labels and their (normalized) probabilities
- an overall confidence score between 0 and 1
- optional evidence phrases extracted from the transcript

---

## Technology overview

Backend:

- Python

- FastAPI

- Pydantic

- Pydub

Frontend:

- TypeScript

- Web Audio API

Cloud services:

- Google Speech_v2

- Gemini (LLM)

- Firestore

-Google Bucket

Infrastructure:

- Docker

- GitHub Actions

- Google Artifact Registry

- Pytest

- Ruff

---

## Local testing (production build):

Test the full application with static files served by FastAPI:

```bash
cd web/
npm run build
cp -r dist ../app/static
cd ..
uvicorn app.main:app
```

or run ./rebuild.sh from root.

Visit `http://localhost:8000` to access the complete application.

---

## GCP setup

This application requires the following Google Cloud services:

1. **Google Speech_v2**
2. **Gemini API** (via Vertex AI)
3. **Cloud Firestore**
4. **Google Bucket**
5. **Google Artifact Registry**
6. **Workload Identity Federation**

#### Prerequisites

- A Google Cloud Platform account
- A GCP project with billing enabled
- The `gcloud` CLI installed
- The 6 services above enabled and configured on the project

#### Authentication using Application Default Credentials (ADC)

This application uses Application Default Credentials (ADC) to authenticate with Google Cloud services. ADC automatically finds credentials based on the application environment.
