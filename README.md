# STT-Mood

Minimal web application that records microphone audio, transcribes speech using Google Cloud Speech-to-Text, and analyzes the mood of the transcript using a Google LLM (Gemini), then saves it on Firestore.

---

## What the application does

1. Records microphone audio in the browser.
2. Uploads the audio to a Python backend.
3. Transcribes speech using Google Cloud Speech-to-Text.
4. Sends the transcript to Gemini for mood analysis.
5. Stores the transcript and mood result in Firestore.
6. Returns the transcript and mood to the browser for display.

---

## Mood analysis

The mood analysis step uses a large language model to classify the overall emotional tone of the transcript.

The model returns structured JSON that includes:

- a mood label
- a confidence score between 0 and 1
- optional evidence phrases extracted from the transcript

---

## Technology overview

Backend:

- Python

- FastAPI

- Pydantic

Frontend:

- TypeScript

- Minimal UI using browser MediaRecorder

Cloud services:

- Google Cloud Speech-to-Text

- Gemini (LLM)

- Firestore

Infrastructure:

- Docker

- GitHub Actions

- Google Artifact Registry

- Pytest

- Ruff

---

## Development

### Development workflow (separate frontend/backend):

**Terminal 1 - Frontend dev server:**

```bash
cd web/
npm install
npm run dev
```

**Terminal 2 - Backend API (from project root):**

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend runs at `http://localhost:5173`, backend at `http://localhost:8000`. CORS is configured to allow cross-origin requests between them.

### Local testing (production build):

Test the full application with static files served by FastAPI:

```bash
cd web/
npm run build
cp -r dist ../app/static
cd ..
uvicorn app.main:app
```

Visit `http://localhost:8000` to access the complete application.

### GCP setup

This application requires the following Google Cloud services:

1. **Google Cloud Speech-to-Text API**
2. **Gemini API** (via Vertex AI)
3. **Cloud Firestore**
4. **Google Artifact Registry**
5. **Workload Identity Federation**

#### Prerequisites

- A Google Cloud Platform account
- A GCP project with billing enabled
- The `gcloud` CLI installed
- The 5 services above enabled and configured on the project

#### Authentication using Application Default Credentials (ADC)

This application uses Application Default Credentials (ADC) to authenticate with Google Cloud services. ADC automatically finds credentials based on the application environment.
