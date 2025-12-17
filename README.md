# SoundScaper (I'm bad at naming things)

A small full-stack project that combines a Python backend, a TypeScript frontend, and basic Google Cloud services.

The application records microphone audio in the browser, converts speech to text, generates background music using a realtime music generation API, and allows generated songs to be streamed again from a Library page.

---

## Why I ended up with this concept:

When deciding on a project idea, the first thing I did was go through Google API endpoints to see if I found anything interesting. I have always been into making music, so I was surprised when I saw a little demo video of Lyria Realtime. Thats when I came up with this full-stack program where I can develop skills with FastAPI, Pydantic, Pytest, setting up CI/CD pipeline, Docker containers, and GCP services. As a bonus, the concept is similar to the Outer Edge prototype (Second iteration is always the best according to mythical man-month), so I believe I can make a smaller and simpler version using Google's Lyria Realtime API instead of the MyWayv audio files.

### Learning goals

- Build a Python API using FastAPI
- Use Pydantic for request validation and configuration
- Integrate external APIs in a structured way
- Run automated checks using CI
- Package and run the application using Docker
- Build a simple frontend using TypeScript and Tailwind
- Use basic GCP services for storage and persistence
- Design with Pytest testing in mind

### Concerns

- Maybe too similar to Outer Edge?
- Maybe bigger project than Dr. Alon wanted? not sure on time constraints.
- Costs of Lyria API for scaling up (should be negligible if only a study project with no users though)

---

## High-level flow

1. The user records audio using their microphone in the browser.
2. The frontend sends the audio file and music settings to the backend.
3. The backend transcribes the audio using ElevenLabs Speech to Text.
4. The backend generates music using the Google Lyria Realtime API.
5. The generated audio is saved to Google Cloud Storage.
6. Song metadata is saved in Firestore.
7. The saved song appears in a Library view and can be streamed again.

---

## Tech stack

Backend:
- Python 3.11
- FastAPI
- Pydantic

Testing:
- pytest
- pytest-mock

Frontend:
- TypeScript
- Tailwind CSS
- Vite
- MediaRecorder API

Infrastructure:
- Docker
- GitHub Actions

Google Cloud:
- Firestore for song metadata
- Cloud Storage for generated audio

External APIs:
- ElevenLabs Speech to Text
- Google Lyria Realtime music generation

---

## Data storage and streaming

Generated audio is stored as WAV files in Google Cloud Storage.

Firestore stores a document per song containing:
- song id
- creation timestamp
- transcript
- music settings
- Cloud Storage object path

For the Library view, the backend generates a signed URL for each song. The frontend streams audio directly from Cloud Storage using a standard HTML audio element.

---

## API endpoints

POST /v1/pipeline  
Runs the full pipeline from audio upload to music generation and persistence.

GET /v1/songs  
Returns a list of saved songs for the Library view.

GET /v1/songs/{song_id}  
Returns song metadata and a signed streaming URL.

---

## Configuration

Example .env file:
ELEVENLABS_API_KEY=...
LYRIA_API_KEY=...

GCP_PROJECT_ID=...
GCS_BUCKET_NAME=...

TEST_USER_ID=test-user

---

## Development

Backend:
pip install -r requirements.txt
uvicorn app.main:app --reload

Frontend:
cd web
npm install
npm run dev

---

## CI and Docker

GitHub Actions runs checks on every push and pull request.

The application is built using a multi-stage Docker build:
- one stage builds the frontend assets
- one stage runs the Python backend and serves the built frontend
