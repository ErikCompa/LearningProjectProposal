import "./style.css";
import RecordButton from "./components/recordButton.ts";
import RealtimeTranscript from "./components/realtimeTranscript.ts";
import TranscriptionList from "./components/transcriptionList.ts";
import TranscriptionModeToggle from "./components/transcriptionModeToggle.ts";
import AudioRecorder from "./audio/audioRecorder.ts";
import FirestoreService from "./services/firestoreService.ts";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <div>
    <h1>Mood Detection</h1>
    <div id="record-button-container"></div>
    <div id="transcription-mode-toggle-container"></div>
    <div id="realtime-transcript-container"></div>
    <div id="transcription-list-container"></div>
  </div>
`;

const recordButtonContainer = document.querySelector<HTMLDivElement>(
  "#record-button-container",
)!;

const transcriptionModeToggleContainer = document.querySelector<HTMLDivElement>(
  "#transcription-mode-toggle-container",
)!;

const realtimeTranscriptContainer = document.querySelector<HTMLDivElement>(
  "#realtime-transcript-container",
)!;

const transcriptionListContainer = document.querySelector<HTMLDivElement>(
  "#transcription-list-container",
)!;

const audioRecorder = new AudioRecorder();
const firestoreService = new FirestoreService();

const transcriptionModeToggle = new TranscriptionModeToggle(
  transcriptionModeToggleContainer,
  (mode) => audioRecorder.setTranscriptionMode(mode),
);

const transcriptionListComponent = new TranscriptionList(
  transcriptionListContainer,
);

const realtimeTranscriptComponent = new RealtimeTranscript(
  realtimeTranscriptContainer,
);

const handleRecordingComplete = async (): Promise<void> => {
  const mode = transcriptionModeToggle.getStreamingMode();

  if (mode === "stream") {
    realtimeTranscriptComponent.clear();
  }

  const records = await firestoreService.getFromFirestore();
  transcriptionListComponent.update(records);
};

const recordButton = new RecordButton(
  recordButtonContainer,
  audioRecorder,
  handleRecordingComplete,
);
