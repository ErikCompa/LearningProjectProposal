import "./styles/style.css";
import RecordButton from "./components/recordButton";
import RealtimeTranscript from "./components/realtimeTranscript.ts";
import TranscriptionList from "./components/transcriptionList.ts";
import TranscriptionModeToggle from "./components/transcriptionModeToggle.ts";
import AudioRecorder from "./audio/audioRecorder.ts";
import FirestoreService from "./services/firestoreService.ts";
import StreamingService from "./services/streamingService";
import UpdateButton from "./components/updateButton";
import STTProviderToggle from "./components/sttProviderToggle.ts";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <div>
    <h1>Mood Detection</h1>
    <div id="record-button-container"></div>
    <div id="transcription-mode-toggle-container"></div>
    <div id="stt-provider-toggle-container"></div>
    <div id="update-button-container"></div>
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

const updateButtonContainer = document.querySelector<HTMLDivElement>(
  "#update-button-container",
)!;

const sttProviderToggleContainer = document.querySelector<HTMLDivElement>(
  "#stt-provider-toggle-container",
)!;

const transcriptionListComponent = new TranscriptionList(
  transcriptionListContainer,
);

const realtimeTranscriptComponent = new RealtimeTranscript(
  realtimeTranscriptContainer,
);

const firestoreService = new FirestoreService();

const handleProcessingComplete = async (): Promise<void> => {
  const records = await firestoreService.getFromFirestore();
  transcriptionListComponent.update(records);
  transcriptionListComponent.show();
};

const handleRecordingStart = (): void => {
  const mode = transcriptionModeToggle.getStreamingMode();
  transcriptionListComponent.hide();
  if (mode === "stream") {
    realtimeTranscriptComponent.clear();
  }
};

const handleRecordingComplete = async (): Promise<void> => {
  const mode = transcriptionModeToggle.getStreamingMode();
  if (mode === "stream") {
    realtimeTranscriptComponent.clear();
  }
};

const streamingService = new StreamingService(
  (transcript, isFinal, stability) => {
    realtimeTranscriptComponent.update(transcript, isFinal, stability);
  },
  handleProcessingComplete,
  () => recordButton.setEnabled(true),
);

const audioRecorder = new AudioRecorder(
  streamingService,
  handleProcessingComplete,
);

const transcriptionModeToggle = new TranscriptionModeToggle(
  transcriptionModeToggleContainer,
  (mode) => audioRecorder.setTranscriptionMode(mode),
);

const sttProviderToggle = new STTProviderToggle(
  sttProviderToggleContainer,
  (provider) => {
    console.log("[FRONTEND] Toggle changed to:", provider);
    audioRecorder.setSTTProvider(provider);
    streamingService.setProvider(provider);
    realtimeTranscriptComponent.setProvider(provider);
  },
);

// Set initial stt provider
streamingService.setProvider(sttProviderToggle.getProvider());
audioRecorder.setSTTProvider(sttProviderToggle.getProvider());
realtimeTranscriptComponent.setProvider(sttProviderToggle.getProvider());

const recordButton = new RecordButton(
  recordButtonContainer,
  audioRecorder,
  streamingService,
  handleRecordingComplete,
  handleRecordingStart,
);

new UpdateButton(
  updateButtonContainer,
  firestoreService,
  transcriptionListComponent,
);
