import "./styles/style.css";
import RecordButton from "./components/recordButton";
import RealtimeTranscript from "./components/realtimeTranscript.ts";
import AgentStatus from "./components/agentStatus.ts";
import AudioRecorder from "./audio/audioRecorder.ts";
import StreamingService from "./services/streamingService";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <div>
    <h1>Mood Detection Agent</h1>
    <div id="record-button-container"></div>
    <div id="agent-status-container"></div>
    <div id="realtime-transcript-container"></div>
  </div>
`;

const recordButtonContainer = document.querySelector<HTMLDivElement>(
  "#record-button-container",
)!;

const agentStatusContainer = document.querySelector<HTMLDivElement>(
  "#agent-status-container",
)!;

const realtimeTranscriptContainer = document.querySelector<HTMLDivElement>(
  "#realtime-transcript-container",
)!;

const agentStatus = new AgentStatus(agentStatusContainer);
const realtimeTranscript = new RealtimeTranscript(realtimeTranscriptContainer);

const streamingService = new StreamingService(
  // onTranscriptUpdate
  (transcript: string, isFinal: boolean) => {
    realtimeTranscript.update(transcript, isFinal);
  },
  // onQuestion
  (question: string) => {
    agentStatus.showQuestion(question);
    realtimeTranscript.clear();
    realtimeTranscript.show();
    recordButton.setEnabled(true);
    // Keep ball grey during 5 second reading period
    recordButton.setSessionActive(false);
  },
  // onListening
  () => {
    agentStatus.showListening();
    // NOW turn ball red - we're actually recording the answer
    recordButton.setSessionActive(true);
  },
  // onAnalyzing
  () => {
    agentStatus.showAnalyzing();
    realtimeTranscript.hide();
    recordButton.setEnabled(false);
    // Turn ball grey during analysis
    recordButton.setSessionActive(false);
  },
  // onResult
  (mood: string, confidence: number) => {
    agentStatus.showResult(mood, confidence);
    recordButton.setEnabled(true);
    recordButton.setSessionActive(false);
    audioRecorder.stopRecording();
  },
  // onNoResult
  (message: string) => {
    agentStatus.showNoResult(message);
    recordButton.setEnabled(true);
    recordButton.setSessionActive(false);
    audioRecorder.stopRecording();
  },
  // onError
  (message: string) => {
    agentStatus.showError(message);
    console.error("Agent error:", message);
    alert(`Error: ${message}`);
    recordButton.setEnabled(true);
    recordButton.setSessionActive(false);
    audioRecorder.stopRecording();
  },
  // onWebSocketClosed
  () => {
    recordButton.setEnabled(true);
    console.log("WebSocket connection closed");
  },
);

const audioRecorder = new AudioRecorder(streamingService);

const recordButton = new RecordButton(recordButtonContainer, audioRecorder);
