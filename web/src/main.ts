import "./styles/style.css";
import RecordButton from "./components/recordButton";
import RealtimeTranscript from "./components/realtimeTranscript.ts";
import AgentStatus from "./components/agentStatus.ts";
import AudioRecorder from "./audio/audioRecorder.ts";
import StreamingService from "./services/streamingService";
import { StreamingServiceHelper } from "./services/streamingServiceHelper";
import MusicRecommendation from "./components/musicRecommendation";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <div>
    <h1>Mood Detection Agent</h1>
    <div id="record-button-container"></div>
    <div id="agent-status-container"></div>
    <div id="realtime-transcript-container"></div>
    <div id="music-recommendation-container"></div>
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

const musicRecommendationContainer = document.querySelector<HTMLDivElement>(
  "#music-recommendation-container",
)!;

const agentStatus = new AgentStatus(agentStatusContainer);
const realtimeTranscript = new RealtimeTranscript(realtimeTranscriptContainer);
const musicRecommendation = new MusicRecommendation(
  musicRecommendationContainer,
);

const helper = new StreamingServiceHelper(
  agentStatus,
  realtimeTranscript,
  null as any,
  null as any,
  musicRecommendation,
);

const streamingService = new StreamingService(
  helper.onTranscriptUpdate,
  helper.onQuestionAudio,
  helper.onQuestion,
  helper.onListening,
  helper.onAnalyzing,
  helper.onResult,
  helper.onNoResult,
  helper.onError,
  helper.onWebSocketClosed,
  helper.onMusicRecommendation,
);

streamingService.setHelper(helper);

const audioRecorder = new AudioRecorder(streamingService);

const recordButton = new RecordButton(recordButtonContainer, audioRecorder);

helper["recordButton"] = recordButton;
helper["audioRecorder"] = audioRecorder;
