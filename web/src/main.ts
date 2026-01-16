import "./style.css";
import Recorder from "./recorder.ts";
import RecordList from "./recordList.ts";
import RealtimeAudio from "./realtimeAudio.ts";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <div>
    <h1>Mood Detection</h1>
    <div id="recorder-container"></div>
    <div id="record-list-container"></div>
    <div id="realtime-audio-container"></div>
  </div>
`;

const recorderContainer = document.querySelector<HTMLDivElement>(
  "#recorder-container"
)!;
const recordListContainer = document.querySelector<HTMLDivElement>(
  "#record-list-container"
)!;
const realtimeAudioContainer = document.querySelector<HTMLDivElement>(
  "#realtime-audio-container"
)!;

const recordList = new RecordList(recordListContainer);
const realtimeAudio = new RealtimeAudio(realtimeAudioContainer);
new Recorder(recorderContainer, recordList, realtimeAudio);
