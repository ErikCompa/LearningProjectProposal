import AgentStatus from "../components/agentStatus";
import RealtimeTranscript from "../components/realtimeTranscript";
import RecordButton from "../components/recordButton";
import AudioRecorder from "../audio/audioRecorder";
import MusicRecommendation from "../components/musicRecommendation";

export class StreamingServiceHelper {
  private audioChunks: Uint8Array[] = [];
  private agentStatus: AgentStatus;
  private realtimeTranscript: RealtimeTranscript;
  private recordButton: RecordButton;
  private audioRecorder: AudioRecorder;
  private musicRecommendation: MusicRecommendation;
  private websocket: WebSocket | null = null;

  constructor(
    agentStatus: AgentStatus,
    realtimeTranscript: RealtimeTranscript,
    recordButton: RecordButton,
    audioRecorder: AudioRecorder,
    musicRecommendation: MusicRecommendation,
  ) {
    this.agentStatus = agentStatus;
    this.realtimeTranscript = realtimeTranscript;
    this.recordButton = recordButton;
    this.audioRecorder = audioRecorder;
    this.musicRecommendation = musicRecommendation;

    this.onTranscriptUpdate = this.onTranscriptUpdate.bind(this);
    this.onQuestionAudio = this.onQuestionAudio.bind(this);
    this.onQuestion = this.onQuestion.bind(this);
    this.onListening = this.onListening.bind(this);
    this.onAnalyzing = this.onAnalyzing.bind(this);
    this.onResult = this.onResult.bind(this);
    this.onNoResult = this.onNoResult.bind(this);
    this.onEmptyTranscript = this.onEmptyTranscript.bind(this);
    this.onError = this.onError.bind(this);
    this.onWebSocketClosed = this.onWebSocketClosed.bind(this);
    this.onMusicRecommendation = this.onMusicRecommendation.bind(this);
  }

  public setWebSocket(websocket: WebSocket): void {
    this.websocket = websocket;
  }

  public onTranscriptUpdate(transcript: string, isFinal: boolean): void {
    this.realtimeTranscript.update(transcript, isFinal);
  }

  public onQuestionAudio(chunk: string): void {
    const bytes = new Uint8Array(atob(chunk).length);
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = atob(chunk).charCodeAt(i);
    }
    this.audioChunks.push(bytes);
  }

  public async onQuestion(question: string): Promise<void> {
    this.agentStatus.showQuestion(question);
    this.realtimeTranscript.clear();
    this.realtimeTranscript.show();
    this.recordButton.setEnabled(false);
    this.recordButton.setSessionActive(false);

    if (this.audioChunks.length > 0) {
      const totalLength = this.audioChunks.reduce(
        (sum, chunk) => sum + chunk.length,
        0,
      );
      const combined = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of this.audioChunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }

      const blob = new Blob([combined.buffer as ArrayBuffer], {
        type: "audio/mpeg",
      });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);

      const audioFinished = new Promise<void>((resolve) => {
        audio.onended = () => {
          URL.revokeObjectURL(url);
          resolve();
        };

        audio.onerror = (error) => {
          console.error("[HELPER] Audio playback error:", error);
          URL.revokeObjectURL(url);
          resolve();
        };
      });

      try {
        await audio.play();
        await audioFinished;
      } catch (error) {
        console.error("[HELPER] Error playing audio:", error);
      }

      this.audioChunks = [];
    }

    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({ type: "audio_playback_finished" }));
    } else {
      console.error("[HELPER] Cannot signal - WebSocket not open");
    }
  }

  public onListening(): void {
    this.agentStatus.showListening();
    this.recordButton.setEnabled(true);
    this.recordButton.setSessionActive(true);
  }

  public onAnalyzing(): void {
    this.agentStatus.showAnalyzing();
    this.realtimeTranscript.hide();
    this.recordButton.setEnabled(false);
    this.recordButton.setSessionActive(false);
  }

  public onResult(mood: string, confidence: number): void {
    this.agentStatus.showResult(mood, confidence);
    this.recordButton.setEnabled(false);
    this.recordButton.setSessionActive(false);
  }

  public onNoResult(message: string): void {
    this.agentStatus.showNoResult(message);
    this.recordButton.setEnabled(false);
    this.recordButton.setSessionActive(false);
  }

  public async onEmptyTranscript(message: string): Promise<void> {
    this.agentStatus.showError(message);
    this.realtimeTranscript.clear();
    this.recordButton.setEnabled(false);
    this.recordButton.setSessionActive(false);

    if (this.audioChunks.length > 0) {
      const totalLength = this.audioChunks.reduce(
        (sum, chunk) => sum + chunk.length,
        0,
      );
      const combined = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of this.audioChunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }

      const blob = new Blob([combined.buffer as ArrayBuffer], {
        type: "audio/mpeg",
      });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);

      const audioFinished = new Promise<void>((resolve) => {
        audio.onended = () => {
          URL.revokeObjectURL(url);
          resolve();
        };

        audio.onerror = (error) => {
          console.error(
            "[HELPER] Empty transcript audio playback error:",
            error,
          );
          URL.revokeObjectURL(url);
          resolve();
        };
      });

      try {
        await audio.play();
        await audioFinished;
      } catch (error) {
        console.error("[HELPER] Error playing empty transcript audio:", error);
      }

      this.audioChunks = [];
    }

    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({ type: "audio_playback_finished" }));
      console.log("[HELPER] Signaled audio playback finished for retry");
    } else {
      console.error("[HELPER] Cannot signal - WebSocket not open");
    }
  }

  public onError(message: string): void {
    this.agentStatus.showError(message);
    console.error("Agent error:", message);
    this.recordButton.setEnabled(true);
    this.recordButton.setSessionActive(false);
    this.audioRecorder.stopRecording();
  }

  public onMusicRecommendation(music: string): void {
    this.musicRecommendation.show(music);
    this.recordButton.setEnabled(true);
    this.recordButton.setSessionActive(false);
    this.audioRecorder.stopRecording();
  }

  public onWebSocketClosed(): void {
    this.recordButton.setEnabled(true);
    this.recordButton.setSessionActive(false);
    this.audioChunks = [];
  }
}
