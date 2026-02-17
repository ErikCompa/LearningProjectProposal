import AgentStatus from "../components/agentStatus";
import RealtimeTranscript from "../components/realtimeTranscript";
import RecordButton from "../components/recordButton";
import AudioRecorder from "../audio/audioRecorder";
import MusicRecommendation from "../components/musicRecommendation";

export class StreamingServiceHelper {
  private audioChunks: Uint8Array[] = [];
  private agentStatus: AgentStatus;
  private emotionGraph: any;
  private realtimeTranscript: RealtimeTranscript;
  private recordButton: RecordButton;
  private audioRecorder: AudioRecorder;
  private musicRecommendation: MusicRecommendation;
  private websocket: WebSocket | null = null;
  private currentAgentPartial: string = "";
  private jsonBuffer: string = "";
  private pendingQuestion: string | null = null;

  constructor(
    agentStatus: AgentStatus,
    emotionGraph: any,
    realtimeTranscript: RealtimeTranscript,
    recordButton: RecordButton,
    audioRecorder: AudioRecorder,
    musicRecommendation: MusicRecommendation,
  ) {
    this.agentStatus = agentStatus;
    this.emotionGraph = emotionGraph;
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
    this.onIntermediateResult = this.onIntermediateResult.bind(this);
    this.onAgentStream = this.onAgentStream.bind(this);
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
    if (this.pendingQuestion && this.pendingQuestion !== question) {
      return;
    }

    this.pendingQuestion = null;

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

  public onResult(): void {
    this.agentStatus.showResult();
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

  public onIntermediateResult(
    mood: string,
    confidence: number,
    negativeEmotionPercentages: Record<string, number> | null,
  ): void {
    this.emotionGraph.update(mood, confidence, negativeEmotionPercentages);
  }

  public onAgentStream(payload: any, isFinal: boolean): void {
    if (!isFinal) {
      const deltaText =
        typeof payload === "string"
          ? payload
          : (payload?.delta ?? String(payload ?? ""));

      const looksLikeJsonFragment =
        deltaText.trim().startsWith("{") ||
        this.jsonBuffer.length > 0 ||
        deltaText.includes('"question"') ||
        deltaText.includes('"song"') ||
        deltaText.includes('":');

      if (looksLikeJsonFragment) {
        this.jsonBuffer += deltaText;
        try {
          const parsed = JSON.parse(this.jsonBuffer);
          if (parsed?.question) {
            this.agentStatus.showQuestion(parsed.question);
          } else if (parsed?.song) {
            this.musicRecommendation.show(parsed.song);
            this.agentStatus.clear();
          }
        } catch (e) {
          const match = this.jsonBuffer.match(/"question"\s*:\s*"([^"]*)$/);
          if (match) {
            this.agentStatus.showQuestion(match[1]);
          }
          const sMatch = this.jsonBuffer.match(/"song"\s*:\s*"([^"]*)$/);
          if (sMatch && sMatch[1].length > 2) {
            this.musicRecommendation.show(sMatch[1]);
            this.agentStatus.clear();
          }
        }
        return;
      }

      this.currentAgentPartial += String(deltaText || "");
      console.debug(
        "[HELPER] appending delta -> currentAgentPartial:",
        this.currentAgentPartial,
      );
      this.agentStatus.showQuestion(this.currentAgentPartial);
      return;
    }

    let finalPayload: any = payload;
    console.debug(
      "[HELPER] final payload before parse:",
      finalPayload,
      "typeof:",
      typeof finalPayload,
    );
    if (typeof finalPayload === "string") {
      try {
        finalPayload = JSON.parse(finalPayload);
        console.debug("[HELPER] final payload parsed JSON:", finalPayload);
      } catch {
        console.debug("[HELPER] final payload not JSON, using raw string");
      }
    }

    this.currentAgentPartial = "";
    this.jsonBuffer = "";

    if (finalPayload && typeof finalPayload === "object") {
      if ("song" in finalPayload && finalPayload.song) {
        this.musicRecommendation.show(finalPayload.song);
        this.agentStatus.clear();
        return;
      }
      if ("question" in finalPayload && finalPayload.question) {
        this.agentStatus.showQuestion(finalPayload.question);

        this.pendingQuestion = finalPayload.question;

        if (
          "emotion" in finalPayload &&
          typeof finalPayload.confidence === "number"
        ) {
          this.onIntermediateResult(
            finalPayload.emotion,
            finalPayload.confidence,
            finalPayload.negative_emotion_percentages || null,
          );
        }

        return;
      }
    }

    this.agentStatus.showQuestion(String(finalPayload));
  }
}
