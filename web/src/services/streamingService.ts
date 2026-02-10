const WS_URL = import.meta.env.VITE_AGENT_URL;

export default class StreamingService {
  private websocket: WebSocket | null = null;
  private onTranscriptUpdate?: (transcript: string, isFinal: boolean) => void;
  private onQuestionAudio?: (chunk: any) => void;
  private onQuestion?: (question: string) => void;
  private onListening?: () => void;
  private onAnalyzing?: () => void;
  private onIdle?: () => void;
  private onResult?: (mood: string, confidence: number) => void;
  private onNoResult?: (message: string) => void;
  private onEmptyTranscript?: (message: string) => void;
  private onError?: (message: string) => void;
  private onWebSocketClosed?: () => void;
  private onMusicRecommendation?: (music: string) => void;
  private helper: any = null;

  constructor(
    onTranscriptUpdate?: (transcript: string, isFinal: boolean) => void,
    onQuestionAudio?: (chunk: any) => void,
    onQuestion?: (question: string) => void,
    onListening?: () => void,
    onAnalyzing?: () => void,
    onIdle?: () => void,
    onResult?: (mood: string, confidence: number) => void,
    onNoResult?: (message: string) => void,
    onEmptyTranscript?: (message: string) => void,
    onError?: (message: string) => void,
    onWebSocketClosed?: () => void,
    onMusicRecommendation?: (music: string) => void,
  ) {
    this.onTranscriptUpdate = onTranscriptUpdate;
    this.onQuestionAudio = onQuestionAudio;
    this.onQuestion = onQuestion;
    this.onListening = onListening;
    this.onAnalyzing = onAnalyzing;
    this.onIdle = onIdle;
    this.onResult = onResult;
    this.onNoResult = onNoResult;
    this.onEmptyTranscript = onEmptyTranscript;
    this.onError = onError;
    this.onWebSocketClosed = onWebSocketClosed;
    this.onMusicRecommendation = onMusicRecommendation;
  }

  public setHelper(helper: any): void {
    this.helper = helper;
  }

  public connect(): void {
    this.websocket = new WebSocket(WS_URL);

    this.websocket.onopen = () => {
      console.log("WebSocket connection opened");
      if (this.helper) {
        this.helper.setWebSocket(this.websocket);
      }
    };

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "transcript":
            if (this.onTranscriptUpdate) {
              this.onTranscriptUpdate(data.transcript, data.is_final);
            }
            break;
          case "question_audio_base_64":
            if (this.onQuestionAudio) {
              this.onQuestionAudio(data.chunk);
            }
            break;
          case "question":
            if (this.onQuestion) {
              this.onQuestion(data.text);
            }
            break;
          case "listening":
            if (this.onListening) {
              this.onListening();
            }
            break;
          case "analyzing":
            if (this.onAnalyzing) {
              this.onAnalyzing();
            }
            break;
          case "idle":
            if (this.onIdle) {
              this.onIdle();
            }
            break;
          case "result":
            if (this.onResult) {
              this.onResult(data.mood, data.confidence);
            }
            break;
          case "no_result":
            if (this.onNoResult) {
              this.onNoResult(data.message);
            }
            break;
          case "empty_transcript":
            if (this.onEmptyTranscript) {
              this.onEmptyTranscript(data.message);
            }
            break;
          case "error":
            if (this.onError) {
              this.onError(data.message);
            }
            break;
          case "music_recommendation":
            if (this.onMusicRecommendation) {
              this.onMusicRecommendation(data.music);
            }
            break;
        }
      } catch (error) {
        if (this.onError) {
          this.onError(`Error parsing message: ${error}`);
        }
      }
    };

    this.websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      if (this.websocket && this.websocket.readyState !== WebSocket.CLOSED) {
        if (this.onError) {
          this.onError("WebSocket connection error");
        }
      }
    };

    this.websocket.onclose = () => {
      console.log("WebSocket connection closed");
      if (this.onWebSocketClosed) {
        this.onWebSocketClosed();
      }
    };
  }

  public disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  public processStreamingAudio(data: Int16Array): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(data.buffer);
    }
  }

  public isWebSocketOpen(): boolean {
    return this.websocket?.readyState === WebSocket.OPEN;
  }

  public getWebSocket(): WebSocket | null {
    return this.websocket;
  }
}
