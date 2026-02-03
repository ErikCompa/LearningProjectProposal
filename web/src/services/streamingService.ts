const WS_URL = import.meta.env.VITE_AGENT_URL;

export default class StreamingService {
  private websocket: WebSocket | null = null;
  private onTranscriptUpdate?: (transcript: string, isFinal: boolean) => void;
  private onQuestion?: (question: string) => void;
  private onListening?: () => void;
  private onAnalyzing?: () => void;
  private onResult?: (mood: string, confidence: number) => void;
  private onNoResult?: (message: string) => void;
  private onError?: (message: string) => void;
  private onWebSocketClosed?: () => void;

  constructor(
    onTranscriptUpdate: (transcript: string, isFinal: boolean) => void,
    onQuestion: (question: string) => void,
    onListening: () => void,
    onAnalyzing: () => void,
    onResult: (mood: string, confidence: number) => void,
    onNoResult: (message: string) => void,
    onError: (message: string) => void,
    onWebSocketClosed: () => void,
  ) {
    this.onTranscriptUpdate = onTranscriptUpdate;
    this.onQuestion = onQuestion;
    this.onListening = onListening;
    this.onAnalyzing = onAnalyzing;
    this.onResult = onResult;
    this.onNoResult = onNoResult;
    this.onError = onError;
    this.onWebSocketClosed = onWebSocketClosed;
  }

  public connect(): void {
    if (
      this.websocket &&
      (this.websocket.readyState === WebSocket.OPEN ||
        this.websocket.readyState === WebSocket.CONNECTING)
    ) {
      this.websocket.close();
      return;
    }
    this.websocket = new WebSocket(WS_URL);

    this.websocket.onopen = () => {
      console.log("WebSocket connected");
    };

    this.websocket.onclose = () => {
      if (this.onWebSocketClosed) {
        this.onWebSocketClosed();
      }
      this.websocket = null;
    };

    this.websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      if (this.onError) {
        this.onError("WebSocket connection error");
      }
    };

    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "transcript":
          if (this.onTranscriptUpdate) {
            this.onTranscriptUpdate(data.transcript, data.is_final || false);
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
        case "error":
          if (this.onError) {
            this.onError(data.message);
          }
          break;
        default:
          console.warn("Unknown message type:", data.type);
      }
    };
  }

  public disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
    }
  }

  public processStreamingAudio(data: Int16Array): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(data.buffer);
    }
  }

  public isWebSocketOpen(): boolean {
    return !!this.websocket && this.websocket.readyState === WebSocket.OPEN;
  }
}
