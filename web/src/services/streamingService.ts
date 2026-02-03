const WS_URL = import.meta.env.VITE_WS_URL;

export default class StreamingService {
  private websocket: WebSocket | null = null;
  private onTranscriptUpdate?: (
    transcript: string,
    isFinal: boolean,
    stability: number,
  ) => void;
  private onProcessingComplete?: () => void;
  private onWebSocketClosed?: () => void;
  private provider: string = "google";

  constructor(
    onTranscriptUpdate?: (
      transcript: string,
      isFinal: boolean,
      stability: number,
    ) => void,
    onProcessingComplete?: () => void,
    onWebSocketClosed?: () => void,
  ) {
    this.onTranscriptUpdate = onTranscriptUpdate;
    this.onProcessingComplete = onProcessingComplete;
    this.onWebSocketClosed = onWebSocketClosed;
  }

  public setProvider(provider: string): void {
    console.log("[STREAMING SERVICE] Setting provider to:", provider);
    this.provider = provider;
  }

  public connect(): void {
    console.log("[STREAMING SERVICE] Connecting with provider:", this.provider);
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
      if (this.websocket) {
        console.log("[FRONTEND] Sending provider config:", this.provider);
        this.websocket.send(JSON.stringify({ provider: this.provider }));
      }
    };

    this.websocket.onclose = () => {
      if (this.onProcessingComplete) {
        this.onProcessingComplete();
      }
      if (this.onWebSocketClosed) {
        this.onWebSocketClosed();
      }
      this.websocket = null;
    };

    this.websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (this.onTranscriptUpdate && data.transcript) {
        this.onTranscriptUpdate(
          data.transcript,
          data.is_final || false,
          data.stability || 0.0,
        );
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
