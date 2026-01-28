const WS_URL = import.meta.env.VITE_WS_URL;

export default class StreamingService {
  private websocket: WebSocket | null = null;
  private onTranscriptUpdate?: (
    transcript: string,
    isFinal: boolean,
    stability: number,
  ) => void;
  private onProcessingComplete?: () => void;

  constructor(
    onTranscriptUpdate?: (
      transcript: string,
      isFinal: boolean,
      stability: number,
    ) => void,
    onProcessingComplete?: () => void,
  ) {
    this.onTranscriptUpdate = onTranscriptUpdate;
    this.onProcessingComplete = onProcessingComplete;
  }

  public connect(): void {
    this.websocket = new WebSocket(WS_URL);

    this.websocket.onclose = () => {
      if (this.onProcessingComplete) {
        this.onProcessingComplete();
      }
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
      this.websocket = null;
    }
  }

  public processStreamingAudio(data: Int16Array): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(data.buffer);
    }
  }
}
