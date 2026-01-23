export default class StreamingService {
  private websocket: WebSocket | null = null;
  private onTranscriptUpdate?: (
    transcript: string,
    isFinal: boolean,
    stability: number,
  ) => void;

  constructor(
    onTranscriptUpdate?: (
      transcript: string,
      isFinal: boolean,
      stability: number,
    ) => void,
  ) {
    this.onTranscriptUpdate = onTranscriptUpdate;
  }

  public connect(): void {
    this.websocket = new WebSocket(
      "ws://localhost:8000/v1/ws/stream_process_audio/",
    );

    this.websocket.onopen = () => {
      console.log("Websocket open");
    };

    this.websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    this.websocket.onclose = (event) => {
      console.log("WebSocket connection closed:", event);
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
