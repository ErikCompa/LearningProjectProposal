import StreamingService from "../services/streamingService";

export default class AudioRecorder {
  private isRecording: boolean = false;
  private audioContext: AudioContext = new AudioContext();
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private streamingService: StreamingService;
  private recorderNode: AudioWorkletNode | null = null;

  constructor(streamingService: StreamingService) {
    this.streamingService = streamingService;
  }

  public getRecordingStatus(): boolean {
    return this.isRecording;
  }

  public async startRecording(): Promise<void> {
    if (this.audioContext && this.audioContext.state !== "closed") {
      await this.audioContext.close();
    }
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    this.source = null;
    this.stream = null;

    this.setRecordingStatus(true);
    await this.startStreamRecording();
  }

  public async stopRecording(): Promise<void> {
    this.setRecordingStatus(false);
    await this.stopStreamRecording();
    this.audioContext = null as any;
    this.source = null;
    this.stream = null;
  }

  private setRecordingStatus(status: boolean): void {
    this.isRecording = status;
  }

  private async startStreamRecording(): Promise<void> {
    if (this.audioContext && this.audioContext.state !== "closed") {
      await this.audioContext.close();
    }
    this.audioContext = new AudioContext({ sampleRate: 16000 });

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    // get mic stream
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });

    // set context source to stream
    this.source = this.audioContext.createMediaStreamSource(this.stream);

    // add recorder worklet custom node
    await this.audioContext.audioWorklet.addModule("/recorderNode.js");
    this.recorderNode = new AudioWorkletNode(
      this.audioContext,
      "recorder-node",
    );
    this.source.connect(this.recorderNode);

    // connect to streaming service
    this.streamingService.connect();

    // handle incoming audio data from worklet
    this.recorderNode.port.onmessage = (event) => {
      const workletInput = event.data[0];
      if (workletInput) {
        const float32Data = new Float32Array(workletInput);
        const int16Data = this.linear16PCM(float32Data);
        this.streamingService.processStreamingAudio(int16Data);
      }
    };
    this.audioContext.resume();
  }

  private async stopStreamRecording(): Promise<void> {
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.recorderNode) {
      this.recorderNode.disconnect();
      this.recorderNode.port.onmessage = null;
      this.recorderNode = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      await this.audioContext.close();
    }
    this.streamingService.disconnect();
  }

  // convert float32 to linear16
  private linear16PCM(float32Array: Float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      let s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16Array;
  }
}
