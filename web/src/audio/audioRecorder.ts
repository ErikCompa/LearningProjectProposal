import type { TranscriptionMode } from "./types";
import BatchService from "../services/batchService";
import StreamingService from "../services/streamingService";

export default class AudioRecorder {
  private isRecording: boolean = false;
  private audioContext: AudioContext = new AudioContext();
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private lin16buffer: Int16Array = new Int16Array();
  private batchService: BatchService;
  private streamingService: StreamingService;
  private transcriptionMode: TranscriptionMode = "stream";
  private onProcessingComplete?: () => void;
  private recorderNode: AudioWorkletNode | null = null;

  constructor(
    onTranscriptUpdate?: (
      transcript: string,
      isFinal: boolean,
      stability: number,
    ) => void,
    onProcessingComplete?: () => void,
  ) {
    this.onProcessingComplete = onProcessingComplete;
    this.batchService = new BatchService();
    this.streamingService = new StreamingService(
      onTranscriptUpdate,
      onProcessingComplete,
    );
  }

  public getRecordingStatus(): boolean {
    return this.isRecording;
  }

  public setTranscriptionMode(mode: TranscriptionMode): void {
    this.transcriptionMode = mode;
  }

  public async startRecording(): Promise<void> {
    if (this.audioContext && this.audioContext.state !== "closed") {
      await this.audioContext.close();
    }
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    this.source = null;
    this.stream = null;
    this.lin16buffer = new Int16Array();

    this.setRecordingStatus(true);
    if (this.transcriptionMode === "stream") {
      await this.startStreamRecording();
    } else {
      await this.startBatchRecording();
    }
  }

  public async stopRecording(): Promise<void> {
    this.setRecordingStatus(false);
    if (this.transcriptionMode === "stream") {
      await this.stopStreamRecording();
    } else {
      await this.stopBatchRecording();
    }
    this.audioContext = null as any;
    this.source = null;
    this.stream = null;
    this.lin16buffer = new Int16Array();
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

  private async startBatchRecording(): Promise<void> {
    // new audio context
    this.audioContext = new AudioContext({ sampleRate: 16000 });

    // get mic stream
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });

    // set context source to stream
    this.source = this.audioContext.createMediaStreamSource(this.stream);

    // add recorder worklet custom node
    await this.audioContext.audioWorklet.addModule("/recorderNode.js");
    const recorderNode = new AudioWorkletNode(
      this.audioContext,
      "recorder-node",
    );
    this.source.connect(recorderNode);

    // initialize buffer
    this.lin16buffer = new Int16Array();

    // handle incoming audio data from worklet
    recorderNode.port.onmessage = (event) => {
      const workletInput = event.data[0];
      this.lin16buffer = new Int16Array([
        ...this.lin16buffer,
        ...this.linear16PCM(workletInput),
      ]);
    };
    this.audioContext.resume();
  }

  private async stopBatchRecording(): Promise<void> {
    // stop stream and context
    this.stream!.getTracks().forEach((track) => track.stop());
    this.audioContext.close();

    // call batchService with lin16buffer
    await this.batchService.processBatchAudio(this.lin16buffer);

    if (this.onProcessingComplete) {
      this.onProcessingComplete();
    }

    this.lin16buffer = new Int16Array();
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
