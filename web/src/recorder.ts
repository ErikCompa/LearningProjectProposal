import type RealtimeAudio from "./realtimeAudio.ts";
import RecordList from "./recordList.ts";
import Toggle from "./toggle.ts";

// firestore collection structure
interface FirestoreRecord {
  uid: string;
  transcript: string;
  transcript_confidence: number;
  mood: {
    mood: string;
    confidence: number;
    evidence?: string[] | null;
  };
  created_at: string;
}

export default class Recorder {
  private isRecording: boolean = false;
  private button: HTMLButtonElement;
  private recordList: RecordList;
  public records: FirestoreRecord[] = [];
  private websocket: WebSocket | null = null;
  private streaming: boolean = true;
  private realtimeAudio: RealtimeAudio | null = null;
  private toggle_streaming: Toggle;
  private audioContext: AudioContext = new AudioContext();
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private pcmData: Float32Array[] = [];
  private lin16buffer: Int16Array = new Int16Array();

  constructor(
    container: HTMLElement,
    toggle_streaming: Toggle,
    recordList: RecordList,
    realtimeAudio: RealtimeAudio,
  ) {
    this.recordList = recordList;
    this.realtimeAudio = realtimeAudio;
    this.toggle_streaming = toggle_streaming;
    this.button = document.createElement("button");
    this.button.className = "recorder-button";
    this.button.addEventListener("click", () => this.toggle());
    container.appendChild(this.button);
  }

  // Button toggle handler
  private async toggle(): Promise<void> {
    this.streaming = this.toggle_streaming.getStreamingMode();
    if (!this.streaming) {
      if (this.isRecording) {
        this.stop();
      } else {
        this.start();
      }
    } else {
      // streaming mode
      if (this.isRecording) {
        await this.stream_stop();
      } else {
        await this.stream_start();
      }
    }
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

  private async stream_start(): Promise<void> {
    // start websocket
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
      console.log("WebSocket message received:", event.data);
      const data = JSON.parse(event.data);
      if (data.transcript) {
        this.realtimeAudio?.update(
          data.transcript,
          data.is_final,
          data.stability,
        );
      }
    };

    // remove list if present
    this.recordList?.clear();
    this.realtimeAudio?.clear();

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    this.source = this.audioContext.createMediaStreamSource(this.stream);

    await this.audioContext.audioWorklet.addModule("recorderNode.js");
    const recorderNode = new AudioWorkletNode(
      this.audioContext,
      "recorder-node",
    );
    this.source.connect(recorderNode);

    this.pcmData = [];
    recorderNode.port.onmessage = (event) => {
      const workletInput = event.data[0];
      this.pcmData.push(workletInput);
      const lin16buffer = this.linear16PCM(workletInput);

      // send to websocket
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(lin16buffer.buffer);
      }
    };
    this.audioContext.resume();
    this.isRecording = true;
    await this.updateUI();
    console.log("Recording started");
  }

  // stop recording
  private async stream_stop(): Promise<void> {
    this.stream!.getTracks().forEach((track) => track.stop()); // Stop the microphone track
    this.audioContext.close(); // Close the audio context

    // Combine pcmChunks into a single Float32Array
    const totalLength = this.pcmData.reduce(
      (acc, chunk) => acc + chunk.length,
      0,
    );
    const pcmData = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.pcmData) {
      pcmData.set(chunk, offset);
      offset += chunk.length;
    }

    // give few sec for final transcription to come back
    setTimeout(() => {
      if (this.websocket) {
        this.websocket.close();
        this.websocket = null;
        this.updateUI();
      }
    }, 5000);

    this.isRecording = false;
    console.log("Recording stopped");
  }
  // start recording
  private async start(): Promise<void> {
    // gets perms
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    this.source = this.audioContext.createMediaStreamSource(this.stream);

    await this.audioContext.audioWorklet.addModule("recorderNode.js");
    const recorderNode = new AudioWorkletNode(
      this.audioContext,
      "recorder-node",
    );
    this.source.connect(recorderNode);

    this.pcmData = [];
    this.lin16buffer = new Int16Array();
    recorderNode.port.onmessage = (event) => {
      const workletInput = event.data[0];
      this.pcmData.push(workletInput);
      this.lin16buffer = new Int16Array([
        ...this.lin16buffer,
        ...this.linear16PCM(workletInput),
      ]);
    };
    this.audioContext.resume();
    this.isRecording = true;
    await this.updateUI();
    console.log("Recording started");
  }

  // stop recording
  private async stop(): Promise<void> {
    this.stream!.getTracks().forEach((track) => track.stop()); // Stop the microphone track
    this.audioContext.close(); // Close the audio context

    // Combine pcmChunks into a single Float32Array
    const totalLength = this.pcmData.reduce(
      (acc, chunk) => acc + chunk.length,
      0,
    );
    const pcmData = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.pcmData) {
      pcmData.set(chunk, offset);
      offset += chunk.length;
    }
    this.batchProcessAudio(); // process audio after stopping
    this.isRecording = false;
    this.updateUI();
    console.log("Recording stopped");
  }

  // process audio after recording stops
  private async batchProcessAudio() {
    try {
      // wav is an ArrayBuffer
      const arrayBuffer = new ArrayBuffer(this.lin16buffer.byteLength);
      new Uint8Array(arrayBuffer).set(new Uint8Array(this.lin16buffer.buffer));
      var blob = new Blob([arrayBuffer], { type: "application/octet-stream" });

      const form = new FormData();

      form.append("file", blob, "speech.wav");

      const res = await fetch("http://localhost:8000/v1/batch_process_audio/", {
        method: "POST",
        body: form,
      });

      if (!res.status || res.status !== 200) {
        const text = await res.text();
        throw new Error(`Processing audio failed: ${text}`);
      }
    } catch (err) {
      console.error("Error processing audio:", err);
      return;
    }
  }

  // get records from firestore
  private async getFromFirestore() {
    const response = await fetch("http://localhost:8000/v1/firestore_get/", {
      method: "GET",
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Firestore retrieval failed: ${text}`);
    }

    return await response.json();
  }

  // update button UI
  private async updateUI(): Promise<void> {
    if (this.isRecording) {
      this.button.classList.add("active");
    } else {
      this.button.classList.remove("active");
      this.records = [];
      try {
        this.records = await this.getFromFirestore();
        console.log("Retrieved records from Firestore:", this.records);
        this.realtimeAudio?.clear();
        this.recordList?.update(this.records);
      } catch (err) {
        console.error("Failed to retrieve records from Firestore:", err);
      }
    }
  }
}
