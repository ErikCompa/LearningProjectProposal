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
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private recordList: RecordList;
  public records: FirestoreRecord[] = [];
  private websocket: WebSocket | null = null;
  private streaming: boolean = true;
  private realtimeAudio: RealtimeAudio | null = null;
  private toggle_streaming: Toggle;

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
  private toggle(): void {
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
        this.stream_stop();
      } else {
        this.stream_start();
      }
    }
  }

  private stream_start(): void {
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
        this.realtimeAudio?.update(data.transcript, data.is_final);
      }
    };

    // remove list if present
    this.recordList?.clear();
    this.realtimeAudio?.clear();

    // gets perms
    navigator.mediaDevices
      .getUserMedia({ audio: true, video: false })
      .then((stream) => {
        this.mediaRecorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        }); // new instance

        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            // process chunks
            if (
              this.websocket &&
              this.websocket.readyState === WebSocket.OPEN
            ) {
              this.websocket.send(event.data);
            }
          }
        };

        // on stop event handler
        this.mediaRecorder.onstop = async () => {
          // retrieve from firestore
          this.records = [];
          try {
            this.records = await this.getFromFirestore();
            console.log("Retrieved records from Firestore:", this.records);
            this.realtimeAudio?.clear();
            this.recordList?.update(this.records);
          } catch (err) {
            console.error("Failed to retrieve records from Firestore:", err);
          }
        };

        this.mediaRecorder.start(250);
        this.isRecording = true;
        this.updateUI();
        console.log("Recording started");
      })
      .catch((err) => {
        console.error("Error accessing microphone:", err);
        this.isRecording = false;
        this.updateUI();
      });
  }

  // stop recording
  private stream_stop(): void {
    // stop after 5 seconds to allow final messages to be processed
    setTimeout(() => {
      if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
        this.mediaRecorder.stop();
        this.mediaRecorder.stream.getTracks().forEach((track) => track.stop());
        this.mediaRecorder = null;
      }
      if (this.websocket) {
        this.websocket.close();
        this.websocket = null;
      }
    }, 5000);

    this.isRecording = false;
    this.updateUI();
    console.log("Recording stopped");
  }
  // start recording
  private start(): void {
    // gets perms
    navigator.mediaDevices
      .getUserMedia({ audio: true, video: false })
      .then((stream) => {
        this.audioChunks = [];
        this.mediaRecorder = new MediaRecorder(stream); // new instance

        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            this.audioChunks.push(event.data); // get blob into chunk arr
          }
        };

        // on stop event handler
        this.mediaRecorder.onstop = async () => {
          await this.batchProcessAudio();
          this.audioChunks = [];
        };

        this.mediaRecorder.start();
        this.isRecording = true;
        this.updateUI();
        console.log("Recording started");
      })
      .catch((err) => {
        console.error("Error accessing microphone:", err);
        this.isRecording = false;
        this.updateUI();
      });
  }

  // stop recording
  private stop(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      this.mediaRecorder = null;
    }
    this.isRecording = false;
    this.updateUI();
    console.log("Recording stopped");
  }

  // process audio after recording stops
  private async batchProcessAudio() {
    const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" }); //transform blob into webm format

    // Call combined endpoint
    try {
      if (!audioBlob || audioBlob.size === 0) {
        throw new Error("Invalid audio blob for transcription");
      }

      const form = new FormData();

      form.append("file", audioBlob, "speech.webm");

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

    // retrieve from firestore
    this.records = [];
    try {
      this.records = await this.getFromFirestore();
      console.log("Retrieved records from Firestore:", this.records);
      this.recordList?.update(this.records);
    } catch (err) {
      console.error("Failed to retrieve records from Firestore:", err);
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
  private updateUI(): void {
    if (this.isRecording) {
      this.button.classList.add("active");
    } else {
      this.button.classList.remove("active");
    }
  }
}
