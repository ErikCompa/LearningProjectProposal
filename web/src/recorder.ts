import RecordList from './recordList.ts';

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

  constructor(container: HTMLElement, recordList: RecordList) {
    this.recordList = recordList;
    this.button = document.createElement('button');
    this.button.className = 'recorder-button';
    this.button.addEventListener('click', () => this.toggle());
    container.appendChild(this.button);
  }

  // Button toggle handler
  private toggle(): void {
    if (this.isRecording) {
      this.stop();
    } else {
      this.start();
    }
  }

  // start recording
  private start(): void {
    // gets perms
    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      .then(stream => {
        this.audioChunks = [];
        this.mediaRecorder = new MediaRecorder(stream); // new instance
        
        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            this.audioChunks.push(event.data); // get blob into chunk arr
          }
        };
        
        // on stop event handler
        this.mediaRecorder.onstop = async () => {
          await this.processAudio();
          this.audioChunks = [];
        };
        
        this.mediaRecorder.start();
        this.isRecording = true;
        this.updateUI();
        console.log('Recording started');
      })
      .catch(err => {
        console.error('Error accessing microphone:', err);
        this.isRecording = false;
        this.updateUI();
      });
  }

  // stop recording
  private stop(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      this.mediaRecorder = null;
    }
    this.isRecording = false;
    this.updateUI();
    console.log('Recording stopped');
  }

  // process audio after recording stops
  private async processAudio() {
    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' }); //transform blob into webm format

    // Call combined endpoint
    try {
      if (!audioBlob || audioBlob.size === 0) {
        throw new Error('Invalid audio blob for transcription');
      }

      const form = new FormData();

      form.append('file', audioBlob, 'speech.webm');

      const res = await fetch(
        'http://localhost:8000/v1/process_audio/',
        {
          method: 'POST',
          body: form,
        }
      );

      if (!res.status || res.status !== 200) {
        const text = await res.text();
        throw new Error(`Processing audio failed: ${text}`);
      }

    } catch (err) {
      console.error('Error processing audio:', err);
      return;
    }

    // retrieve from firestore
    this.records = [];
    try {
      this.records = await this.getFromFirestore();
      console.log('Retrieved records from Firestore:', this.records);
      this.recordList?.update(this.records);
    } catch (err) {
      console.error('Failed to retrieve records from Firestore:', err);
    }
  }

  // get records from firestore
  private async getFromFirestore() {
    const response = await fetch(
      'http://localhost:8000/v1/firestore_get/',
      {
        method: 'GET',
      }
    );

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Firestore retrieval failed: ${text}`);
    }

    return await response.json();
  }
  
  // update button UI
  private updateUI(): void {
    if (this.isRecording) {
      this.button.classList.add('active');
    } else {
      this.button.classList.remove('active');
    }
  }
}
