export default class Recorder {
  private isRecording: boolean = false;
  private button: HTMLButtonElement;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private onAudioReady?: (audioBlob: Blob) => void;

  constructor(container: HTMLElement, onAudioReady?: (audioBlob: Blob) => void) {
    this.onAudioReady = onAudioReady;
    this.button = document.createElement('button');
    this.button.className = 'recorder-button';
    this.button.addEventListener('click', () => this.toggle());
    container.appendChild(this.button);
  }

  private toggle(): void {
    if (this.isRecording) {
      this.stop();
    } else {
      this.start();
    }
  }

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
        
        this.mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' }); //transform blob into webm format
          if (this.onAudioReady) {
            this.onAudioReady(audioBlob);
          }
          
          // Upload the audio
          try {
            const transcript = await this.uploadSpeech(audioBlob);
            console.log('Transcription result:', transcript);
          } catch (err) {
            console.error('Failed to upload audio:', err);
          }
          
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

  private async uploadSpeech(audioBlob: Blob) {
    const form = new FormData();

    form.append('file', audioBlob, 'speech.webm');

    const response = await fetch(
      'http://localhost:8000/v1/transcribe/',
      {
        method: 'POST',
        body: form,
      }
    );

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Transcription failed: ${text}`);
    }

    return await response.json();
  }

  private updateUI(): void {
    if (this.isRecording) {
      this.button.classList.add('active');
    } else {
      this.button.classList.remove('active');
    }
  }
}
