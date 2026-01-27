import AudioRecorder from "../audio/audioRecorder";

export default class RecordButton {
  private container: HTMLElement;
  private buttonElement: HTMLButtonElement;
  private audioRecorder: AudioRecorder;
  private onRecordingComplete?: () => Promise<void>;
  private onRecordingStart?: () => void;

  constructor(
    container: HTMLElement,
    audioRecorder: AudioRecorder,
    onRecordingComplete?: () => Promise<void>,
    onRecordingStart?: () => void,
  ) {
    this.container = container;
    this.audioRecorder = audioRecorder;
    this.onRecordingComplete = onRecordingComplete;
    this.onRecordingStart = onRecordingStart;
    this.buttonElement = document.createElement("button");
    this.buttonElement.className = "record-button";
    this.buttonElement.addEventListener("click", () => this.toggle());
    this.container.appendChild(this.buttonElement);
  }

  private async toggle(): Promise<void> {
    this.updateButtonUI(null);
    try {
      if (this.audioRecorder.getRecordingStatus()) {
        await this.audioRecorder.stopRecording();
        if (this.onRecordingComplete) {
          await this.onRecordingComplete();
        }
      } else {
        await this.audioRecorder.startRecording();
        if (this.onRecordingStart) {
          this.onRecordingStart();
        }
      }
    } catch (error) {
      this.updateButtonUI(this.audioRecorder.getRecordingStatus());
    }
  }

  private updateButtonUI(isRecording: boolean | null): void {
    if (isRecording !== null) {
      if (isRecording) {
        this.buttonElement.classList.add("active");
      } else {
        this.buttonElement.classList.remove("active");
      }
      return;
    }
    if (this.buttonElement.classList.contains("active")) {
      this.buttonElement.classList.remove("active");
    } else {
      this.buttonElement.classList.add("active");
    }
  }
}
