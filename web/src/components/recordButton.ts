import AudioRecorder from "../audio/audioRecorder";

export default class RecordButton {
  private container: HTMLElement;
  private buttonElement: HTMLButtonElement;
  private audioRecorder: AudioRecorder;

  constructor(container: HTMLElement, audioRecorder: AudioRecorder) {
    this.container = container;
    this.audioRecorder = audioRecorder;
    this.buttonElement = document.createElement("button");
    this.buttonElement.className = "record-button";
    this.buttonElement.addEventListener("click", () => this.toggle());
    this.container.appendChild(this.buttonElement);
  }

  private async toggle(): Promise<void> {
    try {
      if (this.audioRecorder.getRecordingStatus()) {
        // User clicked to stop recording early
        await this.audioRecorder.stopRecording();
        this.updateButtonUI(false);
      } else {
        // Start recording and session
        await this.audioRecorder.startRecording();
        this.updateButtonUI(true);
      }
    } catch (error) {
      console.error("Error toggling recording:", error);
      this.updateButtonUI(false);
    }
  }

  private updateButtonUI(isRecording: boolean): void {
    if (isRecording) {
      this.buttonElement.classList.add("active");
    } else {
      this.buttonElement.classList.remove("active");
    }
  }

  public setEnabled(enabled: boolean): void {
    this.buttonElement.disabled = !enabled;
  }

  public setSessionActive(active: boolean): void {
    this.updateButtonUI(active);
  }
}
