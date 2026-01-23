import AudioRecorder from "../audio/audioRecorder";

export default class RecordButton {
  private container: HTMLElement;
  private buttonElement: HTMLButtonElement;
  private audioRecorder: AudioRecorder;
  private onRecordingComplete?: () => Promise<void>;

  constructor(
    container: HTMLElement,
    audioRecorder: AudioRecorder,
    onRecordingComplete?: () => Promise<void>,
  ) {
    this.container = container;
    this.audioRecorder = audioRecorder;
    this.onRecordingComplete = onRecordingComplete;
    this.buttonElement = document.createElement("button");
    this.buttonElement.addEventListener("click", () => this.toggle());
    this.container.appendChild(this.buttonElement);
    this.updatebuttonUI();
  }

  private async toggle(): Promise<void> {
    if (this.audioRecorder.getRecordingStatus()) {
      await this.audioRecorder.stopRecording();
      if (this.onRecordingComplete) {
        await this.onRecordingComplete();
      }
    } else {
      await this.audioRecorder.startRecording();
    }
    this.updatebuttonUI();
  }

  private updatebuttonUI(): void {
    if (this.audioRecorder.getRecordingStatus()) {
      this.buttonElement.classList.add("active");
    } else {
      this.buttonElement.classList.remove("active");
    }
  }
}
