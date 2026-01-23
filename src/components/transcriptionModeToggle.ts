import { type TranscriptionMode } from "../audio/types";

export default class TranscriptionModeToggle {
  private container: HTMLElement;
  private toggleElement: HTMLInputElement;
  private isStreamingMode: boolean = true;
  private onModeChange?: (mode: TranscriptionMode) => void;

  constructor(
    container: HTMLElement,
    onModeChange?: (mode: TranscriptionMode) => void,
  ) {
    this.container = container;
    this.onModeChange = onModeChange;
    this.toggleElement = document.createElement("input");
    this.toggleElement.type = "checkbox";
    this.toggleElement.checked = this.isStreamingMode;
    this.toggleElement.addEventListener("change", () => this.handleToggle());
    const label = document.createElement("label");
    label.textContent = "Transcription Mode: stream";
    label.appendChild(this.toggleElement);
    this.container.appendChild(label);
  }

  private handleToggle(): void {
    this.isStreamingMode = this.toggleElement.checked;
    const label = this.container.querySelector("label");
    this.updateLabel(label);
    if (this.onModeChange) {
      this.onModeChange(this.getStreamingMode());
    }
  }

  private updateLabel(label: HTMLLabelElement | null): void {
    if (!label) return;
    label.textContent =
      "Transcription Mode: " + (this.isStreamingMode ? "stream" : "batch");
    label.appendChild(this.toggleElement);
  }

  public getStreamingMode(): TranscriptionMode {
    return this.isStreamingMode ? "stream" : "batch";
  }
}
