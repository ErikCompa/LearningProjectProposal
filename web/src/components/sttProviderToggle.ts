export type STTProvider = "google" | "elevenlabs";

export default class STTProviderToggle {
  private container: HTMLElement;
  private toggleElement: HTMLInputElement;
  private isElevenLabs: boolean = false;
  private onProviderChange?: (provider: STTProvider) => void;

  constructor(
    container: HTMLElement,
    onProviderChange?: (provider: STTProvider) => void,
  ) {
    this.container = container;
    this.onProviderChange = onProviderChange;
    this.toggleElement = document.createElement("input");
    this.toggleElement.type = "checkbox";
    this.toggleElement.checked = this.isElevenLabs;
    this.toggleElement.addEventListener("change", () => this.handleToggle());
    const label = document.createElement("label");
    label.textContent = "STT Provider: ";
    const span = document.createElement("span");
    span.textContent = "google";
    label.appendChild(span);
    label.appendChild(this.toggleElement);
    this.container.appendChild(label);
  }

  private handleToggle(): void {
    this.isElevenLabs = this.toggleElement.checked;
    const label = this.container.querySelector("label");
    this.updateLabel(label);
    if (this.onProviderChange) {
      this.onProviderChange(this.getProvider());
    }
  }

  private updateLabel(label: HTMLLabelElement | null): void {
    if (!label) return;
    const span = label.querySelector("span");
    if (span) {
      span.textContent = this.isElevenLabs ? "elevenlabs" : "google";
    }
  }

  public getProvider(): STTProvider {
    return this.isElevenLabs ? "elevenlabs" : "google";
  }
}
