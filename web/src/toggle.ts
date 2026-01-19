export default class Toggle {
  private container: HTMLElement;
  private toggleElement: HTMLInputElement;
  private isStreaming: boolean = true;

  constructor(container: HTMLElement) {
    this.container = container;
    this.toggleElement = document.createElement("input");
    this.toggleElement.type = "checkbox";
    this.toggleElement.checked = this.isStreaming;
    this.toggleElement.addEventListener("change", () => this.handleToggle());
    const label = document.createElement("label");
    label.textContent = "Streaming Mode";
    label.appendChild(this.toggleElement);
    this.container.appendChild(label);
  }

  private handleToggle(): void {
    this.isStreaming = this.toggleElement.checked;
    console.log(`Streaming mode: ${this.isStreaming}`);
  }

  public getStreamingMode(): boolean {
    return this.isStreaming;
  }
}
