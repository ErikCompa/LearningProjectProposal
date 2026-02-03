export default class RealtimeTranscript {
  private container: HTMLElement;
  private realtimeTranscriptElement: HTMLElement;
  private finalText: string = "";
  private incomingText: string = "";

  constructor(container: HTMLElement) {
    this.container = container;
    this.realtimeTranscriptElement = document.createElement("div");
    this.realtimeTranscriptElement.className = "realtime-transcript";
    this.realtimeTranscriptElement.style.display = "none";
    this.container.appendChild(this.realtimeTranscriptElement);
  }

  public update(newTranscript: string, isFinal: boolean = false): void {
    this.realtimeTranscriptElement.style.display = "block";

    if (isFinal) {
      this.finalText += newTranscript + " ";
      this.incomingText = "";
    } else {
      this.incomingText = newTranscript;
    }

    this.realtimeTranscriptElement.innerHTML = `
      <div class="realtime-transcript">
        <span class="final">${this.finalText}</span>
        <span class="incoming">${this.incomingText}</span> 
      </div>
    `;
  }

  public clear(): void {
    this.finalText = "";
    this.incomingText = "";
    this.realtimeTranscriptElement.innerHTML = "";
    this.realtimeTranscriptElement.style.display = "none";
  }

  public hide(): void {
    this.realtimeTranscriptElement.style.display = "none";
  }

  public show(): void {
    this.realtimeTranscriptElement.style.display = "block";
  }
}
