export default class RealtimeTranscript {
  private container: HTMLElement;
  private realtimeTranscriptElement: HTMLElement;
  private final: string = "";
  private incoming: string = "";

  constructor(container: HTMLElement) {
    this.container = container;
    this.realtimeTranscriptElement = document.createElement("div");
    this.realtimeTranscriptElement.className = "realtime-transcript";
    this.realtimeTranscriptElement.style.display = "none";
    this.container.appendChild(this.realtimeTranscriptElement);
  }

  public update(newTranscript: string, isFinal: boolean = false): void {
    this.realtimeTranscriptElement.style.display = "block";
    this.realtimeTranscriptElement.innerHTML = "";

    if (isFinal) {
      this.final += newTranscript;
      this.incoming = "";
    } else {
      this.incoming = newTranscript;
    }

    this.realtimeTranscriptElement.innerHTML = `
        <div class="realtime-transcript">
            Realtime Transcript: 
            <span class="final"> ${this.final} </span>
            <span class="incoming">${this.incoming}</span> 
        </div>`;
  }

  public clear(): void {
    this.final = "";
    this.incoming = "";
    this.realtimeTranscriptElement.innerHTML = "";
    this.realtimeTranscriptElement.style.display = "none";
  }
}
