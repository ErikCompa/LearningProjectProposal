export default class RealtimeAudio {
  private container: HTMLElement;
  private realtimeAudioElement: HTMLElement;
  private fullTranscript: string = "";

  constructor(container: HTMLElement) {
    this.container = container;
    this.realtimeAudioElement = document.createElement("div");
    this.realtimeAudioElement.className = "record-realtime-audio";
    this.container.appendChild(this.realtimeAudioElement);
  }

  public update(newTranscript: string): void {
    this.realtimeAudioElement.innerHTML = "";

    if (this.fullTranscript.length == 0) {
      this.fullTranscript = newTranscript;
    } else {
      this.fullTranscript += ". " + newTranscript;
    }

    this.realtimeAudioElement.innerHTML = `<div class="realtime-transcript">
                                            Realtime Transcript: 
                                            ${this.fullTranscript}
                                            </div>`;
  }

  public clear(): void {
    this.fullTranscript = "";
    this.realtimeAudioElement.innerHTML = "";
  }
}
