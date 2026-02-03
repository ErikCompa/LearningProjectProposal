export default class RealtimeTranscript {
  private container: HTMLElement;
  private realtimeTranscriptElement: HTMLElement;
  private final: string = "";
  private incoming: string = "";
  private provider: string = "google";

  constructor(container: HTMLElement) {
    this.container = container;
    this.realtimeTranscriptElement = document.createElement("div");
    this.realtimeTranscriptElement.className = "realtime-transcript";
    this.realtimeTranscriptElement.style.display = "none";
    this.container.appendChild(this.realtimeTranscriptElement);
  }

  public setProvider(provider: string): void {
    this.provider = provider;
  }

  public update(
    newTranscript: string,
    isFinal: boolean = false,
    stability: number = 0.0,
  ): void {
    this.realtimeTranscriptElement.style.display = "block";
    this.realtimeTranscriptElement.innerHTML = "";

    if (isFinal) {
      if (this.provider === "elevenlabs") {
        this.final += newTranscript + " ";
      } else {
        this.final += newTranscript + ". ";
      }
      this.incoming = "";
    } else {
      if (this.provider === "elevenlabs") {
        this.incoming = newTranscript;
      } else {
        this.incoming = this.merge(this.incoming, newTranscript, stability);
      }
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

  public merge(oldText: string, newText: string, stability: number): string {
    oldText = oldText.trim();
    newText = newText.trim();

    // subject to change in next message, might be wrong word
    if (stability < 0.3) {
      return oldText;
    }

    if (oldText === "") {
      return newText;
    }

    if (newText === "") {
      return oldText;
    }

    // same phrase
    if (oldText === newText) {
      return oldText;
    }

    // partial overlap
    // old text completly inside new text -> return new text
    if (newText.includes(oldText)) {
      return newText;
    }

    // new text completly inside old text -> return old text
    if (oldText.includes(newText)) {
      return oldText;
    }

    let oldWords = oldText.split(" ");
    let newWords = newText.split(" ");

    let final = "";
    let foundOverlap = false;
    // overlap at the end of old text and start of new text -> return old text + non-overlapping part of new text
    for (let i = oldWords.length; i > 0; i--) {
      let overlap = oldWords.slice(-i).join(" ");
      if (newText.startsWith(overlap)) {
        final = oldText + " " + newWords.slice(i).join(" ");
        foundOverlap = true;
      } else {
        continue;
      }
    }
    if (foundOverlap) {
      return final.trim();
    }

    final = "";
    foundOverlap = false;
    // overlap at the start of old text and end of new text -> return non-overlapping part of old text + new text
    for (let i = newWords.length; i > 0; i--) {
      let overlap = newWords.slice(-i).join(" ");
      if (oldText.startsWith(overlap)) {
        final = newText + " " + oldWords.slice(i).join(" ");
        foundOverlap = true;
      } else {
        continue;
      }
    }
    if (foundOverlap) {
      return final.trim();
    }

    // completly different phrases -> reutrn both appended
    return oldText + " " + newText;
  }
}
