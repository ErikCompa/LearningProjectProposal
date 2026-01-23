import firestoreRecord from "../audio/types";

export default class TranscriptionList {
  private container: HTMLElement;
  private listElement: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.listElement = document.createElement("div");
    this.listElement.className = "transcription-list";
    this.container.appendChild(this.listElement);
  }

  // update the list with new transcriptions
  public update(transcriptions: firestoreRecord[]): void {
    this.listElement.innerHTML = "";

    if (transcriptions.length === 0) {
      return;
    }

    // sort to display most recent first
    transcriptions.sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    );

    // loop through transcriptions to create elements to be displayed, appending to listElement
    transcriptions.forEach((transcription) => {
      const transcriptionItem = document.createElement("div");
      transcriptionItem.className = "transcription-item";

      const timestamp = new Date(transcription.created_at).toLocaleString();

      transcriptionItem.innerHTML = `
        <div>Created at: ${timestamp}</div>
        <div>Transcript: ${transcription.transcript}</div>
        <div>Mood: ${transcription.mood.mood}</div>
        <div>Mood Confidence: ${(transcription.mood.confidence * 100).toFixed(
          0,
        )}%</div>
        <div>Evidence: ${transcription.mood.evidence?.join(", ")}</div>
      `;

      this.listElement.appendChild(transcriptionItem);
    });
  }

  public clear(): void {
    this.listElement.innerHTML = "";
  }
}
