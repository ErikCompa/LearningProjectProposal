import type firestoreRecord from "../audio/types";

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
    this.listElement.style.display = "block";
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

      // Only show top 50 moods
      const top50Moods = transcription.top_50_moods
        ? transcription.top_50_moods
            .map(
              (m: { label: string; score: number }) =>
                `${m.label} (${(m.score * 100).toFixed(0)}%)`,
            )
            .join(", ")
        : "";

      // Only show top 50 evidences
      const top50Evidences = transcription.top_50_evidences
        ? transcription.top_50_evidences
            .map(
              (e: { label: string; explanation: string }) =>
                `${e.label}: ${e.explanation}`,
            )
            .join("; ")
        : "";

      // All moods (not just top 50)
      const allMoods = transcription.moods
        ? transcription.moods
            .map(
              (m: { label: string; score: number }) =>
                `${m.label} (${(m.score * 100).toFixed(0)}%)`,
            )
            .join(", ")
        : "";

      transcriptionItem.innerHTML = `
        <div>Created at: ${timestamp}</div>
        <div>Transcript: ${transcription.transcript}</div>
        <div>All Moods: ${allMoods}</div>
        <div>Mood Confidence: ${(transcription.mood_confidence * 100).toFixed(0)}%</div>
        <div>Top 50% Moods: ${top50Moods}</div>
        <div>Top 50% Evidence: ${top50Evidences}</div>
      `;

      this.listElement.appendChild(transcriptionItem);
    });
  }

  public clear(): void {
    this.listElement.innerHTML = "";
  }

  public hide(): void {
    this.listElement.style.display = "none";
  }

  public show(): void {
    this.listElement.style.display = "block";
  }
}
