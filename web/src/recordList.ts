// from firestore
interface Response {
  uid: string;
  transcript: string;
  transcript_confidence: number;
  mood: {
    mood: string;
    confidence: number;
    evidence?: string[] | null;
  };
  created_at: string;
}

export default class RecordList {
  private container: HTMLElement;
  private listElement: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.listElement = document.createElement("div");
    this.listElement.className = "record-list";
    this.container.appendChild(this.listElement);
  }

  public update(records: Response[]): void {
    this.listElement.innerHTML = "";

    if (records.length === 0) {
      return;
    }

    records.sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );

    // Loop through records and create elements to be displayed
    records.forEach((record) => {
      const recordItem = document.createElement("div");
      recordItem.className = "record-item";

      const timestamp = new Date(record.created_at).toLocaleString();

      recordItem.innerHTML = `
        <div>Created at: ${timestamp}</div>
        <div>Transcript: ${record.transcript}</div>
        <div>Transcript Confidence: ${(
          record.transcript_confidence * 100
        ).toFixed(0)}%</div>
        <div>Mood: ${record.mood.mood}</div>
        <div>Mood Confidence: ${(record.mood.confidence * 100).toFixed(
          0
        )}%</div>
        <div>Evidence: ${record.mood.evidence?.join(", ")}</div>
      `;

      this.listElement.appendChild(recordItem);
    });
  }

  public clear(): void {
    this.listElement.innerHTML = "";
  }
}
