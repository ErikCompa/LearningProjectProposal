import FirestoreService from "../services/firestoreService";
import TranscriptionList from "./transcriptionList";

export default class UpdateButton {
  private container: HTMLElement;
  private buttonElement: HTMLButtonElement;
  private firestoreService: FirestoreService;
  private transcriptionListComponent: TranscriptionList;

  constructor(
    container: HTMLElement,
    firestoreService: FirestoreService,
    transcriptionListComponent: TranscriptionList,
  ) {
    this.container = container;
    this.firestoreService = firestoreService;
    this.transcriptionListComponent = transcriptionListComponent;
    this.buttonElement = document.createElement("button");
    this.buttonElement.textContent = "Update";
    this.buttonElement.className = "update-button";
    this.buttonElement.addEventListener("click", () => this.handleClick());
    this.container.appendChild(this.buttonElement);
  }

  private async handleClick(): Promise<void> {
    this.buttonElement.disabled = true;
    try {
      const records = await this.firestoreService.getFromFirestore();
      this.transcriptionListComponent.update(records);
      this.transcriptionListComponent.show();
    } catch (err) {
      alert("Failed to update records.");
    }
    this.buttonElement.disabled = false;
  }
}
