export default class MusicRecommendation {
  private container: HTMLElement;
  private recommendationElement: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.recommendationElement = document.createElement("div");
    this.recommendationElement.className = "music-recommendation";
    this.recommendationElement.style.display = "none";
    this.container.appendChild(this.recommendationElement);
  }

  public show(song: string): void {
    this.recommendationElement.innerHTML = `
      <div class="music-content">
        <div class="music-label">Recommended for you:</div>
        <div class="music-song">${song}</div>
      </div>
    `;
    this.recommendationElement.style.display = "block";
  }

  public clear(): void {
    this.recommendationElement.innerHTML = "";
    this.recommendationElement.style.display = "none";
  }
}
