export default class AgentStatus {
  private container: HTMLElement;
  private statusElement: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.statusElement = document.createElement("div");
    this.statusElement.className = "agent-status";
    this.container.appendChild(this.statusElement);
    this.clear();
  }

  public showQuestion(text: string): void {
    this.statusElement.innerHTML = `
      <div class="agent-question">
        <div class="question-label">Question:</div>
        <div class="question-text">${text}</div>
        <div class="listening-indicator reading">Reading question...</div>
      </div>
    `;
    this.statusElement.style.display = "block";
  }

  public showListening(): void {
    // Update the listening indicator to show we're now recording
    const indicator = this.statusElement.querySelector(".listening-indicator");
    if (indicator) {
      indicator.textContent = "Listening for your answer...";
      indicator.classList.remove("reading");
    }
  }

  public showAnalyzing(): void {
    this.statusElement.innerHTML = `
      <div class="agent-analyzing">
        <div class="analyzing-text">Analyzing your response...</div>
        <div class="analyzing-dots">
          <span>.</span><span>.</span><span>.</span>
        </div>
      </div>
    `;
    this.statusElement.style.display = "block";
  }

  public showResult(mood: string, confidence: number): void {
    const confidencePercent = (confidence * 100).toFixed(0);
    this.statusElement.innerHTML = `
      <div class="agent-result">
        <div class="result-label">Your mood:</div>
        <div class="result-mood">${mood}</div>
        <div class="result-confidence">Confidence: ${confidencePercent}%</div>
      </div>
    `;
    this.statusElement.style.display = "block";
  }

  public showNoResult(message: string): void {
    this.statusElement.innerHTML = `
      <div class="agent-no-result">
        <div class="no-result-text">${message}</div>
      </div>
    `;
    this.statusElement.style.display = "block";
  }

  public showError(message: string): void {
    this.statusElement.innerHTML = `
      <div class="agent-error">
        <div class="error-text">Error: ${message}</div>
      </div>
    `;
    this.statusElement.style.display = "block";
  }

  public clear(): void {
    this.statusElement.innerHTML = "";
    this.statusElement.style.display = "none";
  }
}
