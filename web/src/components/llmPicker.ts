export default class LLMPicker {
  private container: HTMLElement;
  private selectElement: HTMLSelectElement | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.render();
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="llm-picker">
        <label for="llm-select">AI Model:</label>
        <select id="llm-select" class="llm-select">
          <option value="openai">OpenAI</option>
          <option value="gemini">Gemini</option>
        </select>
      </div>
    `;
    this.selectElement = this.container.querySelector("#llm-select");
  }

  public getSelectedLLM(): string {
    return "openai"; // Currently hardcoded to "openai"
  }

  public setEnabled(enabled: boolean): void {
    if (this.selectElement) {
      this.selectElement.disabled = !enabled;
    }
  }
}
