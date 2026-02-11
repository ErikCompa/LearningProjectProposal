import Chart from "chart.js/auto";

export default class EmotionGraph {
  private container: HTMLElement;
  private emotionGraphElement: HTMLElement;
  private chartInstance: Chart | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.emotionGraphElement = document.createElement("div");
    this.emotionGraphElement.className = "emotion-graph";
    this.emotionGraphElement.style.display = "none";
    this.container.appendChild(this.emotionGraphElement);
  }

  public update(
    mood: string,
    confidence: number,
    negativeEmotionPercentages: Record<string, number> | null,
  ): void {
    this.show();
    const NEGATIVE_EMOTIONS = [
      "Depressed",
      "Sad",
      "Stressed",
      "Anxious",
      "Angry",
      "Frustrated",
      "Unfocused",
      "Confused",
    ];
    this.emotionGraphElement.innerHTML = `
      <div class="emotion-graph-content">
        <div class="result-label">Main detected mood:</div>
        <div class="result-mood">${mood}</div>
        <div class="result-confidence">Confidence: ${(confidence * 100).toFixed(0)}%</div>
        <div class="emotion-chart-container">
          <canvas id="emotion-bar-chart" width="400" height="250"></canvas>
        </div>
      </div>
    `;
    const canvas = this.emotionGraphElement.querySelector(
      "#emotion-bar-chart",
    ) as HTMLCanvasElement | null;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const labels = NEGATIVE_EMOTIONS;
    const data = labels.map(
      (emotion) =>
        (negativeEmotionPercentages && negativeEmotionPercentages[emotion]) ||
        0,
    );
    if (this.chartInstance) {
      this.chartInstance.destroy();
    }
    this.chartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Negative Emotion (%)",
            data,
            backgroundColor: [
              "#ef4444",
              "#f59e42",
              "#fbbf24",
              "#38bdf8",
              "#6366f1",
              "#a21caf",
              "#14b8a6",
              "#64748b",
            ],
            borderRadius: 6,
            borderSkipped: false,
          },
        ],
      },
      options: {
        indexAxis: "y",
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: "%" } },
        },
        plugins: {
          legend: { display: false },
        },
        animation: false,
        responsive: false,
      },
    });
  }

  public show(): void {
    this.emotionGraphElement.style.display = "flex";
  }

  public hide(): void {
    this.emotionGraphElement.style.display = "none";
  }
}
