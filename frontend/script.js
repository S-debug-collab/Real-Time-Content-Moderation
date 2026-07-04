const API_URL = "/predict";

const textarea = document.getElementById("comment-input");
const charCount = document.getElementById("char-count");
const predictBtn = document.getElementById("predict-btn");
const btnLabel = document.getElementById("btn-label");
const resultBox = document.getElementById("result");
const resultBadge = document.getElementById("result-badge");
const resultLabel = document.getElementById("result-label");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const errorBox = document.getElementById("error");

// Maps each predicted class to a CSS color variable defined in style.css,
// so the whole badge/bar re-colors itself based on the model's output.
const CLASS_COLORS = {
  "Neutral": "var(--neutral)",
  "Toxic": "var(--toxic)",
  "Offensive": "var(--offensive)",
  "Hate Speech": "var(--hate)",
};

textarea.addEventListener("input", () => {
  charCount.textContent = `${textarea.value.length} / 2000`;
});

predictBtn.addEventListener("click", async () => {
  const text = textarea.value.trim();

  hideError();

  if (!text) {
    showError("Please enter a comment before analyzing.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();
    renderResult(data.prediction, data.confidence);
  } catch (err) {
    showError(err.message || "Something went wrong. Is the API running?");
  } finally {
    setLoading(false);
  }
});

function renderResult(prediction, confidence) {
  const color = CLASS_COLORS[prediction] || "var(--text-muted)";
  document.documentElement.style.setProperty("--current", color);

  resultLabel.textContent = prediction;
  const pct = Math.round(confidence * 100);
  confidenceValue.textContent = `${pct}%`;
  confidenceFill.style.width = `${pct}%`;

  resultBox.classList.remove("result--hidden");
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("error--hidden");
}

function hideError() {
  errorBox.classList.add("error--hidden");
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnLabel.textContent = isLoading ? "Analyzing..." : "Analyze comment";
}
