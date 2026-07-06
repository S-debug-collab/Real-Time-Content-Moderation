const API_URL = "/predict";

const textarea = document.getElementById("comment-input");
const charCount = document.getElementById("char-count");
const predictBtn = document.getElementById("predict-btn");
const btnLabel = document.getElementById("btn-label");

const resultBox = document.getElementById("result");
const predictionList = document.getElementById("prediction-list");
const statusDiv = document.getElementById("status");
const labelsDiv = document.getElementById("labels");

const errorBox = document.getElementById("error");

const CLASS_COLORS = {
    toxic: "#FBBF24",
    severe_toxic: "#EF4444",
    obscene: "#FB923C",
    threat: "#DC2626",
    insult: "#F97316",
    identity_hate: "#991B1B",
};

textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length} / 2000`;
});

predictBtn.addEventListener("click", async () => {

    const text = textarea.value.trim();

    hideError();

    if (!text) {
        showError("Please enter a comment.");
        return;
    }

    setLoading(true);

    try {

        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(body.detail || "Prediction failed.");
        }

        const data = await response.json();

        renderResult(data);

    } catch (err) {

        showError(err.message);

    } finally {

        setLoading(false);

    }

});

function renderResult(data) {

    predictionList.innerHTML = "";
    statusDiv.innerHTML = "";
    labelsDiv.innerHTML = "";

    //------------------------------------------
    // STATUS
    //------------------------------------------

    if (data.neutral) {

        statusDiv.innerHTML = `
            <h3>🟢 Status</h3>
            <p><strong>Neutral Comment</strong></p>
        `;

    } else {

        statusDiv.innerHTML = `
            <h3>🔴 Status</h3>
            <p><strong>Potentially Toxic</strong></p>
        `;

    }

    //------------------------------------------
    // LABELS
    //------------------------------------------

    if (data.predicted_labels.length === 0) {

        labelsDiv.innerHTML = `
            <h3>Detected Labels</h3>
            <p>None</p>
        `;

    } else {

        let html = "<h3>Detected Labels</h3><ul>";

        data.predicted_labels.forEach(label => {

            html += `<li>${formatLabel(label)}</li>`;

        });

        html += "</ul>";

        labelsDiv.innerHTML = html;

    }

    //------------------------------------------
    // PROBABILITIES
    //------------------------------------------

    const sorted = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

    sorted.forEach(([label, probability]) => {

        const percent = Math.round(probability * 100);

        const color =
            CLASS_COLORS[label] || "#38BDF8";

        const div = document.createElement("div");

        div.style.marginBottom = "18px";

        div.innerHTML = `
            <div style="
                display:flex;
                justify-content:space-between;
                margin-bottom:6px;
            ">

                <strong>${formatLabel(label)}</strong>

                <span>${percent}%</span>

            </div>

            <div class="result__bar">

                <div
                    class="result__bar-fill"
                    style="
                        width:${percent}%;
                        background:${color};
                    ">
                </div>

            </div>
        `;

        predictionList.appendChild(div);

    });

    resultBox.classList.remove("result--hidden");

}

function formatLabel(label) {

    return label
        .replace(/_/g, " ")
        .replace(/\b\w/g, c => c.toUpperCase());

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

    btnLabel.textContent =
        isLoading
            ? "Analyzing..."
            : "Analyze Comment";

}
