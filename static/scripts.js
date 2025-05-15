// ========== Loading Bar Functions ==========
function startLoading() {
  const loadingBar = document.getElementById("loading-bar");
  if (loadingBar) {
    loadingBar.style.width = "0%";
    loadingBar.style.display = "block";
    setTimeout(() => (loadingBar.style.width = "100%"), 50);
  }
}

function stopLoading() {
  const loadingBar = document.getElementById("loading-bar");
  if (loadingBar) {
    setTimeout(() => {
      loadingBar.style.display = "none";
      loadingBar.style.width = "0%";
    }, 400);
  }
}

// ========== Spinner Functions ==========
function showSpinner() {
  document.getElementById("spinner")?.style.setProperty("display", "block");
}

function hideSpinner() {
  document.getElementById("spinner")?.style.setProperty("display", "none");
}

// ========== Toast Notification ==========
function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast-container";
  toast.innerHTML = `<div class="toast-message">${message}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ========== Tab Switching ==========
function setActiveTab(tabId) {
  document
    .querySelectorAll(".tab-button")
    .forEach((btn) => btn.classList.remove("active"));
  document
    .querySelectorAll(".tab-content")
    .forEach((tab) => tab.classList.remove("active"));

  document
    .querySelector(`.tab-button[data-tab="${tabId}"]`)
    ?.classList.add("active");
  document.getElementById(tabId)?.classList.add("active");
}

// ========== Load Data into Table ==========
async function loadData() {
  const fileInput = document.getElementById('fileInput');
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');
  const file = fileInput.files[0];

  if (!file) {
    alert("Please choose a CSV file.");
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  document.getElementById('loadBtn').innerText = 'Loading...';
  document.getElementById('loadBtn').disabled = true;

  try {
    const response = await fetch('/preprocess', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();

    if (result.error) {
      tableBody.innerHTML = `<tr><td colspan="100%">${result.error}</td></tr>`;
      return;
    }

    const sample = result.sample;
    if (Array.isArray(sample) && sample.length) {
      const keys = Object.keys(sample[0]);
      tableHead.innerHTML = `<tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
      tableBody.innerHTML = sample.map(row =>
        `<tr>${keys.map(k => `<td>${row[k]}</td>`).join('')}</tr>`
      ).join('');
    } else {
      tableBody.innerHTML = `<tr><td colspan="100%">No data available</td></tr>`;
    }

    console.log('Preprocessing Info:', result.info);

  } catch (error) {
    console.error('Error:', error);
    tableBody.innerHTML = `<tr><td colspan="100%">Error loading data.</td></tr>`;
  } finally {
    document.getElementById('loadBtn').innerText = 'Load Dataset';
    document.getElementById('loadBtn').disabled = false;
  }
}

async function trainAllModelsAndRenderChart() {
  startLoading();
  try {
    const res = await fetch("/train/all_models", { method: "POST" });
    const data = await res.json();
    renderGroupedBarChart(data);
    showToast("All models trained and metrics updated.");
  } catch (err) {
    console.error("Failed to train models or fetch metrics:", err);
    showToast("Error training models.");
  } finally {
    stopLoading();
  }
}

let comparisonChart = null;

function renderGroupedBarChart(metrics) {
  const ctx = document.getElementById("chart-comparison").getContext("2d");

  const labels = ["Accuracy", "Precision", "Recall", "F1 Score"];

  const datasets = [
    {
      label: "Random Forest",
      backgroundColor: "#4caf50",
      data: [
        metrics.rf?.accuracy ?? 0,
        metrics.rf?.precision ?? 0,
        metrics.rf?.recall ?? 0,
        metrics.rf?.f1_score ?? 0,
      ],
    },
    {
      label: "Isolation Forest",
      backgroundColor: "#2196f3",
      data: [
        metrics.iso?.accuracy ?? 0,
        metrics.iso?.precision ?? 0,
        metrics.iso?.recall ?? 0,
        metrics.iso?.f1_score ?? 0,
      ],
    },
    {
      label: "Autoencoder",
      backgroundColor: "#ff9800",
      data: [
        metrics.auto?.accuracy ?? 0,
        metrics.auto?.precision ?? 0,
        metrics.auto?.recall ?? 0,
        metrics.auto?.f1_score ?? 0,
      ],
    },
  ];

  if (comparisonChart) {
    comparisonChart.data.datasets = datasets;
    comparisonChart.update();
  } else {
    comparisonChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: "Model Performance Comparison",
          },
          legend: {
            position: "top",
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
          },
        },
      },
    });
  }
}

// ========== Training Functions ==========
async function trainModel(endpoint, modelName, metricPrefix) {
  startLoading();
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: modelName }),
    });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    const stats = data.stats ?? data; // fallback for direct stat returns

    if (metricPrefix && stats) {
      const keys = ["accuracy", "precision", "recall", "f1_score"];
      for (const key of keys) {
        const el = document.getElementById(`${metricPrefix}-${key}`);
        if (el) el.textContent = (stats[key] ?? 0).toFixed(6);
      }

      // Optional: show confusion matrix or total anomalies
      const anomalyRate = stats.anomaly_rate ?? stats.anomaly_percentage;
      const total = stats.total;
      const anomalyCount = stats.anomalies_detected;

      const anomalyLabel = document.getElementById(`${metricPrefix}-anomaly-rate`);
      if (anomalyLabel && anomalyRate != null) {
        anomalyLabel.textContent = `${anomalyRate}% (${anomalyCount} / ${total})`;
      }
    }

    showToast(data.message || `${modelName} trained.`);
  } catch (error) {
    console.error(`Error training ${modelName}:`, error);
    showToast(`Failed to train ${modelName}.`);
  } finally {
    stopLoading();
  }
}

const trainRF = () => trainModel("/train/randomforest", "rf_model", "rf");
const trainISO = () => trainModel("/train/isolationforest", "iso_model", "iso");


async function trainAutoencoder() {
  startLoading();
  try {
    const res = await fetch("/train/autoencoder", { method: "POST" });
    const result = await res.json();
    if (result.error) throw new Error(result.error);

    showToast(result.message || "Autoencoder training completed.");

    document.getElementById("auto-accuracy").textContent = (result.accuracy * 100).toFixed(2) + "%";
    document.getElementById("auto-precision").textContent = (result.precision * 100).toFixed(2) + "%";
    document.getElementById("auto-recall").textContent = (result.recall * 100).toFixed(2) + "%";
    document.getElementById("auto-f1").textContent = (result.f1_score * 100).toFixed(2) + "%";
    document.getElementById("auto-anomaly-rate").textContent = (result.anomaly_rate * 100).toFixed(2) + "%";
  } catch (error) {
    console.error("Error training autoencoder:", error);
    showToast("Failed to train autoencoder.");
  } finally {
    stopLoading();
  }
}


async function predictAutoencoder() {
  try {
    const res = await fetch("/predict/autoencoder/all");
    const result = await res.json();
    const output = document.getElementById("auto-output");

    if (Array.isArray(result) && result.length > 0) {
      output.textContent = `Predictions (showing first 5):\n${JSON.stringify(result.slice(0, 5), null, 2)}`;
    } else if (Array.isArray(result.predictions) && result.predictions.length > 0) {
      output.textContent = `Predictions (showing first 5):\n${JSON.stringify(result.predictions.slice(0, 5), null, 2)}`;
    } else {
      output.textContent = "No valid prediction results found.";
    }
  } catch (error) {
    console.error("Error predicting with autoencoder:", error);
    document.getElementById("auto-output").textContent = "Failed to get predictions.";
  }
}

// ========== Evaluation Functions ==========
async function evaluateModel(endpoint, resultId, modelName) {
  startLoading();
  try {
    const res = await fetch(endpoint);
    const data = await res.json();
    const { predictions, stats } = data;

    if (Array.isArray(predictions) && predictions.length) {
      renderModelTable(predictions, "model-eval-head", "model-eval-body");
      document.getElementById(resultId).textContent = `${modelName} evaluated on ${predictions.length} samples.`;
      updateChartWithStats(stats);
    } else {
      document.getElementById(resultId).textContent = `No evaluation data available for ${modelName}.`;
      showToast(`No evaluation data for ${modelName}.`);
    }
    console.log(`${modelName} Evaluation Stats:`, stats);
  } catch (error) {
    console.error(`Error evaluating ${modelName}:`, error);
    showToast(`Failed to evaluate ${modelName}.`);
  } finally {
    stopLoading();
  }
}

const evaluateRF = () =>
  evaluateModel("/predict/randomforest/all", "rfEvalResult", "Random Forest");

const evaluateISO = () =>
  evaluateModel("/predict/isolationforest/all", "isoEvalResult", "Isolation Forest");

// ========== Utility Function ==========
function renderModelTable(predictions, headId, bodyId) {
  const tableHead = document.getElementById(headId);
  const tableBody = document.getElementById(bodyId);
  if (!predictions.length) return;

  const keys = Object.keys(predictions[0]);
  tableHead.innerHTML = `<tr>${keys.map((k) => `<th>${k}</th>`).join("")}</tr>`;
  tableBody.innerHTML = predictions
    .map((row) => `<tr>${keys.map((k) => `<td>${row[k]}</td>`).join("")}</tr>`)
    .join("");
}

document
  .querySelectorAll(".tab-button")
  .forEach((button) =>
    button.addEventListener("click", () => setActiveTab(button.dataset.tab))
  );

  const toggleInput = document.getElementById("toggle-input"); 

  if (toggleInput) {
    if (localStorage.getItem("darkMode") === "true") {
      document.body.classList.add("dark");
      toggleInput.checked = true;
    }
  
    toggleInput.addEventListener("change", () => {
      const isDark = document.body.classList.toggle("dark");
      localStorage.setItem("darkMode", isDark);
      showToast(isDark ? "Toggled dark mode" : "Toggled light mode");
    });
  }
  

setActiveTab("dataset");

// document
//   .getElementById("user-input-form")
//   ?.addEventListener("submit", async function (e) {
//     e.preventDefault();

//     const form = e.target;
//     const formData = new FormData(form);
//     const data = {};

//     for (let [key, value] of formData.entries()) {
//       // Try to parse numbers automatically
//       const num = parseFloat(value);
//       data[key] = isNaN(num) ? value : num;
//     }

//     startLoading();
//     try {
//       const res = await fetch("/predict/single", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(data),
//       });

//       const result = await res.json();
//       console.log("Prediction result:", result);

//       const output = document.getElementById("user-input-output");
//       if (output) {
//         output.textContent = `Prediction Result:\n${JSON.stringify(
//           result,
//           null,
//           2
//         )}`;
//       }

//       showToast(result.message || "Prediction completed.");
//     } catch (error) {
//       console.error("Error submitting user input:", error);
//       // showToast("Prediction failed.");
//       const output = document.getElementById("user-input-output");
//       if (output) {
//         output.textContent = "Prediction failed due to an error.";
//       }
//     } finally {
//       stopLoading();
//     }
//   });

document
  .getElementById("user-input-form")
  ?.addEventListener("submit", async function (e) {
    e.preventDefault();
    const model = document.getElementById("model-select").value;
    const inputs = document.querySelectorAll("#input-fields input");
    const inputData = {};
    let missingFields = [];

    inputs.forEach((input) => {
      if (!input.value.trim()) {
        missingFields.push(input.name);
      } else {
        inputData[input.name] = parseFloat(input.value);
      }
    });

    if (missingFields.length > 0) {
      showToast(`Please fill in all fields: ${missingFields.join(", ")}`);
      return;
    }

    showSpinner();
    let endpoint = "";
    switch (model) {
      case "rf":
        endpoint = "/predict/randomforest/manual";
        break;
      case "iso":
        endpoint = "/predict/isolationforest/manual";
        break;
      case "auto":
        endpoint = "/predict/autoencoder/manual";
        break;
      default:
        showToast("Invalid model selected.");
        hideSpinner();
        return;
    }

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });
      const result = await res.json();

      if (result.error) throw new Error(result.error);

      const predictionElement = document.getElementById("prediction-result");
      predictionElement.innerHTML = `
        <strong>${result.label}</strong><br />
        ${result.confidence ? `Confidence: ${result.confidence}` : ""}
      `;
      predictionElement.style.color =
        result.prediction === 1 ? "red" : "green";
    } catch (err) {
      showToast(`Prediction failed: ${err.message}`);
    } finally {
      hideSpinner();
    }
  });

async function renderInputFields() {
  try {
    const response = await fetch("/preprocess", { method: "POST" });
    const result = await response.json();
    const sample = result.sample[0];
    const container = document.getElementById("input-fields");
    container.innerHTML = "";

    Object.keys(sample).forEach((key) => {
      const input = document.createElement("input");
      input.name = key;
      input.placeholder = key;
      input.type = "number";
      container.appendChild(input);
    });
  } catch (e) {
    console.error("Failed to render input fields:", e);
  }
}

// renderInputFields();

const inputFieldsContainer = document.getElementById("input-fields");
const featureNames = [
  "Time",
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  "Amount",
];

featureNames.forEach((feature) => {
  const label = document.createElement("label");
  label.for = feature;
  label.textContent = feature;

  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.name = feature;
  input.id = feature;
  input.required = true;

  inputFieldsContainer.appendChild(label);
  inputFieldsContainer.appendChild(input);
});

function fillExample() {
  const example = {
    Time: 12345,
    V1: -1.2,
    V2: 0.3,
    V3: -0.5,
    V4: 1.3,
    V5: 0.9,
    V6: -1.1,
    V7: 0.2,
    V8: 0.4,
    V9: -0.2,
    V10: 1.0,
    V11: -1.3,
    V12: 0.7,
    V13: -0.1,
    V14: 0.3,
    V15: -0.8,
    V16: 0.6,
    V17: -0.5,
    V18: 1.4,
    V19: -1.2,
    V20: 0.2,
    V21: 0.9,
    V22: -0.7,
    V23: 0.1,
    V24: -0.3,
    V25: 0.8,
    V26: -0.9,
    V27: 0.5,
    V28: 0.3,
    Amount: 250.0,
  };
  Object.keys(example).forEach((key) => {
    const input = document.getElementById(key);
    if (input) input.value = parseFloat(example[key]);
  });
}

function fillFraudExample() {
  const example = {
    Time: 406,
    V1: -2.3122265423263,
    V2: 1.95199201064158,
    V3: -1.60985073229769,
    V4: 3.9979055875468,
    V5: -0.522187864667764,
    V6: -1.42654531920595,
    V7: -2.53738730624579,
    V8: 1.39165724829804,
    V9: -2.77008927719433,
    V10: -2.77227214465915,
    V11: 3.20203320709635,
    V12: -2.89990738849473,
    V13: -0.595221881324605,
    V14: -4.28925378244217,
    V15: 0.389724120274487,
    V16: -1.14074717980657,
    V17: -2.83005567450437,
    V18: -0.0168224681808257,
    V19: 0.416955705037907,
    V20: 0.126910559061474,
    V21: 0.517232370861764,
    V22: -0.0350493686052974,
    V23: -0.465211076182388,
    V24: 0.320198198514526,
    V25: 0.0445191674731724,
    V26: 0.177839798284401,
    V27: 0.261145002567677,
    V28: -0.143275874698919,
    Amount: 0
  };
  Object.keys(example).forEach((key) => {
    const input = document.getElementById(key);
    if (input) input.value = parseFloat(example[key]);
  });
}