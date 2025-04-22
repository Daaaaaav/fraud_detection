// Show loading indicator
function startLoading() {
  const loadingBar = document.getElementById("loading-bar");
  loadingBar.style.width = "0%";
  loadingBar.style.display = "block";
  setTimeout(() => {
    loadingBar.style.width = "100%";
  }, 50);
}

// Hide loading indicator
function stopLoading() {
  const loadingBar = document.getElementById("loading-bar");
  setTimeout(() => {
    loadingBar.style.display = "none";
    loadingBar.style.width = "0%";
  }, 400);
}

// Load dataset
async function loadData() {
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');
  startLoading();
  try {
    const response = await fetch('/preprocess', { method: 'POST' });
    const result = await response.json();

    if (result.sample && Array.isArray(result.sample)) {
      const keys = Object.keys(result.sample[0]);
      tableHead.innerHTML = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
      tableBody.innerHTML = result.sample.map(row =>
        '<tr>' + keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>'
      ).join('');
    } else {
      tableBody.innerHTML = '<tr><td colspan="100%">No data available</td></tr>';
    }

    console.log('Preprocessing Info:', result.info);
  } catch (error) {
    console.error('Error loading data:', error);
    tableBody.innerHTML = '<tr><td colspan="100%">Terjadi kesalahan saat memuat data.</td></tr>';
  } finally {
    stopLoading();
  }
}

// Train Random Forest
async function trainRF() {
  startLoading();
  try {
    const res = await fetch('/train/randomforest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'rf_model' })
    });

    const data = await res.json();
    document.getElementById('trainRFResult').textContent = JSON.stringify(data, null, 2);

    if (document.getElementById('rf-metrics')) {
      document.getElementById('rf-metrics').textContent = `
      Model: ${data.model || "Random Forest"}
      Accuracy: ${(data.accuracy || 0).toFixed(2)}%
      Precision: ${(data.precision || 0).toFixed(2)}
      Recall: ${(data.recall || 0).toFixed(2)}
      F1 Score: ${(data.f1_score || 0).toFixed(2)}
      Train Samples: ${data.train_samples}
      Test Samples: ${data.test_samples}
      `.trim();
    }

    showToast(data.message || "Random Forest trained.");
    updateChartWithStats(data);
  } catch (err) {
    console.error("Error training RF:", err);
    showToast("Failed to train Random Forest model.");
  } finally {
    stopLoading();
  }
}

// Train Isolation Forest
async function trainISO() {
  startLoading();
  try {
    const res = await fetch('/train/isolationforest', { method: 'POST' });
    const data = await res.json();

    document.getElementById('trainISOResult').textContent = JSON.stringify(data, null, 2);
    showToast(data.message);
    updateChartWithStats(data);
  } catch (err) {
    console.error("Error training ISO:", err);
    showToast("Failed to train Isolation Forest model.");
  } finally {
    stopLoading();
  }
}

// Train Autoencoder
async function trainAutoencoder() {
  startLoading();
  try {
    const res = await fetch('/train/autoencoder', { method: 'POST' });
    const result = await res.json();

    const output = document.getElementById('auto-output');
    output.textContent = `Autoencoder Training:\n${JSON.stringify(result, null, 2)}`;
    showToast(result.message || "Autoencoder training completed.");
    updateChartWithStats(result);
  } catch (error) {
    console.error('Error training autoencoder:', error);
    document.getElementById('auto-output').textContent = 'Failed to train autoencoder.';
    showToast("Training failed for Autoencoder.");
  } finally {
    stopLoading();
  }
}

// Predict Autoencoder
async function predictAutoencoder() {
  startLoading();
  try {
    const res = await fetch('/predict/autoencoder/all');
    const result = await res.json();

    console.log("Autoencoder prediction result:", result);
    const output = document.getElementById('auto-output');

    if (Array.isArray(result)) {
      output.textContent = `Predictions (first 5):\n${JSON.stringify(result.slice(0, 5), null, 2)}`;
    } else if (Array.isArray(result.predictions)) {
      output.textContent = `Predictions (first 5):\n${JSON.stringify(result.predictions.slice(0, 5), null, 2)}`;
    } else {
      output.textContent = `Autoencoder Prediction:\n${JSON.stringify(result, null, 2)}`;
    }

    // Update chart with stats
    if (result.stats) {
      updateChartWithStats(result.stats);
    }
  } catch (error) {
    console.error('Error predicting with autoencoder:', error);
    document.getElementById('auto-output').textContent = 'Failed to get predictions.';
    showToast("Autoencoder prediction failed.");
  } finally {
    stopLoading();
  }
}

async function evaluateRF() {
  startLoading();
  try {
    const res = await fetch('/predict/randomforest/all');
    const data = await res.json();

    console.log("RF Evaluation Response:", data); // <--- Added to debug

    const metrics = data.metrics || (Array.isArray(data) ? data : null);

    if (metrics && Array.isArray(metrics) && metrics.length > 0) {
      renderModelTable(metrics, 'model-eval-head', 'model-eval-body');
      document.getElementById('rfEvalResult').textContent = 'Random Forest evaluated on 100 samples.';
      updateChartWithStats(metrics[0]); // Assuming first row is representative
    } else {
      document.getElementById('rfEvalResult').textContent = 'No evaluation data available for Random Forest.';
      showToast("No evaluation data available.");
    }
  } catch (err) {
    console.error("Error evaluating RF:", err);
    showToast("Failed to evaluate Random Forest.");
  } finally {
    stopLoading();
  }
}

async function evaluateModel(endpoint, resultId, modelName) {
  startLoading();
  try {
    const res = await fetch(endpoint);
    const data = await res.json();

    console.log(`${modelName} Evaluation Response:`, data);

    const metrics = data.metrics || (Array.isArray(data) ? data : null);
    if (metrics && Array.isArray(metrics) && metrics.length > 0) {
      renderModelTable(metrics, 'model-eval-head', 'model-eval-body');
      document.getElementById(resultId).textContent = `${modelName} evaluated on ${metrics.length} samples.`;
      updateChartWithStats(metrics[0]);
    } else {
      document.getElementById(resultId).textContent = `No evaluation data available for ${modelName}.`;
      showToast(`No evaluation data for ${modelName}.`);
    }
  } catch (err) {
    console.error(`Error evaluating ${modelName}:`, err);
    showToast(`Failed to evaluate ${modelName}.`);
  } finally {
    stopLoading();
  }
}


// Evaluate Isolation Forest
async function evaluateISO() {
  startLoading();
  try {
    const res = await fetch('/predict/isolationforest/all');
    const data = await res.json();

    renderModelTable(data, 'model-eval-head', 'model-eval-body');
    document.getElementById('isoEvalResult').textContent = 'Isolation Forest evaluated on 100 samples.';
    updateChartWithStats(data);
  } catch (err) {
    console.error("Error evaluating ISO:", err);
    showToast("Failed to evaluate Isolation Forest.");
  } finally {
    stopLoading();
  }
}

// Render model evaluation table
function renderModelTable(data, headId, bodyId) {
  const head = document.getElementById(headId);
  const body = document.getElementById(bodyId);

  if (!data || !Array.isArray(data) || data.length === 0) {
    body.innerHTML = '<tr><td colspan="100%">No data</td></tr>';
    return;
  }

  const keys = Object.keys(data[0]);
  head.innerHTML = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';
  body.innerHTML = data.map(row =>
    '<tr>' + keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>'
  ).join('');
}

// Toast notification
function showToast(message) {
  const toastContainer = document.createElement('div');
  toastContainer.className = 'toast-container';

  const toastMessage = document.createElement('div');
  toastMessage.className = 'toast-message';
  toastMessage.textContent = message;

  toastContainer.appendChild(toastMessage);
  document.body.appendChild(toastContainer);

  setTimeout(() => {
    toastContainer.remove();
  }, 3000);
}

// Tab switching
function setActiveTab(tabId) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelector(`.tab-button[data-tab="${tabId}"]`)?.classList.add('active');
  document.getElementById(tabId)?.classList.add('active');
}

document.querySelectorAll('.tab-button').forEach(button => {
  button.addEventListener('click', () => {
    const targetTabId = button.dataset.tab;
    setActiveTab(targetTabId);
  });
});

const toggleInput = document.getElementById("toggle-input");
if (toggleInput) {
  if (localStorage.getItem("darkMode") === "true") {
    document.body.classList.add("dark");
    toggleInput.checked = true;
  }

  toggleInput.addEventListener("change", () => {
    document.body.classList.toggle("dark");
    localStorage.setItem("darkMode", document.body.classList.contains("dark"));
    showToast("Toggled dark mode");
  });
}

setActiveTab("dataset");

// Chart.js instance and update function
let performanceChartInstance = null;

function updateChartWithStats(data) {
  const canvas = document.getElementById('performanceChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const label = data.model || 'Autoencoder';
  const metrics = [
    data.accuracy || 0,
    data.precision || 0,
    data.recall || 0,
    data.f1_score || 0
  ];

  const chartData = {
    labels: ['Accuracy (%)', 'Precision', 'Recall', 'F1 Score'],
    datasets: [{
      label: label,
      data: metrics,
      backgroundColor: ['#4CAF50', '#2196F3', '#FFC107', '#E91E63'],
      borderWidth: 1
    }]
  };

  const chartOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 100
      }
    }
  };

  if (performanceChartInstance) {
    performanceChartInstance.destroy();
  }

  performanceChartInstance = new Chart(ctx, {
    type: 'bar',
    data: chartData,
    options: chartOptions
  });
}
