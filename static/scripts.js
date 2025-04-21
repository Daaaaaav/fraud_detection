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

// Evaluate Random Forest
async function evaluateRF() {
  startLoading();
  try {
    const res = await fetch('/predict/randomforest/all');
    const data = await res.json();

    if (data && data.metrics) {
      renderModelTable(data.metrics, 'model-eval-head', 'model-eval-body');
      document.getElementById('rfEvalResult').textContent = 'Random Forest evaluated on 100 samples.';
      updateChartWithStats(data.metrics);
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

// Evaluate Isolation Forest
async function evaluateISO() {
  startLoading();
  try {
    const res = await fetch('/predict/isolationforest/all');
    const data = await res.json();

    renderModelTable(data, 'model-eval-head', 'model-eval-body');
    document.getElementById('isoEvalResult').textContent = 'Isolation Forest evaluated on 100 samples.';
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

// Dark mode toggle
const toggleInput = document.getElementById("toggle-input");
if (toggleInput) {
  toggleInput.addEventListener("change", () => {
    document.body.classList.toggle("dark");
    showToast("Toggled dark mode");
  });
}

// Initialize default tab
setActiveTab("dataset");

// Initialize Chart
const ctx = document.getElementById('performanceChart');
if (ctx) {
  window.myChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
      datasets: [{
        label: 'Model Performance',
        data: [0, 0, 0, 0],
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          ticks: {
            stepSize: 0.1
          }
        }
      }
    }
  });
}

// Chart updater
function updateChartWithStats(stats) {
  if (!window.myChart) return;
  const dataset = window.myChart.data.datasets[0];
  dataset.data = [
    stats.accuracy ?? 0,
    stats.precision ?? 0,
    stats.recall ?? 0,
    stats.f1_score ?? 0
  ];
  window.myChart.update();
}

// Train Autoencoder
async function trainAutoencoder() {
  startLoading();
  try {
    const res = await fetch('/train/autoencoder', { method: 'POST' });
    const result = await res.json();

    const output = document.getElementById('auto-output');
    output.textContent = `Autoencoder Training:\n${JSON.stringify(result, null, 2)}`;
  } catch (error) {
    console.error('Error training autoencoder:', error);
    document.getElementById('auto-output').textContent = 'Failed to train autoencoder.';
  } finally {
    stopLoading();
  }
}

async function predictAutoencoder() {
  startLoading();
  try {
    const res = await fetch('/predict/autoencoder/all');
    const result = await res.json();
    
    console.log("Autoencoder prediction result:", result);

    const output = document.getElementById('auto-output');

    if (Array.isArray(result)) {
      output.textContent = `Predictions (showing first 5):\n${JSON.stringify(result.slice(0, 5), null, 2)}`;
    } else if (Array.isArray(result.predictions)) {
      output.textContent = `Predictions (showing first 5):\n${JSON.stringify(result.predictions.slice(0, 5), null, 2)}`;
    } else {
      output.textContent = `Autoencoder Prediction Response:\n${JSON.stringify(result, null, 2)}`;
    }
  } catch (error) {
    console.error('Error predicting with autoencoder:', error);
    document.getElementById('auto-output').textContent = 'Failed to get predictions.';
  } finally {
    stopLoading();
  }
}
