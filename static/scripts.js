
function startLoading() {
  const loadingBar = document.getElementById("loading-bar");
  loadingBar.style.width = "0%";
  loadingBar.style.display = "block";
  setTimeout(() => {
    loadingBar.style.width = "100%";
  }, 50);
}

function stopLoading() {
  const loadingBar = document.getElementById("loading-bar");
  setTimeout(() => {
    loadingBar.style.display = "none";
    loadingBar.style.width = "0%";
  }, 400);
}


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
  await evaluateModel('/predict/randomforest/all', 'rfEvalResult', 'Random Forest');
}

async function evaluateISO() {
  await evaluateModel('/predict/isolationforest/all', 'isoEvalResult', 'Isolation Forest');
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

function setActiveTab(tabId) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

  const activeButton = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
  const activeTab = document.getElementById(tabId);

  if (activeButton) activeButton.classList.add('active');
  if (activeTab) activeTab.classList.add('active');
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

let charts = {
  rf: null,
  iso: null,
  auto: null
};

function renderModelChart(modelKey, stats) {
  const chartId = {
    rf: 'chart-rf',
    iso: 'chart-iso',
    auto: 'chart-auto'
  }[modelKey];

  const ctx = document.getElementById(chartId).getContext('2d');
  const data = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [{
      label: modelKey.toUpperCase(),
      data: [
        stats.accuracy || 0,
        stats.precision || 0,
        stats.recall || 0,
        stats.f1_score || 0
      ],
      backgroundColor: ['#4caf50', '#2196f3', '#ffc107', '#e91e63']
    }]
  };

  if (charts[modelKey]) {
    charts[modelKey].data = data;
    charts[modelKey].update();
  } else {
    charts[modelKey] = new Chart(ctx, {
      type: 'bar',
      data,
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: false
          },
          title: {
            display: true,
            text: `${modelKey.toUpperCase()} Model Metrics`
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1
          }
        }
      }
    });
  }
}

function updateChartWithStats(stats) {
  const modelName = stats.model.toLowerCase().replace(/\s|_/g, '');
  if (modelName === 'randomforest' || modelName === 'rfmodel') {
    renderModelChart('rf', stats);
  } else if (modelName === 'isolationforest'  || modelName === 'isomodel') {
    renderModelChart('iso', stats);
  } else if (modelName === 'autoencoder'  || modelName === 'automodel') {
    renderModelChart('auto', stats);
  }
}


document.addEventListener("DOMContentLoaded", () => {
  // Initial fade-in effect
  document.body.classList.add("fade-in");

  // Setup tab logic
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach(button => {
    button.addEventListener("click", () => {
      const target = button.getAttribute("data-target");

      // Remove active and fade-in/fade-out from all
      tabButtons.forEach(btn => btn.classList.remove("active"));
      tabContents.forEach(content => {
        content.classList.remove("active", "fade-in");
        content.classList.add("fade-out");
      });

      const selectedContent = document.getElementById(target);
      button.classList.add("active");

      setTimeout(() => {
        tabContents.forEach(content => content.classList.remove("fade-out"));
        selectedContent.classList.add("active", "fade-in");
      }, 100); 
    });
  });

  const firstTab = document.querySelector(".tab-button");
  if (firstTab) firstTab.click();
});

document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("fade-in");

  requestAnimationFrame(() => {
    document.body.classList.add("loaded");
  });

  document.querySelectorAll(".tab-button").forEach(button => {
    button.addEventListener("click", () => {
      const targetTabId = button.dataset.tab;
      setActiveTab(targetTabId);
    });
  });

  setActiveTab("dataset");
});