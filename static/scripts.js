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
  document.getElementById('spinner')?.style.setProperty('display', 'block');
}

function hideSpinner() {
  document.getElementById('spinner')?.style.setProperty('display', 'none');
}

// ========== Toast Notification ==========
function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'toast-container';
  toast.innerHTML = `<div class="toast-message">${message}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ========== Tab Switching ==========
function setActiveTab(tabId) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

  document.querySelector(`.tab-button[data-tab="${tabId}"]`)?.classList.add('active');
  document.getElementById(tabId)?.classList.add('active');
}

// ========== Load Data into Table ==========
async function loadData() {
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');
  startLoading();
  try {
    const response = await fetch('/preprocess', { method: 'POST' });
    const result = await response.json();

    if (Array.isArray(result.sample) && result.sample.length) {
      const keys = Object.keys(result.sample[0]);
      tableHead.innerHTML = `<tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
      tableBody.innerHTML = result.sample.map(row => 
        `<tr>${keys.map(k => `<td>${row[k]}</td>`).join('')}</tr>`
      ).join('');
    } else {
      tableBody.innerHTML = `<tr><td colspan="100%">No data available</td></tr>`;
    }
    console.log('Preprocessing Info:', result.info);
  } catch (error) {
    console.error('Error loading data:', error);
    tableBody.innerHTML = `<tr><td colspan="100%">Error loading data.</td></tr>`;
  } finally {
    stopLoading();
  }
}

const charts = { rf: null, iso: null, auto: null };

function renderModelChart(modelKey, stats) {
  const chartIds = { rf: 'chart-rf', iso: 'chart-iso', auto: 'chart-auto' };
  const ctx = document.getElementById(chartIds[modelKey])?.getContext('2d');
  if (!ctx) return;

  const data = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [{
      label: modelKey.toUpperCase(),
      data: [
        stats.accuracy ?? 0,
        stats.precision ?? 0,
        stats.recall ?? 0,
        stats.f1_score ?? 0
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
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
}

function updateChartWithStats(stats) {
  if (!stats?.model) return;
  const modelKey = stats.model.toLowerCase().includes('auto') ? 'auto'
                  : stats.model.toLowerCase().includes('isolation') ? 'iso'
                  : 'rf';
  renderModelChart(modelKey, stats);
}

// ========== Training Functions ==========
async function trainModel(endpoint, modelName, resultId, metricId) {
  startLoading();
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName })
    });
    const data = await res.json();

    if (resultId) document.getElementById(resultId).textContent = JSON.stringify(data, null, 2);
    if (metricId && data) {
      document.getElementById(metricId).textContent = `
        Model: ${data.model || modelName}
        Accuracy: ${(data.accuracy || 0).toFixed(2)}%
        Precision: ${(data.precision || 0).toFixed(2)}
        Recall: ${(data.recall || 0).toFixed(2)}
        F1 Score: ${(data.f1_score || 0).toFixed(2)}
        Train Samples: ${data.train_samples}
        Test Samples: ${data.test_samples}
      `.trim();
    }

    showToast(data.message || `${modelName} trained.`);
    updateChartWithStats(data);
  } catch (error) {
    console.error(`Error training ${modelName}:`, error);
    showToast(`Failed to train ${modelName}.`);
  } finally {
    stopLoading();
  }
}

const trainRF = () => trainModel('/train/randomforest', 'rf_model', 'trainRFResult', 'rf-metrics');
const trainISO = () => trainModel('/train/isolationforest', 'iso_model', 'trainISOResult');
const trainCombined = () => trainModel('/train/combined', 'rf_model.pkl', 'trainCombinedResult');

// ========== Evaluation Functions ==========
async function evaluateModel(endpoint, resultId, modelName) {
  startLoading();
  try {
    const res = await fetch(endpoint);
    const data = await res.json();
    const { predictions, stats } = data;

    if (Array.isArray(predictions) && predictions.length) {
      renderModelTable(predictions, 'model-eval-head', 'model-eval-body');
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

const evaluateRF = () => evaluateModel('/predict/randomforest/all', 'rfEvalResult', 'Random Forest');
const evaluateISO = () => evaluateModel('/predict/isolationforest/all', 'isoEvalResult', 'Isolation Forest');

async function evaluateCombined() {
  try {
    showSpinner();
    const response = await fetch('/predict/combined');
    const { stats } = await response.json();
    document.getElementById('combinedEvalResult').textContent = JSON.stringify(stats, null, 2);
  } catch (error) {
    console.error('Error evaluating combined model:', error);
    alert('Error evaluating combined model.');
  } finally {
    hideSpinner();
  }
}

// ========== Utility Function ==========
function renderModelTable(predictions, headId, bodyId) {
  const tableHead = document.getElementById(headId);
  const tableBody = document.getElementById(bodyId);
  if (!predictions.length) return;

  const keys = Object.keys(predictions[0]);
  tableHead.innerHTML = `<tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;
  tableBody.innerHTML = predictions.map(row =>
    `<tr>${keys.map(k => `<td>${row[k]}</td>`).join('')}</tr>`
  ).join('');
}

document.querySelectorAll('.tab-button').forEach(button =>
  button.addEventListener('click', () => setActiveTab(button.dataset.tab))
);

const toggleInput = document.getElementById('toggle-input');
if (toggleInput) {
  if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark');
    toggleInput.checked = true;
  }
  toggleInput.addEventListener('change', () => {
    document.body.classList.toggle('dark');
    localStorage.setItem('darkMode', document.body.classList.contains('dark'));
    showToast('Toggled dark mode');
  });
}

setActiveTab('dataset');

document.getElementById('user-input-form')?.addEventListener('submit', async function (e) {
  e.preventDefault();
  const model = document.getElementById('model-select').value;
  const inputs = document.querySelectorAll('#input-fields input');
  const inputData = {};

  inputs.forEach(input => inputData[input.name] = parseFloat(input.value));

  showSpinner();
  let endpoint = '';
  switch (model) {
    case 'rf':
      endpoint = '/predict/randomforest/user';
      break;
    case 'iso':
      endpoint = '/predict/isolationforest/user';
      break;
    case 'auto':
      endpoint = '/predict/autoencoder/user';
      break;
  }

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputData)
    });
    const result = await res.json();
    if (result.error) throw new Error(result.error);

    document.getElementById('prediction-result').innerText = `Prediction: ${result.label}`;
  } catch (err) {
    showToast(`Prediction failed: ${err.message}`);
  } finally {
    hideSpinner();
  }
});

async function renderInputFields() {
  try {
    const response = await fetch('/preprocess', { method: 'POST' });
    const result = await response.json();
    const sample = result.sample[0];
    const container = document.getElementById('input-fields');
    container.innerHTML = '';

    Object.keys(sample).forEach(key => {
      const input = document.createElement('input');
      input.name = key;
      input.placeholder = key;
      input.type = 'number';
      container.appendChild(input);
    });
  } catch (e) {
    console.error('Failed to render input fields:', e);
  }
}

renderInputFields();

document.getElementById("user-input-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const model = document.getElementById("model-select").value;
  const inputFields = document.querySelectorAll("#input-fields input");
  const values = Array.from(inputFields).map(i => parseFloat(i.value));
  
  if (values.some(isNaN)) {
    showToast("Please enter valid numbers.");
    return;
  }

  startLoading();
  try {
    const response = await fetch(`/predict/${model}/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: values })
    });

    const result = await response.json();
    const resultDiv = document.getElementById("prediction-result");
    
    if (result.error) {
      resultDiv.innerHTML = `<p style="color:red;">${result.error}</p>`;
    } else {
      resultDiv.innerHTML = `
        <p><strong>MSE:</strong> ${result.mse.toFixed(6)}</p>
        <p><strong>Threshold:</strong> ${result.threshold.toFixed(6)}</p>
        <p><strong>Prediction:</strong> ${result.is_fraud ? "Fraudulent" : "Normal"}</p>
      `;
    }
  } catch (error) {
    console.error("Prediction error:", error);
    showToast("Failed to predict.");
  } finally {
    stopLoading();
  }
});

function generateInputFields(numFields = 30) {
  const container = document.getElementById("input-fields");
  container.innerHTML = "";
  for (let i = 0; i < numFields; i++) {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.placeholder = `Feature ${i + 1}`;
    container.appendChild(input);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  generateInputFields(30); 
});

const inputFieldsContainer = document.getElementById("input-fields");
const featureNames = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

featureNames.forEach((feature) => {
  const label = document.createElement("label");
  label.for = feature;
  label.textContent = feature;

  const input = document.createElement("input");
  input.type = "number";
  input.name = feature;
  input.id = feature;
  input.required = true;

  inputFieldsContainer.appendChild(label);
  inputFieldsContainer.appendChild(input);
});

function fillExample() {
  const example = {
    Time: 12345,
    V1: -1.2, V2: 0.3, V3: -0.5, V4: 1.3, V5: 0.9,
    V6: -1.1, V7: 0.2, V8: 0.4, V9: -0.2, V10: 1.0,
    V11: -1.3, V12: 0.7, V13: -0.1, V14: 0.3, V15: -0.8,
    V16: 0.6, V17: -0.5, V18: 1.4, V19: -1.2, V20: 0.2,
    V21: 0.9, V22: -0.7, V23: 0.1, V24: -0.3, V25: 0.8,
    V26: -0.9, V27: 0.5, V28: 0.3, Amount: 250.00
  };
  Object.keys(example).forEach(key => {
    const input = document.getElementById(key);
    if (input) input.value = example[key];
  });
}