// Show loading indicator
function startLoading() {
  console.log('Starting loading...');
  const loadingBar = document.getElementById("loading-bar");
  loadingBar.style.width = "0%";
  loadingBar.style.display = "block";
  setTimeout(() => {
    loadingBar.style.width = "100%";
  }, 50);
}

// Hide loading indicator
function stopLoading() {
  console.log('Stopping loading...');
  const loadingBar = document.getElementById("loading-bar");
  setTimeout(() => {
    loadingBar.style.display = "none";
    loadingBar.style.width = "0%";
  }, 400);
}

// Load data (dataset tab)
async function loadData() {
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');

  startLoading();
  try {
    const response = await fetch('/preprocess', { method: 'POST' });
    const result = await response.json();

    if (result.sample && Array.isArray(result.sample)) {
      const keys = Object.keys(result.sample[0]);

      // Render Header
      tableHead.innerHTML = '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>';

      // Render Rows
      tableBody.innerHTML = result.sample.map(row =>
        '<tr>' + keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>'
      ).join('');
    } else {
      tableBody.innerHTML = '<tr><td colspan="100%">No data available</td></tr>';
    }

    console.log('Preprocessing Info:', result.info);
  } catch (error) {
    console.error('Gagal mengambil data:', error);
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
    showToast(data.message);
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

    renderModelTable(data, 'model-eval-head', 'model-eval-body');
    document.getElementById('rfEvalResult').textContent = 'Random Forest evaluated on 100 samples.';
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
  body.innerHTML = data.map(row => '<tr>' +
    keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>').join('');
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

// DOM Content loaded
document.addEventListener("DOMContentLoaded", () => {
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");
  const toggleInput = document.querySelector("#theme-toggle");

  function setActiveTab(tabName) {
    console.log(`Setting active tab: ${tabName}`);
    tabButtons.forEach((btn) => btn.classList.remove("active"));
    tabContents.forEach((tab) => tab.classList.remove("active"));

    const targetBtn = document.querySelector(`.tab-button[data-tab="${tabName}"]`);
    const targetTab = document.getElementById(tabName);

    if (targetBtn && targetTab) {
      targetBtn.classList.add("active");
      targetTab.classList.add("active");
    }

    startLoading();

    // If dataset tab, trigger data loading
    if (tabName === "dataset") {
      loadData(); // will stop loading when done
    } else {
      setTimeout(stopLoading, 1000);
    }
  }

  tabButtons.forEach((btn) => {
    btn.dataset.tab = btn.textContent.toLowerCase();
    btn.addEventListener("click", () => {
      const tabName = btn.dataset.tab;
      setActiveTab(tabName);
    });
  });

  toggleInput.addEventListener("change", () => {
    console.log('Toggling dark mode');
    document.body.classList.toggle("dark");
    showToast("Toggled dark mode");
  });

  // Default tab
  setActiveTab("dataset");
});


var myChart = new Chart(ctx, {
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


function updateChartWithStats(stats) {
  myChart.data.labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
  myChart.data.datasets[0].label = 'Model Performance';
  myChart.data.datasets[0].data = [
    stats.accuracy ?? 0,
    stats.precision ?? 0,
    stats.recall ?? 0,
    stats.f1_score ?? 0
  ];
  myChart.update();
}
