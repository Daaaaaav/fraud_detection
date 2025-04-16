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

async function loadData() {
  const tableHead = document.getElementById('table-head');
  const tableBody = document.getElementById('table-body');

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

    // Optional: console log
    console.log('Preprocessing Info:', result.info);
  } catch (error) {
    console.error('Gagal mengambil data:', error);
    tableBody.innerHTML = '<tr><td colspan="100%">Terjadi kesalahan saat memuat data.</td></tr>';
  }
}

async function trainRF() {
  const res = await fetch('/train/randomforest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: 'rf_model' })
  });
  const data = await res.json();
  document.getElementById('trainRFResult').textContent = JSON.stringify(data, null, 2);
  showToast(data.message); 
}


async function trainISO() {
  const res = await fetch('/train/isolationforest', { method: 'POST' });
  const data = await res.json();
  document.getElementById('trainISOResult').textContent = JSON.stringify(data, null, 2);
  showToast(data.message);
}


async function evaluateRF() {
  const res = await fetch('/predict/randomforest/all');
  const data = await res.json();

  renderModelTable(data, 'model-eval-head', 'model-eval-body');
  document.getElementById('rfEvalResult').textContent = 'Random Forest evaluated on 100 samples.';
}

async function evaluateISO() {
  const res = await fetch('/predict/isolationforest/all');
  const data = await res.json();

  renderModelTable(data, 'model-eval-head', 'model-eval-body');
  document.getElementById('isoEvalResult').textContent = 'Isolation Forest evaluated on 100 samples.';
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
  body.innerHTML = data.map(row => '<tr>' +
    keys.map(k => `<td>${row[k]}</td>`).join('') + '</tr>').join('');
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
    setTimeout(stopLoading, 1000);
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

  setActiveTab("dataset");
});