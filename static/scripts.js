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
  }
  
  async function trainISO() {
    const res = await fetch('/train/isolationforest', { method: 'POST' });
    const data = await res.json();
    document.getElementById('trainISOResult').textContent = JSON.stringify(data, null, 2);
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
  