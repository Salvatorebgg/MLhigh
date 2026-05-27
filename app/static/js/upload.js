/* ── File Upload Module ────────────────────────────────── */

let _uploadAbortController = null;

async function handleFile(file, options = {}) {
  const uploadBtn = el('uploadDataBtn');

  if (_uploadAbortController) {
    _uploadAbortController.abort();
  }
  _uploadAbortController = new AbortController();
  const timeoutId = setTimeout(() => _uploadAbortController.abort(), 120000);

  if (typeof setStatus === 'function') setStatus('正在读取文件...');
  if (uploadBtn) setLoading(uploadBtn, true);

  const formData = new FormData();
  formData.append('file', file);

  try {
    if (options.fromMethod) toast(`正在读取 ${file.name}...`, 'info');

    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
      signal: _uploadAbortController.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    updateStateFromData(data);
    STATE.datasetName = null;

    if (typeof setStatus === 'function') setStatus(`文件 "${data.filename}" 已解析`);
    toast('数据上传成功！', 'success');

    if (data.sheet_names && data.sheet_names.length > 1) {
      renderSheetSelector(data.sheet_names);
    }

    // Update UI panels
    if (typeof updateMetricGrid === 'function') updateMetricGrid();
    if (typeof renderDataPreview === 'function') renderDataPreview();
    if (typeof updateDatasetMeta === 'function') updateDatasetMeta();
    if (typeof renderDataPanel === 'function') renderDataPanel();
    if (typeof buildMethodVarControls === 'function') buildMethodVarControls();
    if (typeof renderAppearanceControls === 'function') renderAppearanceControls();
    if (typeof buildTableVarControls === 'function') buildTableVarControls();
    if (typeof updateFlowLine === 'function') updateFlowLine(2);
  } catch (err) {
    clearTimeout(timeoutId);
    const msg = err.name === 'AbortError' ? '上传超时，请尝试较小的文件' : '上传失败: ' + err.message;
    if (typeof setStatus === 'function') setStatus(msg, true);
    toast(msg, 'error');
  } finally {
    if (uploadBtn) setLoading(uploadBtn, false);
    _uploadAbortController = null;
  }
}

function updateStateFromData(data) {
  STATE.uploadId = data.upload_id;
  STATE.fileName = data.filename;
  STATE.fileType = data.file_type;
  STATE.sheetNames = data.sheet_names || [];
  STATE.activeSheet = data.sheet_name || null;
  STATE.columns = data.columns;
  STATE.dtypes = data.dtypes || {};
  STATE.variableTypes = data.variable_types || {};
  STATE.rowCount = data.row_count;
  STATE.colCount = data.col_count;
  STATE.previewRows = data.preview || [];
  STATE.summary = data.summary || {};
  saveActiveMethodWorkspace();
}

function renderSheetSelector(sheets) {
  const sheetRow = el('sheetRow');
  if (!sheetRow) return;
  sheetRow.hidden = false;
  const sheetSelect = el('sheetSelect');
  if (!sheetSelect) return;
  sheetSelect.innerHTML = sheets.map(s => `<option value="${s}">${s}</option>`).join('');
  sheetSelect.value = STATE.activeSheet || sheets[0];

  const newSelect = sheetSelect.cloneNode(true);
  sheetSelect.parentNode.replaceChild(newSelect, sheetSelect);
  newSelect.addEventListener('change', async () => {
    const sheet = newSelect.value;
    if (typeof setStatus === 'function') setStatus('切换工作表中...');
    try {
      const data = await apiPost('/api/read-sheet', { upload_id: STATE.uploadId, sheet_name: sheet });
      updateStateFromData({ ...data, sheet_name: sheet, filename: STATE.fileName, file_type: STATE.fileType, sheet_names: STATE.sheetNames });
      if (typeof updateMetricGrid === 'function') updateMetricGrid();
      if (typeof renderDataPreview === 'function') renderDataPreview();
      if (typeof buildMethodVarControls === 'function') buildMethodVarControls();
      if (typeof renderAppearanceControls === 'function') renderAppearanceControls();
      if (typeof renderDataPanel === 'function') renderDataPanel();
      toast(`已切换到工作表: ${sheet}`, 'success');
    } catch (e) {
      toast('切换失败: ' + e.message, 'error');
    }
  });
}

/* ── Example Dataset Loading ───────────────────────────── */
async function loadExampleDataset(name, options = {}) {
  const btn = el('loadExampleBtn');
  if (btn && !options.silent) setLoading(btn, true);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const res = await fetch(`/api/examples/${name}`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || '请求失败');
    }
    const data = await res.json();
    updateStateFromData({ ...data, upload_id: null, filename: data.filename, file_type: '.csv' });
    STATE.datasetName = name;
    saveActiveMethodWorkspace();

    if (typeof updateMetricGrid === 'function') updateMetricGrid();
    if (typeof renderDataPreview === 'function') renderDataPreview();
    if (typeof updateDatasetMeta === 'function') updateDatasetMeta();
    if (typeof buildMethodVarControls === 'function') buildMethodVarControls();
    if (typeof renderAppearanceControls === 'function') renderAppearanceControls();
    if (typeof buildTableVarControls === 'function') buildTableVarControls();
    if (typeof updateFlowLine === 'function') updateFlowLine(2);

    if (!options.silent) toast('示例数据加载成功！', 'success');
    return data;
  } catch (e) {
    clearTimeout(timeoutId);
    const msg = e.name === 'AbortError' ? '请求超时，请检查网络' : '加载失败: ' + e.message;
    toast(msg, 'error');
    throw e;
  } finally {
    if (btn && !options.silent) setLoading(btn, false);
  }
}
