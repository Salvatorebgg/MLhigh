/* ── Global state ─────────────────────────────────────── */
const STATE = {
  uploadId: null,
  fileName: null,
  fileType: null,
  sheetNames: [],
  activeSheet: null,
  columns: [],
  dtypes: {},
  variableTypes: {},
  rowCount: 0,
  colCount: 0,
  previewRows: [],
  summary: {},
  activeMethodId: null,
  activeMethodCategory: 'advanced_stats',
  activeTableType: 'baseline',
  datasetName: null,
  currentPlotlyData: null,
  currentPlotlyLayout: null,
  currentChartBundle: null,
  currentRenderedPlots: [],
  currentTableData: null,
  currentChartSourceData: null,
  currentChartParams: null,
  currentChartResizeObserver: null,
  chartTheme: 'cnsTheme',
  chartPalette: 'default',
  customPalette: null,
  userColors: null,
  markerSize: 8,
  markerShape: 'circle',
  markerOpacity: 0.88,
  lineWidth: 2.5,
  barGap: 0.15,
  chartTitle: '',
  methodWorkspaces: {},
  lastTableData: null,
  lastAnalysisTables: null,
};

/* ── API helpers ──────────────────────────────────────── */
async function apiPost(url, body = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Request failed');
  return res.json();
}

function apiDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || '';
  a.click();
}

/* ── DOM helpers ──────────────────────────────────────── */
function el(id) { return document.getElementById(id); }
function qs(sel, parent) { return (parent || document).querySelector(sel); }
function qsa(sel, parent) { return Array.from((parent || document).querySelectorAll(sel)); }

/* ── Escape helper ────────────────────────────────────── */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

/* ── Toast notifications ──────────────────────────────── */
function toast(msg, type = 'info') {
  const container = el('toastContainer');
  if (!container) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  container.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s ease'; setTimeout(() => t.remove(), 300); }, 3200);
}

/* ── Loading & State ──────────────────────────────────── */
function setLoading(btn, loading) {
  if (loading) {
    btn._origText = btn.textContent;
    btn.textContent = '处理中...';
    btn.disabled = true;
    btn.style.opacity = '0.6';
  } else {
    btn.textContent = btn._origText || btn.textContent;
    btn.disabled = false;
    btn.style.opacity = '1';
  }
}

function resetDatasetState() {
  STATE.uploadId = null;
  STATE.fileName = null;
  STATE.fileType = null;
  STATE.sheetNames = [];
  STATE.activeSheet = null;
  STATE.columns = [];
  STATE.dtypes = {};
  STATE.variableTypes = {};
  STATE.rowCount = 0;
  STATE.colCount = 0;
  STATE.previewRows = [];
  STATE.summary = {};
  STATE.datasetName = null;
}

function getWorkspaceStateFromCurrent() {
  return {
    uploadId: STATE.uploadId,
    fileName: STATE.fileName,
    fileType: STATE.fileType,
    sheetNames: [...(STATE.sheetNames || [])],
    activeSheet: STATE.activeSheet,
    columns: [...(STATE.columns || [])],
    dtypes: { ...(STATE.dtypes || {}) },
    variableTypes: { ...(STATE.variableTypes || {}) },
    rowCount: STATE.rowCount || 0,
    colCount: STATE.colCount || 0,
    previewRows: [...(STATE.previewRows || [])],
    summary: { ...(STATE.summary || {}) },
    datasetName: STATE.datasetName || null,
    methodParams: { ...(STATE.currentChartParams || {}) },
  };
}

function saveActiveMethodWorkspace() {
  if (!STATE.activeMethodId) return;
  STATE.methodWorkspaces[STATE.activeMethodId] = getWorkspaceStateFromCurrent();
}

function loadMethodWorkspace(methodId) {
  const workspace = STATE.methodWorkspaces[methodId];
  if (!workspace) {
    resetDatasetState();
    STATE.currentChartParams = null;
    return;
  }
  STATE.uploadId = workspace.uploadId || null;
  STATE.fileName = workspace.fileName || null;
  STATE.fileType = workspace.fileType || null;
  STATE.sheetNames = [...(workspace.sheetNames || [])];
  STATE.activeSheet = workspace.activeSheet || null;
  STATE.columns = [...(workspace.columns || [])];
  STATE.dtypes = { ...(workspace.dtypes || {}) };
  STATE.variableTypes = { ...(workspace.variableTypes || {}) };
  STATE.rowCount = workspace.rowCount || 0;
  STATE.colCount = workspace.colCount || 0;
  STATE.previewRows = [...(workspace.previewRows || [])];
  STATE.summary = { ...(workspace.summary || {}) };
  STATE.datasetName = workspace.datasetName || null;
  STATE.currentChartParams = { ...(workspace.methodParams || {}) };
}

function saveCurrentMethodParams(params) {
  STATE.currentChartParams = { ...(params || {}) };
  saveActiveMethodWorkspace();
}
