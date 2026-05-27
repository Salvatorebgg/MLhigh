/* ── MLhigh Main Application ─────────────────────────── */

document.addEventListener('DOMContentLoaded', async () => {
  await loadMethodCatalog();
  initMethodCategoryTabs();
  initMethodGrid();
  renderMiniMethodGrid(STATE.activeMethodCategory || 'advanced_stats');
  initCenterTabs();
  initFileInputs();
  initTableTypeTabs();
  initChartThemeSelect();
  initExportButtons();
  initChartTitleInput();
  loadExampleList();
  bootEmptyState();

  const genBtn = el('generateBtn');
  if (genBtn) genBtn.addEventListener('click', runAnalysis);

  const tableGenBtn = el('generateTableBtn');
  if (tableGenBtn) tableGenBtn.addEventListener('click', generateTable);
});

// ── Method Category Tabs ──────────────────────────────
function initMethodCategoryTabs() {
  qsa('#methodCatTabs .cat-tab').forEach(tab => {
    tab.addEventListener('click', function() {
      qsa('#methodCatTabs .cat-tab').forEach(t => t.classList.remove('active'));
      this.classList.add('active');
      STATE.activeMethodCategory = this.dataset.cat;
      renderMiniMethodGrid(this.dataset.cat);
    });
  });
}

// ── Mini Method Grid (event delegation like Basicpicture) ──
function initMethodGrid() {
  const grid = el('miniMethodGrid');
  if (grid) {
    grid.addEventListener('click', function(e) {
      const card = e.target.closest('.mini-method-card');
      if (!card) return;
      const methodId = card.dataset.method;
      if (methodId) selectMethod(methodId);
    });
  }
}

function renderMiniMethodGrid(category) {
  const grid = el('miniMethodGrid');
  if (!grid) return;
  const methods = Object.values(METHOD_CATALOG).filter(m => m.category === category);
  grid.innerHTML = methods.map(m => `
    <div class="mini-method-card ${STATE.activeMethodId === m.id ? 'selected' : ''}" data-method="${m.id}">
      <span class="mini-method-icon">${m.icon}</span>
      <span class="mini-method-name">${m.name}</span>
      <span class="mini-method-desc">${m.description || ''}</span>
    </div>
  `).join('');
}

function selectMethod(methodId) {
  if (STATE.activeMethodId && STATE.activeMethodId !== methodId) {
    saveActiveMethodWorkspace();
  }

  STATE.activeMethodId = methodId;
  loadMethodWorkspace(methodId);
  STATE.currentPlotlyData = null;
  STATE.currentPlotlyLayout = null;
  STATE.currentChartSourceData = null;

  const config = getMethodConfig(methodId);

  if (!STATE.uploadId && !STATE.datasetName && config && config.example_dataset) {
    STATE.datasetName = config.example_dataset;
  }

  qsa('.mini-method-card').forEach(c => c.classList.remove('selected'));
  const activeCard = qs(`.mini-method-card[data-method="${methodId}"]`);
  if (activeCard) activeCard.classList.add('selected');

  const label = el('selectedMethodLabel');
  if (label) label.textContent = config ? config.name : methodId;

  const previewTitle = el('analysisTitle');
  if (previewTitle) previewTitle.textContent = config ? config.name + ' — 分析结果' : '分析结果';

  updateFlowLine(1);
  resetResultsPanel();
  renderDataPanel();
  buildMethodVarControls();
  renderAppearanceControls();
  updateMetricGrid();
  renderDataPreview();
  updateDatasetMeta();
  const dlBtn = el('downloadExampleBtn');
  if (dlBtn) dlBtn.hidden = true;
}

function resetResultsPanel() {
  disconnectResizeObserver();
  const vizContainer = el('chartPreviewContainer');
  if (vizContainer) {
    vizContainer.innerHTML = '<div class="empty-state">选择分析方法并加载数据后，点击"运行分析"</div>';
  }
  const exportBar = el('chartExportBar');
  if (exportBar) exportBar.style.display = 'none';

  const tablesContainer = el('resultTablesContainer');
  if (tablesContainer) tablesContainer.innerHTML = '<div class="empty-state small">表格结果将在分析后显示</div>';

  const diagContainer = el('diagnosticsContainer');
  if (diagContainer) diagContainer.innerHTML = '<div class="empty-state small">诊断图表将在分析后显示</div>';

  const discContainer = el('discussionContainer');
  if (discContainer) discContainer.innerHTML = '<div class="empty-state small">自动生成的结果解释将在分析后显示</div>';
}

// ── Chart Theme Select ─────────────────────────────────
function initChartThemeSelect() {
  const sel = el('chartThemeSelect');
  if (!sel) return;
  sel.value = STATE.chartTheme || 'cnsTheme';
  sel.addEventListener('change', () => {
    STATE.chartTheme = sel.value;
    STATE.userColors = null;
    renderAppearanceControls();
    toast('主题: ' + ((CHART_THEMES[sel.value] && CHART_THEMES[sel.value].name) || sel.value), 'info');
    if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
  });
}

function initChartTitleInput() {
  const input = el('chartTitleInput');
  if (!input) return;
  input.addEventListener('input', () => {
    STATE.chartTitle = input.value;
  });
}

// ── Run Analysis ────────────────────────────────────────
async function runAnalysis() {
  if (!STATE.activeMethodId) {
    toast('请先选择分析方法', 'info');
    return;
  }
  const hasData = (STATE.columns || []).length > 0;
  if (!hasData && !STATE.datasetName) {
    toast('请先加载数据', 'info');
    return;
  }

  const config = getMethodConfig(STATE.activeMethodId);
  if (!STATE.datasetName && config && config.example_dataset) {
    STATE.datasetName = config.example_dataset;
  }

  const btn = el('generateBtn');
  if (btn) setLoading(btn, true);
  setStatus('正在运行分析...');

  try {
    const params = collectMethodParams();
    saveCurrentMethodParams(params);
    const payload = {
      method_id: STATE.activeMethodId,
      use_demo: !STATE.uploadId,
      upload_id: STATE.uploadId || undefined,
      dataset_name: STATE.datasetName || undefined,
      params: params,
    };

    const result = await apiPost('/api/analyze', payload);

    // Save chart data for re-rendering
    STATE.currentPlotlyData = result.charts || [];
    STATE.currentChartSourceData = result.data_summary || {};

    // Render results
    renderResultTables(result.tables || []);
    renderAllCharts(result.charts || []);
    renderAllDiagCharts(result.diagnostics || []);
    renderDiscussion(result.discussion || '');

    // Show export bar
    const exportBar = el('chartExportBar');
    if (exportBar) exportBar.style.display = 'flex';

    // Update data summary
    if (result.data_summary) {
      STATE.summary = result.data_summary;
      updateMetricGrid();
    }

    updateFlowLine(4);
    setStatus('分析完成');
    toast(`${result.method_name || STATE.activeMethodId} 分析完成`, 'success');

    // Switch to visualization tab
    const tabs = qsa('.result-tabs .tab');
    const vizTab = Array.from(tabs).find(t => t.dataset.tab === 'visualization');
    if (vizTab) vizTab.click();

    // Observe chart containers for responsive resizing
    initResizeObserver();

  } catch (e) {
    toast('分析失败: ' + e.message, 'error');
    setStatus('分析失败', true);
    console.error(e);
  } finally {
    if (btn) setLoading(btn, false);
  }
}

// ── ResizeObserver for responsive charts ───────────────
function initResizeObserver() {
  disconnectResizeObserver();
  if (!window.Plotly) return;
  STATE.currentChartResizeObserver = new ResizeObserver((entries) => {
    entries.forEach(entry => {
      const plotEl = entry.target.querySelector('.js-plotly-plot');
      if (plotEl && plotEl._fullLayout) {
        Plotly.Plots.resize(plotEl);
      }
    });
  });
  qsa('.result-chart-container').forEach(el => {
    STATE.currentChartResizeObserver.observe(el);
  });
}

function disconnectResizeObserver() {
  if (STATE.currentChartResizeObserver) {
    STATE.currentChartResizeObserver.disconnect();
    STATE.currentChartResizeObserver = null;
  }
}

// ── Center Panel Tabs ──────────────────────────────────
function initCenterTabs() {
  qsa('.result-tabs .tab').forEach(tab => {
    tab.addEventListener('click', function() {
      qsa('.result-tabs .tab').forEach(t => t.classList.remove('active'));
      qsa('.tab-panel').forEach(p => p.classList.remove('active'));
      this.classList.add('active');
      const target = el(`tab-${this.dataset.tab}`);
      if (target) target.classList.add('active');
      if (this.dataset.tab === 'tables') buildTableVarControls();
    });
  });
}

// ── File Inputs ────────────────────────────────────────
function initFileInputs() {
  const fileInput = el('wsFileInput');
  const uploadBtn = el('uploadDataBtn');

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', () => fileInput.click());
  }
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0], { fromMethod: true });
        fileInput.value = '';
      }
    });
  }

  const loadBtn = el('loadExampleBtn');
  if (loadBtn) {
    loadBtn.addEventListener('click', () => doLoadExample());
  }

  const dlBtn = el('downloadExampleBtn');
  if (dlBtn) {
    dlBtn.addEventListener('click', () => downloadExampleData());
  }
}

async function doLoadExample() {
  if (!STATE.activeMethodId) {
    const firstCard = qs('.mini-method-card');
    if (firstCard && firstCard.dataset.method) {
      selectMethod(firstCard.dataset.method);
    } else {
      toast('请先选择分析方法', 'info');
      return;
    }
  }
  const config = getMethodConfig(STATE.activeMethodId);
  if (!config) { toast('请先选择分析方法', 'info'); return; }

  const loadBtn = el('loadExampleBtn');
  const exampleName = config.example_dataset || 'gee_example';
  if (loadBtn) setLoading(loadBtn, true);
  try {
    await loadExampleDataset(exampleName, { silent: true });
    renderDataPanel();
    buildMethodVarControls();
    buildTableVarControls();
    renderAppearanceControls();
    updateMetricGrid();
    renderDataPreview();
    updateDatasetMeta();
    updateFlowLine(2);
    showDownloadExampleBtn(exampleName);
    setStatus('示例数据已加载');
    toast(`已加载「${config.name}」示例数据`, 'success');
  } catch (e) {
    toast('加载失败', 'error');
  } finally {
    if (loadBtn) setLoading(loadBtn, false);
  }
}

function showDownloadExampleBtn(datasetName) {
  const btn = el('downloadExampleBtn');
  if (btn && datasetName) {
    btn.hidden = false;
    btn.dataset.example = datasetName;
  }
}

function downloadExampleData() {
  const btn = el('downloadExampleBtn');
  const name = (btn && btn.dataset.example) || STATE.datasetName;
  if (!name) { toast('请先加载示例数据', 'info'); return; }
  const a = document.createElement('a');
  a.href = `/api/examples/${name}/download`;
  a.download = `${name}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ── Table Type Tabs ────────────────────────────────────
function initTableTypeTabs() {
  qsa('#tableTypeTabs .cat-tab').forEach(tab => {
    tab.addEventListener('click', function() {
      qsa('#tableTypeTabs .cat-tab').forEach(t => t.classList.remove('active'));
      this.classList.add('active');
      STATE.activeTableType = this.dataset.tt;
    });
  });
}

// ── Export Buttons ──────────────────────────────────────
function initExportButtons() {
  document.addEventListener('click', function(e) {
    const exportBtn = e.target.closest('.export-btn');
    if (exportBtn) {
      e.preventDefault();
      const fmt = exportBtn.dataset.fmt;
      if (fmt === 'png') downloadChartImage('png');
      else if (fmt === 'svg') downloadChartImage('svg');
      else if (fmt === 'csv') downloadChartCSV();
      else if (fmt === 'publication') downloadPublicationChart();
    }
    const tableExportBtn = e.target.closest('.table-export-btn');
    if (tableExportBtn) {
      e.preventDefault();
      const texp = tableExportBtn.dataset.texp;
      if (texp === 'excel') exportTableExcel();
      else if (texp === 'csv') exportTableCSV();
      else if (texp === 'html') exportTableHTML();
      else if (texp === 'clipboard') copyTableToClipboard();
    }
  });
}

// ── Appearance Controls ────────────────────────────────
function getMethodElementTypes(methodId) {
  if (!methodId) return [];
  const types = [];
  const markerMethods = ['cluster', 'dim_reduction', 'sensitivity_analysis',
    'ml_knn', 'ml_svm', 'ml_lasso', 'ml_xgboost', 'ml_dt', 'ml_lr',
    'model_comparison', 'meta_analysis'];
  const lineMethods = ['survival_advanced', 'gee', 'markov_model', 'mixed_effects',
    'ml_cnn', 'bayesian', 'ml_knn', 'ml_svm', 'ml_lasso', 'ml_xgboost',
    'ml_dt', 'ml_lr', 'sensitivity_analysis', 'model_comparison', 'meta_analysis',
    'dim_reduction'];
  const barMethods = ['latin_square', 'ml_dt', 'ml_lr', 'ml_lasso', 'ml_xgboost',
    'ml_rf', 'propensity_score', 'counterfactual', 'mediation', 'nhanes_analysis',
    'dim_reduction', 'model_comparison'];

  if (markerMethods.includes(methodId)) types.push('marker');
  if (lineMethods.includes(methodId)) types.push('line');
  if (barMethods.includes(methodId)) types.push('bar');

  if (types.length === 0) types.push('marker', 'line');
  return [...new Set(types)];
}

function getAppearanceColorCount() {
  const config = getMethodConfig(STATE.activeMethodId);
  if (!config) return 4;
  const hasData = (STATE.columns || []).length > 0;
  if (!hasData) return 4;
  const catVars = STATE.variableTypes?.categorical || [];
  const binVars = STATE.variableTypes?.binary || [];
  const groupCols = [...catVars, ...binVars];
  if (groupCols.length > 0 && STATE.previewRows && STATE.previewRows.length > 0) {
    const firstGroupCol = groupCols[0];
    const uniqueVals = new Set(STATE.previewRows.map(r => r[firstGroupCol]).filter(Boolean));
    return Math.max(2, Math.min(uniqueVals.size, 10));
  }
  return 4;
}

function renderAppearanceControls() {
  const container = el('appearanceControls');
  if (!container) return;
  const methodId = STATE.activeMethodId;
  if (!methodId) {
    container.innerHTML = '';
    return;
  }

  const elemTypes = getMethodElementTypes(methodId);
  const theme = getActiveTheme();
  const palette = theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A', '#6F5AA7', '#7C8B52'];
  const numColors = getAppearanceColorCount();

  let html = '';

  // Color pickers
  html += '<div class="appearance-section">';
  html += '<label class="appearance-label">配色方案</label>';
  html += '<div class="color-picker-row" id="colorPickerRow">';
  if (STATE.userColors) {
    STATE.userColors = STATE.userColors.slice(0, numColors);
  }
  for (let i = 0; i < numColors; i++) {
    const currentColor = (STATE.userColors && STATE.userColors[i]) || palette[i % palette.length];
    html += `<input type="color" class="color-swatch" data-idx="${i}" value="${currentColor}" title="颜色 ${i + 1}">`;
  }
  html += '<button class="color-reset-btn" id="resetColorsBtn" title="重置为主题默认色">重置</button>';
  html += '</div></div>';

  // Marker size
  if (elemTypes.includes('marker')) {
    const val = STATE.markerSize || 8;
    html += `<div class="appearance-section">
      <label class="appearance-label">点/标记大小</label>
      <div class="slider-row">
        <input type="range" id="markerSizeInput" min="3" max="20" value="${val}" class="app-slider">
        <span class="slider-val" id="markerSizeVal">${val}</span>
      </div>
    </div>`;

    const shape = STATE.markerShape || 'circle';
    html += `<div class="appearance-section">
      <label class="appearance-label">标记形状</label>
      <div class="shape-select-row">
        <select id="markerShapeInput">
          <option value="circle"${shape === 'circle' ? ' selected' : ''}>● 圆形</option>
          <option value="square"${shape === 'square' ? ' selected' : ''}>■ 方形</option>
          <option value="diamond"${shape === 'diamond' ? ' selected' : ''}>◆ 菱形</option>
          <option value="triangle-up"${shape === 'triangle-up' ? ' selected' : ''}>▲ 三角形</option>
          <option value="triangle-down"${shape === 'triangle-down' ? ' selected' : ''}>▼ 倒三角</option>
          <option value="cross"${shape === 'cross' ? ' selected' : ''}>✚ 十字</option>
          <option value="x"${shape === 'x' ? ' selected' : ''}>✕ X形</option>
          <option value="star"${shape === 'star' ? ' selected' : ''}>★ 星形</option>
          <option value="hexagon"${shape === 'hexagon' ? ' selected' : ''}>⬡ 六边形</option>
          <option value="pentagon"${shape === 'pentagon' ? ' selected' : ''}>⬠ 五边形</option>
        </select>
      </div>
    </div>`;

    const opacity = STATE.markerOpacity != null ? STATE.markerOpacity : 0.88;
    html += `<div class="appearance-section">
      <label class="appearance-label">透明度</label>
      <div class="slider-row">
        <input type="range" id="markerOpacityInput" min="0.1" max="1" step="0.05" value="${opacity}" class="app-slider">
        <span class="slider-val" id="markerOpacityVal">${opacity}</span>
      </div>
    </div>`;
  }

  // Line width
  if (elemTypes.includes('line')) {
    const val = STATE.lineWidth || 2.5;
    html += `<div class="appearance-section">
      <label class="appearance-label">线条宽度</label>
      <div class="slider-row">
        <input type="range" id="lineWidthInput" min="0.5" max="8" step="0.5" value="${val}" class="app-slider">
        <span class="slider-val" id="lineWidthVal">${val}</span>
      </div>
    </div>`;
  }

  // Bar gap
  if (elemTypes.includes('bar')) {
    const val = STATE.barGap != null ? STATE.barGap : 0.15;
    html += `<div class="appearance-section">
      <label class="appearance-label">柱体间距</label>
      <div class="slider-row">
        <input type="range" id="barGapInput" min="0" max="0.6" step="0.05" value="${val}" class="app-slider">
        <span class="slider-val" id="barGapVal">${val}</span>
      </div>
    </div>`;
  }

  // Opacity for non-marker types
  if (!elemTypes.includes('marker') && (elemTypes.includes('bar') || elemTypes.includes('line'))) {
    const opacity = STATE.markerOpacity != null ? STATE.markerOpacity : 0.88;
    html += `<div class="appearance-section">
      <label class="appearance-label">透明度</label>
      <div class="slider-row">
        <input type="range" id="markerOpacityInput" min="0.1" max="1" step="0.05" value="${opacity}" class="app-slider">
        <span class="slider-val" id="markerOpacityVal">${opacity}</span>
      </div>
    </div>`;
  }

  container.innerHTML = html;

  // Bind color pickers
  qsa('.color-swatch', container).forEach(input => {
    input.addEventListener('input', (e) => {
      if (!STATE.userColors) STATE.userColors = [...palette.slice(0, numColors)];
      STATE.userColors[Number(e.target.dataset.idx)] = e.target.value;
    });
    input.addEventListener('change', () => {
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  });

  const resetBtn = el('resetColorsBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      STATE.userColors = null;
      renderAppearanceControls();
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  }

  // Bind sliders
  const markerSlider = el('markerSizeInput');
  if (markerSlider) {
    markerSlider.addEventListener('input', () => {
      STATE.markerSize = Number(markerSlider.value);
      const valEl = el('markerSizeVal');
      if (valEl) valEl.textContent = markerSlider.value;
    });
    markerSlider.addEventListener('change', () => {
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  }

  const shapeSelect = el('markerShapeInput');
  if (shapeSelect) {
    shapeSelect.addEventListener('change', () => {
      STATE.markerShape = shapeSelect.value;
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  }

  const opacitySlider = el('markerOpacityInput');
  if (opacitySlider) {
    opacitySlider.addEventListener('input', () => {
      STATE.markerOpacity = Number(opacitySlider.value);
      const valEl = el('markerOpacityVal');
      if (valEl) valEl.textContent = opacitySlider.value;
    });
    opacitySlider.addEventListener('change', () => {
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  }

  const lineSlider = el('lineWidthInput');
  if (lineSlider) {
    lineSlider.addEventListener('input', () => {
      STATE.lineWidth = Number(lineSlider.value);
      const valEl = el('lineWidthVal');
      if (valEl) valEl.textContent = lineSlider.value;
    });
    lineSlider.addEventListener('change', () => {
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  }

  const barSlider = el('barGapInput');
  if (barSlider) {
    barSlider.addEventListener('input', () => {
      STATE.barGap = Number(barSlider.value);
      const valEl = el('barGapVal');
      if (valEl) valEl.textContent = barSlider.value;
    });
    barSlider.addEventListener('change', () => {
      if (STATE.currentPlotlyData && STATE.currentPlotlyData.length > 0) rerenderExistingCharts();
    });
  }
}

function rerenderExistingCharts() {
  const container = el('chartPreviewContainer');
  if (!container || !window.Plotly) return;
  const plotEls = container.querySelectorAll('.js-plotly-plot');
  if (plotEls.length === 0) return;

  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A', '#6F5AA7'];

  plotEls.forEach(plotEl => {
    try {
      const existingData = plotEl.data;
      if (!existingData) return;
      const traces = polishTracesForPublication(existingData, theme, palette);
      let layout = plotEl.layout || {};
      layout = applyThemeLayout(layout, theme);
      layout = polishLayoutForPublication(layout, '', theme);
      Plotly.react(plotEl, traces, layout);
    } catch (e) {
      console.error('Chart re-render error:', e);
    }
  });

  // Also re-render diagnostic charts
  const diagContainer = el('diagnosticsContainer');
  if (diagContainer) {
    const diagEls = diagContainer.querySelectorAll('.js-plotly-plot');
    diagEls.forEach(plotEl => {
      try {
        const existingData = plotEl.data;
        if (!existingData) return;
        const traces = polishTracesForPublication(existingData, theme, palette);
        let layout = plotEl.layout || {};
        layout = applyThemeLayout(layout, theme);
        layout = polishLayoutForPublication(layout, '', theme);
        Plotly.react(plotEl, traces, layout);
      } catch (e) {
        console.error('Diag chart re-render error:', e);
      }
    });
  }
}

// ── Data Panel ─────────────────────────────────────────
function renderDataPanel() {
  const meta = el('wsDataMeta');
  if (!meta) return;
  const hasData = (STATE.columns || []).length > 0;
  if (hasData) {
    meta.textContent = `${STATE.rowCount || 0} 行 · ${STATE.colCount || 0} 列 · ${STATE.uploadId ? STATE.fileName : (STATE.datasetName || '示例')}`;
  } else {
    const config = getMethodConfig(STATE.activeMethodId);
    meta.textContent = config ? `推荐示例：${config.example_dataset || '—'}` : '请先选择分析方法';
  }
}

// ── Flow Line ──────────────────────────────────────────
function updateFlowLine(activeIndex) {
  qsa('.flow-line span').forEach((span, i) => {
    span.classList.toggle('is-active', i < activeIndex);
  });
}

// ── Status ─────────────────────────────────────────────
function setStatus(msg, isError) {
  const stack = el('statusStack');
  if (!stack) return;
  const div = document.createElement('div');
  div.className = `status-item active${isError ? ' error' : ''}`;
  div.textContent = msg;
  stack.appendChild(div);
  while (stack.children.length > 4) stack.removeChild(stack.firstChild);
}

// ── Metric Grid ────────────────────────────────────────
function updateMetricGrid() {
  const grid = el('metricGrid');
  if (!grid) return;
  const hasData = (STATE.columns || []).length > 0;
  const summary = STATE.summary || {};
  const config = getMethodConfig(STATE.activeMethodId);

  grid.innerHTML = `
    <div class="summary-card"><span>N</span><strong>${hasData ? (STATE.rowCount || '—') : '--'}</strong><small>样本</small></div>
    <div class="summary-card"><span>Vars</span><strong>${hasData ? (STATE.colCount || '—') : '--'}</strong><small>变量</small></div>
    <div class="summary-card"><span>Missing</span><strong>${hasData ? (summary.missing_percent || '—') : '--'}</strong><small>缺失%</small></div>
    <div class="summary-card"><span>Method</span><strong>${config ? config.icon : '--'}</strong><small>${config ? config.name : '—'}</small></div>
  `;
}

// ── Dataset Meta ───────────────────────────────────────
function updateDatasetMeta() {
  const meta = el('datasetMeta');
  if (!meta) return;
  const hasData = (STATE.columns || []).length > 0;
  meta.textContent = hasData
    ? `${STATE.fileName || STATE.datasetName || '已载入'} · ${STATE.rowCount || 0} 行 × ${STATE.colCount || 0} 列`
    : '未载入数据';
}

// ── Example List ───────────────────────────────────────
async function loadExampleList() {
  try {
    const examples = await apiGet('/api/examples');
    const list = el('downloadList');
    if (!list) return;
    const nameMap = {};
    Object.values(METHOD_CATALOG || {}).forEach(m => {
      if (m.example_dataset) nameMap[m.example_dataset] = m.name;
    });
    list.innerHTML = examples.map(ex => {
      const label = nameMap[ex.name] || ex.name.replace(/_example$/, '').replace(/_/g, ' ');
      return `<a class="download-link" href="/api/examples/${ex.name}/download" download="${ex.name}.csv">
        <span>${label}</span><small>${ex.col_count || '-'}列</small>
      </a>`;
    }).join('');
  } catch (e) {}
}

// ── Bootstrap ──────────────────────────────────────────
function bootEmptyState() {
  updateMetricGrid();
  renderDataPreview();
  updateDatasetMeta();
}
