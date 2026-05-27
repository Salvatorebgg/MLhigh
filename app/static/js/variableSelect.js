/* ── Variable Selection Controls ─────────────────────── */

function buildMethodVarControls() {
  const container = el('varControls');
  if (!container) return;
  const config = getMethodConfig(STATE.activeMethodId);
  if (!config || !config.params || config.params.length === 0) {
    container.innerHTML = '<div class="empty-state small">该方法无需额外参数配置</div>';
    return;
  }
  const hasData = (STATE.columns || []).length > 0;
  let html = '';
  config.params.forEach(p => {
    if (p.type === 'select') {
      html += `<div class="field-row">
        <label>${p.label}</label>
        <select id="param_${p.key}" class="chart-var-select">
          <option value="">-- 选择变量 --</option>`;
      if (hasData) {
        STATE.columns.forEach(col => {
          const sel = col === p.default ? ' selected' : '';
          html += `<option value="${col}"${sel}>${col}</option>`;
        });
      } else if (p.default) {
        html += `<option value="${p.default}" selected>${p.default}</option>`;
      }
      if (p.options) {
        p.options.forEach(opt => {
          const sel = opt === p.default ? ' selected' : '';
          html += `<option value="${opt}"${sel}>${opt}</option>`;
        });
      }
      html += `</select></div>`;
    } else if (p.type === 'number') {
      html += `<div class="field-row">
        <label>${p.label}</label>
        <input type="number" id="param_${p.key}" value="${p.default || ''}" />
      </div>`;
    }
  });
  container.innerHTML = html;
}

function collectMethodParams() {
  const config = getMethodConfig(STATE.activeMethodId);
  if (!config || !config.params) return {};
  const params = {};
  config.params.forEach(p => {
    const input = el(`param_${p.key}`);
    if (input) {
      params[p.key] = p.type === 'number' ? Number(input.value) : input.value;
    }
  });
  return params;
}

function buildTableVarControls() {
  const groupSel = el('tableGroupVar');
  const varSel = el('tableVars');
  if (!groupSel || !varSel) return;
  const hasData = (STATE.columns || []).length > 0;
  if (!hasData) return;
  let opts = '';
  STATE.columns.forEach(col => {
    const isCat = STATE.variableTypes?.categorical?.includes(col) || STATE.variableTypes?.binary?.includes(col);
    opts += `<option value="${col}">${col}${isCat ? ' [分组]' : ''}</option>`;
  });
  groupSel.innerHTML = opts;
  varSel.innerHTML = opts;
}
