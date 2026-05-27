/* ── Table Generation & Rendering ────────────────────── */

function renderResultTables(tables) {
  if (!tables || tables.length === 0) return;
  const container = el('resultTablesContainer');
  if (!container) return;

  let html = '';
  tables.forEach((table, idx) => {
    const rows = table.rows || [];
    const headers = table.headers || (rows.length > 0 ? Object.keys(rows[0]) : []);
    if (rows.length === 0) return;
    html += `<div class="result-card">
      <h3>${table.title || '表格 ' + (idx + 1)}</h3>
      <div class="result-table-wrap">
        <table class="three-line">
          <thead><tr>${headers.map(h => `<th>${escapeHtml(String(h))}</th>`).join('')}</tr></thead>
          <tbody>`;
    rows.slice(0, 50).forEach(row => {
      html += '<tr>';
      headers.forEach(h => {
        const val = row[h];
        html += `<td>${val !== undefined && val !== null ? escapeHtml(String(val)) : ''}</td>`;
      });
      html += '</tr>';
    });
    if (rows.length > 50) html += `<caption>显示前50行 / 共${rows.length}行</caption>`;
    html += '</tbody></table></div></div>';
  });
  container.innerHTML = html || '<div class="empty-state small">表格结果将在分析后显示</div>';
  STATE.lastAnalysisTables = tables;
}

function renderDataPreview() {
  const target = el('previewTable');
  if (!target) return;
  const rows = STATE.previewRows || [];
  const cols = (STATE.columns || []).slice(0, 8);
  if (!rows.length || !cols.length) {
    target.innerHTML = '<div class="empty-state small">等待数据载入</div>';
    return;
  }
  const maxRows = Math.min(rows.length, 5);
  let html = '<table class="three-line"><thead><tr>';
  cols.forEach(c => { html += `<th>${escapeHtml(String(c))}</th>`; });
  html += '</tr></thead><tbody>';
  for (let i = 0; i < maxRows; i++) {
    html += '<tr>';
    cols.forEach(c => {
      const v = rows[i][c];
      html += `<td>${v !== undefined && v !== null ? escapeHtml(String(v)) : ''}</td>`;
    });
    html += '</tr>';
  }
  if (rows.length > maxRows) html += `<caption>前${maxRows}行 / 共${rows.length}行</caption>`;
  html += '</tbody></table>';
  target.innerHTML = html;
}

function renderDiscussion(discussionText) {
  const container = el('discussionContainer');
  if (!container) return;
  if (!discussionText || discussionText.length < 10) {
    container.innerHTML = '<div class="empty-state small">暂无结果讨论</div>';
    return;
  }
  const htmlText = simpleMarkdownToHtml(discussionText);
  container.innerHTML = `<div class="discussion-content">${htmlText}</div>`;
}

function simpleMarkdownToHtml(md) {
  let html = md;
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  const lines = html.split('\n');
  let result = [];
  let inList = false;
  let listType = '';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const ulMatch = line.match(/^[-•]\s+(.+)/);
    const olMatch = line.match(/^\d+\.\s+(.+)/);

    if (ulMatch) {
      if (!inList || listType !== 'ul') {
        if (inList) result.push(`</${listType}>`);
        result.push('<ul>');
        inList = true;
        listType = 'ul';
      }
      result.push(`<li>${ulMatch[1]}</li>`);
    } else if (olMatch) {
      if (!inList || listType !== 'ol') {
        if (inList) result.push(`</${listType}>`);
        result.push('<ol>');
        inList = true;
        listType = 'ol';
      }
      result.push(`<li>${olMatch[1]}</li>`);
    } else {
      if (inList) {
        result.push(`</${listType}>`);
        inList = false;
        listType = '';
      }
      if (line.startsWith('<h2>') || line.startsWith('<h3>')) {
        result.push(line);
      } else if (line.trim() === '') {
        result.push('');
      } else {
        result.push(`<p>${line}</p>`);
      }
    }
  }
  if (inList) result.push(`</${listType}>`);
  return result.join('\n');
}

async function generateTable() {
  const btn = el('generateTableBtn');
  if (btn) setLoading(btn, true);
  try {
    const varSel = el('tableVars');
    const selectedVars = varSel && varSel.selectedOptions
      ? Array.from(varSel.selectedOptions).map(o => o.value)
      : [];
    const payload = {
      use_demo: !STATE.uploadId,
      upload_id: STATE.uploadId || undefined,
      dataset_name: STATE.datasetName || undefined,
      table_type: STATE.activeTableType || 'baseline',
      group_var: el('tableGroupVar')?.value || undefined,
      variables: selectedVars.length > 0 ? selectedVars : undefined,
      decimal_places: parseInt(el('tableDecimals')?.value || '2'),
    };
    const result = await apiPost('/api/table/baseline', payload);
    const container = el('tableResultContainer');
    if (container && result.rows) {
      const headers = result.columns || result.headers || Object.keys(result.rows[0]);
      let html = '<table class="three-line"><thead><tr>';
      headers.forEach(h => { html += `<th>${escapeHtml(String(h))}</th>`; });
      html += '</tr></thead><tbody>';
      result.rows.forEach(row => {
        html += '<tr>';
        headers.forEach(h => {
          html += `<td>${row[h] !== undefined ? escapeHtml(String(row[h])) : ''}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      container.innerHTML = html;
    }
    const toolbar = el('tableToolbar');
    if (toolbar) toolbar.style.display = 'flex';
    STATE.lastTableData = result.rows || [];
    toast('三线表已生成', 'success');
  } catch (e) {
    toast('表格生成失败: ' + e.message, 'error');
  } finally {
    if (btn) setLoading(btn, false);
  }
}
