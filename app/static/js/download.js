/* ── Download & Export ───────────────────────────────── */

async function downloadChartImage(format) {
  const container = el('chartPreviewContainer');
  const plotEl = container ? container.querySelector('.js-plotly-plot') : null;
  if (!plotEl || !window.Plotly) {
    toast('请先生成图表', 'info');
    return;
  }
  try {
    if (format === 'png') {
      const imgData = await Plotly.toImage(plotEl, { format: 'png', width: 1200, height: 700 });
      const link = document.createElement('a');
      link.href = imgData;
      link.download = `chart_${STATE.activeMethodId || 'export'}.png`;
      link.click();
    } else if (format === 'svg') {
      const imgData = await Plotly.toImage(plotEl, { format: 'svg', width: 1200, height: 700 });
      const link = document.createElement('a');
      link.href = imgData;
      link.download = `chart_${STATE.activeMethodId || 'export'}.svg`;
      link.click();
    }
    toast(`图表导出为 ${format.toUpperCase()}`, 'success');
  } catch (e) {
    toast('导出失败: ' + e.message, 'error');
  }
}

async function downloadChartCSV() {
  const plotlyData = STATE.currentPlotlyData;
  if (!plotlyData || plotlyData.length === 0) {
    toast('请先生成图表', 'info');
    return;
  }
  try {
    let csvContent = '';
    plotlyData.forEach((chart, ci) => {
      const fig = typeof chart.plotly === 'string' ? JSON.parse(chart.plotly) : chart.plotly;
      if (!fig || !fig.data) return;
      csvContent += `# ${chart.title || 'Chart ' + (ci + 1)}\n`;
      const traces = Array.isArray(fig.data) ? fig.data : [fig.data];
      traces.forEach((trace, ti) => {
        const name = trace.name || `trace_${ti}`;
        csvContent += `## ${name}\n`;
        const xVals = trace.x || [];
        const yVals = trace.y || [];
        csvContent += 'x,y\n';
        const len = Math.max(Array.isArray(xVals) ? xVals.length : 0, Array.isArray(yVals) ? yVals.length : 0);
        for (let i = 0; i < Math.min(len, 1000); i++) {
          csvContent += `${Array.isArray(xVals) ? (xVals[i] ?? '') : xVals},${Array.isArray(yVals) ? (yVals[i] ?? '') : yVals}\n`;
        }
        csvContent += '\n';
      });
    });
    const blob = new Blob(['﻿' + csvContent], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `data_${STATE.activeMethodId || 'export'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast('图表数据导出为 CSV', 'success');
  } catch (e) {
    toast('CSV导出失败: ' + e.message, 'error');
  }
}

async function downloadPublicationChart() {
  const container = el('chartPreviewContainer');
  const plotEls = container ? container.querySelectorAll('.js-plotly-plot') : null;
  if (!plotEls || plotEls.length === 0) {
    toast('请先生成图表', 'info');
    return;
  }
  try {
    for (let i = 0; i < plotEls.length; i++) {
      const imgData = await Plotly.toImage(plotEls[i], {
        format: 'png',
        width: 2400,
        height: 1600,
        scale: 3,
      });
      const link = document.createElement('a');
      link.href = imgData;
      link.download = `publication_${STATE.activeMethodId || 'chart'}_${i + 1}.png`;
      link.click();
      await new Promise(r => setTimeout(r, 300));
    }
    toast(`已导出 ${plotEls.length} 张出版级图表 (300 DPI)`, 'success');
  } catch (e) {
    toast('出版级导出失败: ' + e.message, 'error');
  }
}

async function exportTableExcel() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  try {
    const res = await fetch('/api/export/table-excel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ table_data: STATE.lastTableData }),
    });
    if (!res.ok) throw new Error('Export failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'table_export.xlsx';
    a.click();
    URL.revokeObjectURL(url);
    toast('表格已导出为 Excel', 'success');
  } catch (e) {
    toast('导出失败', 'error');
  }
}

async function exportTableCSV() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  try {
    const res = await fetch('/api/export/table-csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ table_data: STATE.lastTableData }),
    });
    if (!res.ok) throw new Error('Export failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'table_export.csv';
    a.click();
    URL.revokeObjectURL(url);
    toast('表格已导出为 CSV', 'success');
  } catch (e) {
    toast('导出失败', 'error');
  }
}

async function exportTableHTML() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  try {
    const res = await fetch('/api/export/table-html', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ table_data: STATE.lastTableData }),
    });
    if (!res.ok) throw new Error('Export failed');
    const html = await res.text();
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'table_export.html';
    a.click();
    URL.revokeObjectURL(url);
    toast('表格已导出为 HTML', 'success');
  } catch (e) {
    toast('导出失败', 'error');
  }
}

async function copyTableToClipboard() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  const headers = Object.keys(STATE.lastTableData[0]);
  let csv = headers.join('\t') + '\n';
  STATE.lastTableData.forEach(row => {
    csv += headers.map(h => String(row[h] ?? '')).join('\t') + '\n';
  });
  try {
    await navigator.clipboard.writeText(csv);
    toast('表格已复制到剪贴板', 'success');
  } catch (e) {
    toast('复制失败', 'error');
  }
}
