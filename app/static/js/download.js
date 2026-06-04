/* ── Download Module — Enhanced Export (TIFF/PDF/PNG/SVG/CSV) ── */

function initDownloads() {
  // Export buttons are handled via event delegation in app.js
}

async function downloadChartImage(format) {
  const normalizedFormat = String(format || '').toLowerCase();
  if (!['png', 'svg', 'tiff', 'pdf'].includes(normalizedFormat)) {
    toast(`暂不支持 ${String(format).toUpperCase()} 格式`, 'warning');
    return;
  }

  const plotEl = getCurrentPlotElement();
  if (!plotEl) {
    toast('请先生成图表', 'warning');
    return;
  }

  if (!window.Plotly) {
    toast('Plotly 未加载，无法导出图表。请刷新页面后重试。', 'error');
    return;
  }

  const filename = `${safeFilename(STATE.activeMethodId || 'chart')}_${new Date().toISOString().replace(/[:.]/g, '-')}`;

  try {
    if (normalizedFormat === 'svg') {
      const dataUrl = await exportPlotlyDataUrl(plotEl, 'svg', 1);
      downloadDataUrl(dataUrl, `${filename}.svg`);
      toast('SVG 已按当前预览下载', 'success');
      return;
    }

    if (normalizedFormat === 'png') {
      const dataUrl = await exportPlotlyDataUrl(plotEl, 'png', 2.5);
      downloadDataUrl(dataUrl, `${filename}.png`);
      toast('PNG 已按当前预览下载', 'success');
      return;
    }

    const raster = await rasterizeCurrentPlot(plotEl, 2.5);
    if (normalizedFormat === 'tiff') {
      const tiffBytes = encodeCanvasAsTiff(raster.canvas);
      downloadBlob(new Blob([tiffBytes], { type: 'image/tiff' }), `${filename}.tiff`);
      toast('TIFF 已按当前预览下载', 'success');
      return;
    }

    if (normalizedFormat === 'pdf') {
      const pdfBytes = encodeCanvasAsPdf(raster.canvas, raster.cssWidth, raster.cssHeight);
      downloadBlob(new Blob([pdfBytes], { type: 'application/pdf' }), `${filename}.pdf`);
      toast('PDF 已按当前预览下载', 'success');
    }
  } catch (err) {
    console.error('Chart export failed:', err);
    try {
      const fallbackSize = getDisplayedPlotSize(plotEl);
      await Plotly.downloadImage(plotEl, {
        format: normalizedFormat === 'svg' ? 'svg' : 'png',
        width: fallbackSize.width,
        height: fallbackSize.height,
        scale: 2.5,
        filename,
      });
      toast(`${normalizedFormat.toUpperCase()} 已按当前预览尺寸下载`, 'success');
    } catch (fallbackErr) {
      console.error('Plotly downloadImage fallback failed:', fallbackErr);
      toast(`导出 ${normalizedFormat.toUpperCase()} 失败: ${fallbackErr.message || err.message}`, 'error');
    }
  }
}

async function exportPlotlyDataUrl(plotEl, format, scale) {
  const exportSize = await syncPlotForWysiwygExport(plotEl);
  return Plotly.toImage(plotEl, {
    format,
    width: exportSize.width,
    height: exportSize.height,
    scale,
  });
}

function getCurrentPlotElement() {
  const previewEl = el('chartPreviewContainer');
  let plotEl = previewEl ? previewEl.querySelector('.chart-plot.js-plotly-plot') : null;
  if (!plotEl) plotEl = previewEl ? previewEl.querySelector('.js-plotly-plot') : null;
  if (!plotEl) plotEl = previewEl ? previewEl.querySelector('[class*="plotly"]') : null;
  if (!plotEl) plotEl = previewEl ? previewEl.querySelector('div[data-plotly]') : null;
  return plotEl || null;
}

function getDisplayedPlotSize(plotEl) {
  const full = plotEl?._fullLayout || {};
  const width = Math.floor(full.width || plotEl?.clientWidth || STATE.currentPlotlyLayout?.width || 1200);
  const height = Math.floor(full.height || plotEl?.clientHeight || STATE.currentPlotlyLayout?.height || 720);
  return {
    width: Math.max(320, width),
    height: Math.max(320, height),
  };
}

async function syncPlotForWysiwygExport(plotEl) {
  if (!plotEl || !window.Plotly) return getDisplayedPlotSize(plotEl);

  if (typeof Plotly.Plots?.resize === 'function') {
    const resizePromise = Plotly.Plots.resize(plotEl);
    if (resizePromise && typeof resizePromise.then === 'function') {
      await resizePromise;
    }
  }

  const size = getDisplayedPlotSize(plotEl);
  const relayoutPromise = Plotly.relayout(plotEl, {
    width: size.width,
    height: size.height,
    autosize: false,
  });
  if (relayoutPromise && typeof relayoutPromise.then === 'function') {
    await relayoutPromise;
  }
  if (STATE.currentPlotlyLayout) {
    STATE.currentPlotlyLayout = {
      ...STATE.currentPlotlyLayout,
      width: size.width,
      height: size.height,
      autosize: false,
    };
  }
  return size;
}

async function rasterizeCurrentPlot(plotEl, scale) {
  scale = scale || 2.5;
  const size = await syncPlotForWysiwygExport(plotEl);
  const dataUrl = await Plotly.toImage(plotEl, {
    format: 'png',
    width: size.width,
    height: size.height,
    scale,
  });
  const img = await loadImageFromDataUrl(dataUrl);
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  return { canvas, cssWidth: size.width, cssHeight: size.height };
}

function loadImageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('无法读取当前预览图像'));
    img.src = dataUrl;
  });
}

/* ── TIFF Encoder ─────────────────────────────────────── */
function encodeCanvasAsTiff(canvas) {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const pixelBytes = width * height * 3;
  const entryCount = 12;
  const ifdOffset = 8;
  const ifdSize = 2 + entryCount * 12 + 4;
  const bitsOffset = ifdOffset + ifdSize;
  const xResOffset = bitsOffset + 6;
  const yResOffset = xResOffset + 8;
  const pixelOffset = yResOffset + 8;
  const totalBytes = pixelOffset + pixelBytes;
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  bytes[0] = 0x49; bytes[1] = 0x49;
  view.setUint16(2, 42, true);
  view.setUint32(4, ifdOffset, true);
  view.setUint16(ifdOffset, entryCount, true);

  let entry = ifdOffset + 2;
  const writeEntry = (tag, type, count, value) => {
    view.setUint16(entry, tag, true);
    view.setUint16(entry + 2, type, true);
    view.setUint32(entry + 4, count, true);
    if (type === 3 && count === 1) {
      view.setUint16(entry + 8, value, true);
      view.setUint16(entry + 10, 0, true);
    } else {
      view.setUint32(entry + 8, value, true);
    }
    entry += 12;
  };

  writeEntry(256, 4, 1, width);
  writeEntry(257, 4, 1, height);
  writeEntry(258, 3, 3, bitsOffset);
  writeEntry(259, 3, 1, 1);
  writeEntry(262, 3, 1, 2);
  writeEntry(273, 4, 1, pixelOffset);
  writeEntry(277, 3, 1, 3);
  writeEntry(278, 4, 1, height);
  writeEntry(279, 4, 1, pixelBytes);
  writeEntry(282, 5, 1, xResOffset);
  writeEntry(283, 5, 1, yResOffset);
  writeEntry(296, 3, 1, 2);
  view.setUint32(ifdOffset + 2 + entryCount * 12, 0, true);

  view.setUint16(bitsOffset, 8, true);
  view.setUint16(bitsOffset + 2, 8, true);
  view.setUint16(bitsOffset + 4, 8, true);
  view.setUint32(xResOffset, 300, true);
  view.setUint32(xResOffset + 4, 1, true);
  view.setUint32(yResOffset, 300, true);
  view.setUint32(yResOffset + 4, 1, true);

  let p = pixelOffset;
  for (let i = 0; i < rgba.length; i += 4) {
    const a = rgba[i + 3] / 255;
    bytes[p++] = Math.round(rgba[i] * a + 255 * (1 - a));
    bytes[p++] = Math.round(rgba[i + 1] * a + 255 * (1 - a));
    bytes[p++] = Math.round(rgba[i + 2] * a + 255 * (1 - a));
  }
  return bytes;
}

/* ── PDF Encoder ──────────────────────────────────────── */
function encodeCanvasAsPdf(canvas, cssWidth, cssHeight) {
  const jpegDataUrl = canvas.toDataURL('image/jpeg', 0.96);
  const jpegBytes = dataUrlToBytes(jpegDataUrl);
  const pageWidth = Math.max(320, Math.round(cssWidth || canvas.width));
  const pageHeight = Math.max(320, Math.round(cssHeight || canvas.height));
  const imageWidth = canvas.width;
  const imageHeight = canvas.height;
  const encoder = new TextEncoder();
  const chunks = [];
  const offsets = [0];
  let length = 0;

  const push = (chunk) => {
    const bytes = typeof chunk === 'string' ? encoder.encode(chunk) : chunk;
    chunks.push(bytes);
    length += bytes.length;
  };
  const startObject = (id) => {
    offsets[id] = length;
    push(`${id} 0 obj\n`);
  };

  push('%PDF-1.4\n');
  startObject(1);
  push('<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');
  startObject(2);
  push('<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n');
  startObject(3);
  push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n`);
  startObject(4);
  push(`<< /Type /XObject /Subtype /Image /Width ${imageWidth} /Height ${imageHeight} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>\nstream\n`);
  push(jpegBytes);
  push('\nendstream\nendobj\n');
  const content = `q\n${pageWidth} 0 0 ${pageHeight} 0 0 cm\n/Im0 Do\nQ\n`;
  startObject(5);
  push(`<< /Length ${encoder.encode(content).length} >>\nstream\n${content}endstream\nendobj\n`);

  const xrefOffset = length;
  push('xref\n0 6\n0000000000 65535 f \n');
  for (let i = 1; i <= 5; i++) {
    push(`${String(offsets[i]).padStart(10, '0')} 00000 n \n`);
  }
  push(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`);

  const out = new Uint8Array(length);
  let offset = 0;
  chunks.forEach(chunk => {
    out.set(chunk, offset);
    offset += chunk.length;
  });
  return out;
}

function dataUrlToBytes(dataUrl) {
  const base64 = String(dataUrl).split(',')[1] || '';
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/* ── Publication Export ───────────────────────────────── */
async function downloadPublicationChart() {
  const container = el('chartPreviewContainer');
  const plotEls = container ? container.querySelectorAll('.js-plotly-plot') : null;
  if (!plotEls || plotEls.length === 0) {
    toast('请先生成图表', 'info');
    return;
  }
  try {
    for (let i = 0; i < plotEls.length; i++) {
      const raster = await rasterizeCurrentPlot(plotEls[i], 3);
      const tiffBytes = encodeCanvasAsTiff(raster.canvas);
      downloadBlob(new Blob([tiffBytes], { type: 'image/tiff' }), `publication_${STATE.activeMethodId || 'chart'}_${i + 1}.tiff`);
      await new Promise(r => setTimeout(r, 300));
    }
    toast(`已导出 ${plotEls.length} 张出版级图表 (300 DPI TIFF)`, 'success');
  } catch (e) {
    toast('出版级导出失败: ' + e.message, 'error');
  }
}

/* ── CSV Export ───────────────────────────────────────── */
function downloadChartCSV() {
  const renderedPlots = getRenderedPlotEntries();
  if ((!renderedPlots || renderedPlots.length === 0) && !STATE.currentChartSourceData) {
    toast('请先生成图表', 'warning');
    return;
  }

  // Try source data first
  if (STATE.currentChartSourceData && STATE.currentChartParams) {
    const params = STATE.currentChartParams || {};
    const selectedCols = uniqueCsvColumns([
      params.x_var, params.y_var, params.color_var, params.size_var, params.group_var,
      params.time_var, params.event_var, params.outcome_var, params.predictor_var,
      params.target, params.treatment_var, params.outcome_var, params.baseline_var,
      ...(params.value_vars || []),
    ]).filter(c => Array.isArray(STATE.currentChartSourceData[c]));

    if (selectedCols.length > 0) {
      const csv = columnDataToCSV(STATE.currentChartSourceData, selectedCols);
      downloadBlob(new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' }), `${safeFilename(STATE.activeMethodId || 'chart')}_source_data.csv`);
      toast('CSV 已下载', 'success');
      return;
    }
  }

  // Fall back to Plotly trace data
  if (!renderedPlots || renderedPlots.length === 0) {
    toast('请先生成图表', 'info');
    return;
  }

  const csvContent = renderedPlotsToCSV(renderedPlots);
  const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8' });
  downloadBlob(blob, `data_${safeFilename(STATE.activeMethodId || 'export')}.csv`);
  toast(`已导出 ${renderedPlots.length} 张图表的数据`, 'success');
}

function getRenderedPlotEntries() {
  const entries = (STATE.currentRenderedPlots || [])
    .filter(entry => entry && entry.el && entry.el.isConnected)
    .map(entry => ({
      title: entry.title || extractPlotTitleFromElement(entry.el) || `Chart ${entry.index + 1}`,
      data: Array.isArray(entry.data) && entry.data.length ? entry.data : (Array.isArray(entry.el.data) ? entry.el.data : []),
      layout: entry.layout || entry.el.layout || {},
    }));

  if (entries.length > 0) return entries;

  const container = el('chartPreviewContainer');
  const plotEls = container ? Array.from(container.querySelectorAll('.js-plotly-plot')) : [];
  return plotEls.map((plotEl, index) => ({
    title: extractPlotTitleFromElement(plotEl) || `Chart ${index + 1}`,
    data: Array.isArray(plotEl.data) ? plotEl.data : [],
    layout: plotEl.layout || {},
  })).filter(entry => entry.data.length > 0);
}

function renderedPlotsToCSV(plots) {
  const rows = [['chart', 'trace', 'trace_type', 'point', 'x', 'y', 'z', 'text', 'value']];
  (plots || []).forEach((plot, chartIndex) => {
    const traces = Array.isArray(plot.data) ? plot.data : [];
    traces.forEach((trace, traceIndex) => {
      const traceName = trace.name || `trace_${traceIndex + 1}`;
      const traceType = trace.type || 'scatter';
      const xVals = trace.x || trace._input?.x;
      const yVals = trace.y || trace._input?.y;
      const zVals = trace.z || trace._input?.z;
      const textVals = trace.text || trace._input?.text;
      const valueVals = trace.values || trace._input?.values;
      const labelVals = trace.labels || trace._input?.labels;

      const zMatrix = getMatrixData(zVals) || getMatrixData(textVals);
      if (zMatrix) {
        zMatrix.forEach((zRow, rowIndex) => {
          const zValues = Array.isArray(zRow) ? zRow : [zRow];
          zValues.forEach((zValue, colIndex) => {
            rows.push([
              plot.title || `Chart ${chartIndex + 1}`,
              traceName,
              traceType,
              `${rowIndex + 1}:${colIndex + 1}`,
              getIndexedValue(xVals, colIndex),
              getIndexedValue(yVals, rowIndex),
              zValue,
              getMatrixValue(textVals, rowIndex, colIndex),
              getIndexedValue(valueVals, colIndex),
            ]);
          });
        });
        return;
      }

      const len = Math.max(
        getArrayLength(xVals),
        getArrayLength(yVals),
        getArrayLength(valueVals),
        getArrayLength(labelVals),
        getArrayLength(textVals),
        1,
      );
      for (let i = 0; i < len; i++) {
        rows.push([
          plot.title || `Chart ${chartIndex + 1}`,
          traceName,
          traceType,
          i + 1,
          getIndexedValue(xVals, i),
          getIndexedValue(yVals, i),
          '',
          getIndexedValue(textVals, i) || getIndexedValue(labelVals, i),
          getIndexedValue(valueVals, i),
        ]);
      }
    });
  });
  return rows.map(row => row.map(csvEscape).join(',')).join('\n') + '\n';
}

function getMatrixData(value) {
  if (Array.isArray(value) && value.some(row => Array.isArray(row))) return value;
  if (value && Array.isArray(value._inputArray) && value._inputArray.some(row => Array.isArray(row))) {
    return value._inputArray;
  }
  return null;
}

function getArrayLength(value) {
  return Array.isArray(value) ? value.length : (value == null ? 0 : 1);
}

function getIndexedValue(value, index) {
  if (Array.isArray(value)) return value[index] ?? '';
  return index === 0 && value != null ? value : '';
}

function getMatrixValue(value, rowIndex, colIndex) {
  if (Array.isArray(value) && Array.isArray(value[rowIndex])) return value[rowIndex][colIndex] ?? '';
  if (Array.isArray(value)) return value[colIndex] ?? '';
  return rowIndex === 0 && colIndex === 0 && value != null ? value : '';
}

function extractPlotTitleFromElement(plotEl) {
  const layoutTitle = plotEl?._fullLayout?.title || plotEl?.layout?.title;
  if (!layoutTitle) return '';
  if (typeof layoutTitle === 'string') return layoutTitle.replace(/<[^>]*>/g, '').trim();
  if (typeof layoutTitle.text === 'string') return layoutTitle.text.replace(/<[^>]*>/g, '').trim();
  return '';
}

/* ── Table Export ─────────────────────────────────────── */
function exportTableExcel() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  const headers = Object.keys(STATE.lastTableData[0]);
  let csv = headers.join(',') + '\n';
  STATE.lastTableData.forEach(row => {
    const vals = headers.map(c => {
      const v = row[c] !== undefined ? String(row[c]).replace(/,/g, ';') : '';
      return v.includes(' ') ? `"${v}"` : v;
    });
    csv += vals.join(',') + '\n';
  });
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
  downloadBlob(blob, 'table_export.csv');
  toast('表格已导出为 CSV (可用 Excel 打开)', 'success');
}

function exportTableCSV() {
  exportTableExcel();
}

function exportTableHTML() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  const headers = Object.keys(STATE.lastTableData[0]);
  let html = '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>三线表</title>' +
    '<style>body{font-family:\'Noto Sans SC\',sans-serif;padding:24px;}' +
    '.three-line{border-collapse:collapse;width:100%;}' +
    '.three-line thead{border-top:2px solid #000;border-bottom:1px solid #000;}' +
    '.three-line th{padding:8px 12px;text-align:left;font-weight:800;background:#f9fafb;}' +
    '.three-line td{padding:8px 12px;}' +
    '.three-line tbody tr:last-child{border-bottom:2px solid #000;}' +
    '</style></head><body><table class="three-line"><thead><tr>';
  headers.forEach(c => { html += `<th>${escapeHtml(String(c))}</th>`; });
  html += '</tr></thead><tbody>';
  STATE.lastTableData.forEach(row => {
    html += '<tr>';
    headers.forEach(c => { html += `<td>${row[c] !== undefined ? escapeHtml(String(row[c])) : ''}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table></body></html>';
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  downloadBlob(blob, 'table_export.html');
  toast('HTML 表格已下载', 'success');
}

function copyTableToClipboard() {
  if (!STATE.lastTableData || STATE.lastTableData.length === 0) {
    toast('请先生成表格', 'info');
    return;
  }
  const headers = Object.keys(STATE.lastTableData[0]);
  let text = headers.join('\t') + '\n';
  STATE.lastTableData.forEach(row => {
    text += headers.map(c => row[c] !== undefined ? row[c] : '').join('\t') + '\n';
  });
  navigator.clipboard.writeText(text).then(() => {
    toast('表格已复制到剪贴板，可直接粘贴到 Word/Excel', 'success');
  }).catch(() => toast('复制失败，请手动复制', 'error'));
}

/* ── Utility ──────────────────────────────────────────── */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 250);
}

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => a.remove(), 250);
}

function safeFilename(value) {
  return String(value || 'download').replace(/[\\/:*?"<>|]+/g, '_').slice(0, 80);
}

function csvEscape(value) {
  const s = String(value ?? '');
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function uniqueCsvColumns(cols) {
  return [...new Set((cols || []).flat().filter(Boolean))];
}

function columnDataToCSV(data, columns) {
  const n = Math.max(...columns.map(c => (data[c] || []).length), 0);
  let csv = columns.map(csvEscape).join(',') + '\n';
  for (let i = 0; i < n; i++) {
    csv += columns.map(c => csvEscape((data[c] || [])[i] ?? '')).join(',') + '\n';
  }
  return csv;
}

/* ── Download Menu Toggle ─────────────────────────────── */
function toggleDownloadMenu() {
  const menu = el('chartDownloadMenu');
  const btn = el('chartDownloadBtn');
  if (!menu || !btn) return;
  const isOpen = !menu.hidden;
  if (isOpen) {
    menu.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  } else {
    menu.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
  }
}

function closeDownloadMenu() {
  const menu = el('chartDownloadMenu');
  const btn = el('chartDownloadBtn');
  if (menu) menu.hidden = true;
  if (btn) btn.setAttribute('aria-expanded', 'false');
}
