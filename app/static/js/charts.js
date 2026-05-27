/* ── MLhigh Charts Module — Publication-quality rendering ─── */

function renderAllCharts(charts) {
  if (!charts || charts.length === 0) return;
  const container = el('chartPreviewContainer');
  if (!container) return;

  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A', '#6F5AA7'];

  let html = '';
  charts.forEach((chart, i) => {
    const chartId = `result-chart-${i}`;
    html += `<div class="result-card">
      <h3 class="result-card-title">${chart.title || '图表 ' + (i + 1)}</h3>
      <div class="result-chart-container" id="${chartId}"></div>
    </div>`;
  });
  container.innerHTML = html;

  charts.forEach((chart, i) => {
    const chartId = `result-chart-${i}`;
    if (chart.plotly) {
      setTimeout(() => {
        renderPublicationChart(chartId, chart.plotly, palette, theme);
      }, i * 80);
    }
  });
}

function renderAllDiagCharts(diagnostics) {
  if (!diagnostics || diagnostics.length === 0) return;
  const container = el('diagnosticsContainer');
  if (!container) return;

  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A'];

  let html = '';
  diagnostics.forEach((d, i) => {
    const chartId = `diag-chart-${i}`;
    html += `<div class="result-card">
      <h3 class="result-card-title">${d.title || '诊断 ' + (i + 1)}</h3>
      <div class="result-chart-container" id="${chartId}"></div>
    </div>`;
  });
  container.innerHTML = html;

  diagnostics.forEach((d, i) => {
    if (d.plotly) {
      setTimeout(() => {
        renderPublicationChart(`diag-chart-${i}`, d.plotly, palette, theme);
      }, i * 80);
    }
  });
}

function renderPublicationChart(containerId, plotlyJson, palette, theme) {
  const container = el(containerId);
  if (!container || !window.Plotly) return;

  try {
    const fig = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
    if (!fig || !fig.data) return;

    let traces = Array.isArray(fig.data) ? fig.data : [fig.data];
    let layout = fig.layout || {};

    // Compute sensible height from container
    const containerHeight = Math.max(480, container.clientHeight || 480);

    // Apply theme and polish
    traces = polishTracesForPublication(traces, theme, palette);
    layout = applyThemeLayout(layout, theme);

    // Force proper sizing
    layout.autosize = true;
    layout.height = layout.height || containerHeight;
    layout.margin = layout.margin || { l: 60, r: 30, t: 50, b: 60, pad: 8 };

    layout = polishLayoutForPublication(layout, layout._chartType || '', theme);

    Plotly.newPlot(container, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
      modeBarButtonsToAdd: ['toImage'],
      toImageButtonOptions: {
        format: 'png', height: 1440, width: 2160, scale: 2,
        filename: 'mlhigh_chart_' + Date.now(),
      },
    });
  } catch (e) {
    container.innerHTML = '<div class="empty-state small">图表渲染失败</div>';
    console.error('Chart render error:', e);
  }
}

// ── Polish traces for publication quality ───────────────
function polishTracesForPublication(traces, theme, palette) {
  const ink = theme.ink || '#111827';
  const markerLine = theme.markerLine || '#ffffff';
  const userMarkerSize = STATE.markerSize || 8;
  const userLineWidth = STATE.lineWidth || 2.5;
  const userMarkerShape = STATE.markerShape || 'circle';
  const userMarkerOpacity = STATE.markerOpacity != null ? STATE.markerOpacity : (theme.opacity ?? 0.88);

  return (traces || []).map((trace, i) => {
    const t = { ...trace };
    const color = (STATE.userColors && STATE.userColors[i]) || palette[i % palette.length];
    const mode = String(t.mode || '');

    if (t.type === 'scatter' || t.type === 'scattergl') {
      const isLine = mode.includes('lines');
      const isMarker = mode.includes('markers') || (!mode && !isLine);
      const hasArrayColor = Array.isArray(t.marker?.color);

      if (isLine) {
        t.line = {
          ...(t.line || {}),
          color: color,
          width: userLineWidth,
          shape: t.line?.shape || 'spline',
          smoothing: t.line?.smoothing ?? 0.4,
        };
      }
      if (isMarker) {
        t.marker = {
          ...(t.marker || {}),
          color: hasArrayColor ? t.marker.color : color,
          size: userMarkerSize,
          symbol: userMarkerShape,
          opacity: userMarkerOpacity,
          line: { color: markerLine, width: 0.8 },
        };
      }
      if (!isLine && !isMarker) {
        t.mode = 'lines+markers';
        t.line = { color, width: userLineWidth, shape: 'spline', smoothing: 0.4 };
        t.marker = {
          color, size: userMarkerSize, symbol: userMarkerShape,
          opacity: userMarkerOpacity, line: { color: markerLine, width: 0.8 },
        };
      }
    }

    if (t.type === 'bar') {
      const hasArrayColor = Array.isArray(t.marker?.color);
      t.marker = {
        ...(t.marker || {}),
        color: hasArrayColor ? t.marker.color : color,
        opacity: userMarkerOpacity,
        line: { color: '#ffffff', width: 0.6 },
      };
      t.textposition = t.textposition || 'outside';
      t.textfont = { ...(t.textfont || {}), family: theme.fontFamily, size: 10, color: ink };
    }

    if (t.type === 'histogram') {
      t.marker = { ...(t.marker || {}), color, opacity: userMarkerOpacity, line: { color: '#ffffff', width: 0.6 } };
    }

    if (t.type === 'box') {
      t.line = { ...(t.line || {}), color, width: userLineWidth };
      t.fillcolor = t.fillcolor || withAlpha(color, 0.15);
      t.marker = { ...(t.marker || {}), color, size: userMarkerSize, opacity: userMarkerOpacity };
      t.boxmean = t.boxmean !== undefined ? t.boxmean : true;
    }

    if (t.type === 'violin') {
      t.line = { ...(t.line || {}), color, width: userLineWidth };
      t.fillcolor = t.fillcolor || withAlpha(color, 0.22);
      t.meanline = { visible: true, color: ink, width: 1.2 };
    }

    if (t.type === 'heatmap') {
      t.hoverongaps = false;
      if (t.colorbar) {
        t.colorbar = {
          thickness: 18, len: 0.82, outlinewidth: 0,
          tickfont: { family: theme.fontFamily, size: 10, color: ink },
          ...t.colorbar,
        };
      }
    }

    return t;
  });
}

// ── Polish layout for publication quality ────────────────
function polishLayoutForPublication(layout, chartType, theme) {
  const l = { ...layout };
  const ink = theme.ink || '#111827';
  const family = theme.fontFamily || "'Arial', 'Noto Sans SC', sans-serif";
  const axisColor = theme.axisLineColor || '#26313D';

  l.paper_bgcolor = theme.bgColor || '#ffffff';
  l.plot_bgcolor = theme.plotBgColor || '#fafcfb';
  l.separators = '.';
  l.hovermode = l.hovermode || 'closest';
  l.dragmode = l.dragmode || 'pan';

  if (STATE.barGap != null) {
    l.bargap = STATE.barGap;
  }

  // Global font
  l.font = {
    family: family,
    color: ink,
    size: theme.tickFontSize || 12,
  };

  // Hover tooltip styling
  l.hoverlabel = {
    bgcolor: '#ffffff',
    bordercolor: theme.axisLineColor || '#D7DEE8',
    font: { family, color: ink, size: 12 },
    ...(l.hoverlabel || {}),
  };

  // Title styling
  if (typeof l.title === 'string') {
    l.title = { text: l.title };
  }
  l.title = {
    x: 0.02,
    xanchor: 'left',
    font: { family, size: (theme.titleFontSize || 18), color: theme.titleColor || ink },
    ...(l.title || {}),
  };

  // Legend
  l.legend = {
    orientation: 'h',
    x: 0, y: -0.18,
    xanchor: 'left', yanchor: 'top',
    bgcolor: 'rgba(255,255,255,0)',
    borderwidth: 0,
    font: { family, size: (theme.legendFontSize || 11), color: ink },
    ...(l.legend || {}),
  };

  // Axis styling
  const axisKeys = Object.keys(l).filter(k => /^xaxis|^yaxis/.test(k));
  axisKeys.forEach((key) => {
    const prev = l[key] || {};
    if (prev.visible === false) return;
    const isY = key.startsWith('y');
    l[key] = {
      showline: true,
      linewidth: 1.8,
      linecolor: axisColor,
      mirror: false,
      ticks: 'outside',
      ticklen: 5,
      tickwidth: 1.2,
      tickcolor: axisColor,
      zeroline: false,
      showgrid: isY,
      gridcolor: theme.gridColor || 'rgba(31,41,55,0.07)',
      gridwidth: 0.8,
      automargin: true,
      tickfont: { family, size: (theme.tickFontSize || 11), color: ink },
      title: {
        font: { family, size: (theme.axisFontSize || 13), color: ink },
        standoff: 12,
        ...(typeof prev.title === 'string' ? { text: prev.title } : (prev.title || {})),
      },
      ...prev,
    };
  });

  return l;
}

function withAlpha(hex, alpha) {
  if (!hex || !String(hex).startsWith('#')) return hex;
  const clean = String(hex).slice(1);
  const full = clean.length === 3 ? clean.split('').map(c => c + c).join('') : clean;
  const n = parseInt(full, 16);
  if (Number.isNaN(n)) return hex;
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}
