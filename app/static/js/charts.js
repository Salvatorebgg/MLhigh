/* ── MLhigh Charts Module — Publication-quality centered rendering ── */

function renderAllCharts(charts) {
  if (!charts || charts.length === 0) return;
  const container = el('chartPreviewContainer');
  if (!container) return;

  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A', '#6F5AA7'];

  disconnectResizeObserver();
  purgePlotlyChildren(container);
  STATE.currentChartBundle = charts;
  STATE.currentRenderedPlots = [];
  STATE.currentPlotlyData = null;
  STATE.currentPlotlyLayout = null;
  container.classList.remove('chart-gallery');

  // Render single chart directly into container
  if (charts.length === 1 && charts[0].plotly) {
    container.innerHTML = '';
    const plotMount = document.createElement('div');
    plotMount.className = 'chart-plot';
    container.appendChild(plotMount);
    setTimeout(() => {
      renderCenteredChart(plotMount, charts[0].plotly, palette, theme, {
        index: 0,
        title: charts[0].title || 'Chart 1',
        role: 'primary',
      });
    }, 50);
    return;
  }

  container.classList.add('chart-gallery');
  const gallery = document.createElement('div');
  gallery.className = 'chart-gallery-list';
  charts.forEach((chart, i) => {
    const chartId = `result-chart-${i}`;
    const card = document.createElement('div');
    card.className = 'result-card chart-gallery-card';
    card.innerHTML = `
      <h3 class="result-card-title">${escapeHtml(chart.title || '图表 ' + (i + 1))}</h3>
      <div class="result-chart-container" id="${chartId}"></div>
    `;
    gallery.appendChild(card);
  });
  container.innerHTML = '';
  container.appendChild(gallery);

  charts.forEach((chart, i) => {
    const chartId = `result-chart-${i}`;
    if (chart.plotly) {
      setTimeout(() => {
        const cardContainer = el(chartId);
        if (!cardContainer) return;
        const plotMount = document.createElement('div');
        plotMount.className = 'chart-plot';
        cardContainer.appendChild(plotMount);
        renderCenteredChart(plotMount, chart.plotly, palette, theme, {
          index: i,
          title: chart.title || 'Chart ' + (i + 1),
          role: 'gallery',
        });
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

  purgePlotlyChildren(container);
  let html = '';
  diagnostics.forEach((d, i) => {
    const chartId = `diag-chart-${i}`;
    html += `<div class="result-card diag-card">
      <h3 class="result-card-title">${escapeHtml(d.title || '诊断 ' + (i + 1))}</h3>
      <div class="result-chart-container chart-diag-container" id="${chartId}"></div>
    </div>`;
  });
  container.innerHTML = html;

  diagnostics.forEach((d, i) => {
    if (d.plotly) {
      setTimeout(() => {
        const cardContainer = el(`diag-chart-${i}`);
        if (!cardContainer) return;
        const plotMount = document.createElement('div');
        plotMount.className = 'chart-plot';
        cardContainer.appendChild(plotMount);
        renderCenteredChart(plotMount, d.plotly, palette, theme, {
          index: i,
          title: d.title || 'Diagnostic ' + (i + 1),
          role: 'diagnostic',
          trackState: false,
        });
      }, i * 80);
    }
  });
}

// ── Core centered chart renderer (Basicpicture approach) ──
function renderCenteredChart(plotMount, plotlyJson, palette, theme, options = {}) {
  if (!plotMount || !window.Plotly) return;

  try {
    const fig = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
    if (!fig || !fig.data) return;

    let traces = Array.isArray(fig.data) ? fig.data : [fig.data];
    let layout = fig.layout || {};
    const trackState = options.trackState !== false;

    // Apply user's chart title if set
    if (STATE.chartTitle) {
      if (typeof layout.title === 'object') layout.title.text = STATE.chartTitle;
      else layout.title = { text: STATE.chartTitle };
    }

    const defaultMargin = { l: 72, r: 48, t: 72, b: 72 };

    // Polish
    traces = polishTracesForPublication(traces, theme, palette);
    layout = applyThemeLayout(layout, theme);
    layout.margin = { ...defaultMargin, ...(layout.margin || {}) };
    layout = polishLayoutForPublication(layout, '', theme);

    if (layout.showlegend === undefined) {
      layout.showlegend = traces.some(t => t && t.showlegend !== false && t.name);
    }
    if (STATE.barGap != null && traces.some(t => t.type === 'bar' || t.type === 'histogram')) {
      layout.bargap = STATE.barGap;
    }

    // Fit chart to frame — centered with whitespace, not stretched
    const frameSize = fitChartPlotToFrame(plotMount);
    layout.width = frameSize.width;
    layout.height = frameSize.height;
    layout.autosize = false;

    if (trackState && (options.index === 0 || !STATE.currentPlotlyData)) {
      STATE.currentPlotlyData = traces;
      STATE.currentPlotlyLayout = layout;
    }

    Plotly.newPlot(plotMount, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
      toImageButtonOptions: {
        format: 'png', height: 1440, width: 2160, scale: 2,
        filename: 'mlhigh_chart_' + Date.now(),
      },
    }).then(() => {
      if (trackState) {
        const renderedEntry = {
          el: plotMount,
          index: Number.isFinite(options.index) ? options.index : STATE.currentRenderedPlots.length,
          role: options.role || 'primary',
          title: options.title || extractPlotTitle(layout) || 'Chart',
          data: traces,
          layout,
        };
        STATE.currentRenderedPlots = (STATE.currentRenderedPlots || [])
          .filter(entry => entry && entry.el !== plotMount);
        STATE.currentRenderedPlots.push(renderedEntry);
        STATE.currentRenderedPlots.sort((a, b) => (a.index || 0) - (b.index || 0));
      }
      syncCurrentPlotlyLayoutFromDom(plotMount, trackState);
      installChartResizeObserver(plotMount, { trackState });
    });

  } catch (e) {
    plotMount.innerHTML = '<div class="empty-state small">图表渲染失败</div>';
    console.error('Chart render error:', e);
  }
}

// ── Frame sizing: centered with whitespace ──────────────
function getChartFrameSize(plotMount) {
  const preview = plotMount.parentElement;
  const previewStyle = preview ? getComputedStyle(preview) : null;
  const padX = previewStyle ? parseFloat(previewStyle.paddingLeft || 0) + parseFloat(previewStyle.paddingRight || 0) : 0;
  const padY = previewStyle ? parseFloat(previewStyle.paddingTop || 0) + parseFloat(previewStyle.paddingBottom || 0) : 0;
  const maxW = Math.max(520, Math.floor((preview?.clientWidth || 960) - padX));
  const maxH = Math.max(440, Math.floor((preview?.clientHeight || 680) - padY));
  const aspect = 1.62;
  const usableW = Math.max(360, Math.floor(maxW * 0.90));
  const usableH = Math.max(360, Math.floor(maxH * 0.88));
  let width = Math.floor(usableH * aspect);
  let height = usableH;
  if (width > usableW) {
    width = usableW;
    height = Math.floor(width / aspect);
  }
  const minH = 480;
  if (height < Math.min(minH, usableH)) {
    height = Math.min(minH, usableH);
    width = Math.min(usableW, Math.floor(height * aspect));
  }
  return { width, height };
}

function fitChartPlotToFrame(plotMount) {
  const size = getChartFrameSize(plotMount);
  plotMount.style.setProperty('width', `${size.width}px`, 'important');
  plotMount.style.setProperty('height', `${size.height}px`, 'important');
  plotMount.style.setProperty('min-height', `${size.height}px`, 'important');
  plotMount.style.setProperty('max-width', '100%', 'important');
  plotMount.style.setProperty('max-height', '100%', 'important');
  return size;
}

function syncCurrentPlotlyLayoutFromDom(plotMount, trackState = true) {
  if (!plotMount || !plotMount._fullLayout) return;
  const matchingEntry = (STATE.currentRenderedPlots || []).find(entry => entry && entry.el === plotMount);
  if (!trackState && !matchingEntry) return;
  const baseLayout = matchingEntry?.layout || STATE.currentPlotlyLayout || {};
  const width = Math.floor(plotMount._fullLayout.width || plotMount.clientWidth || baseLayout.width || 0);
  const height = Math.floor(plotMount._fullLayout.height || plotMount.clientHeight || baseLayout.height || 0);
  if (width > 0 && height > 0) {
    const nextLayout = { ...baseLayout, width, height, autosize: false };
    if (matchingEntry) matchingEntry.layout = nextLayout;
    if (!matchingEntry || matchingEntry.index === 0) STATE.currentPlotlyLayout = nextLayout;
  }
}

function installChartResizeObserver(plotMount, options = {}) {
  if (!window.ResizeObserver) return;
  let resizeFrame = null;
  const observer = new ResizeObserver(() => {
    if (resizeFrame) cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      const size = fitChartPlotToFrame(plotMount);
      if (window.Plotly && plotMount.isConnected) {
        const p = Plotly.relayout(plotMount, { width: size.width, height: size.height, autosize: false });
        if (p && typeof p.then === 'function') {
          p.then(() => syncCurrentPlotlyLayoutFromDom(plotMount, options.trackState !== false));
        } else {
          syncCurrentPlotlyLayoutFromDom(plotMount, options.trackState !== false);
        }
      }
    });
  });
  observer.observe(plotMount.parentElement || plotMount);
  if (!Array.isArray(STATE.currentChartResizeObserver)) {
    STATE.currentChartResizeObserver = STATE.currentChartResizeObserver ? [STATE.currentChartResizeObserver] : [];
  }
  STATE.currentChartResizeObserver.push(observer);
}

function disconnectResizeObserver() {
  const observers = Array.isArray(STATE.currentChartResizeObserver)
    ? STATE.currentChartResizeObserver
    : (STATE.currentChartResizeObserver ? [STATE.currentChartResizeObserver] : []);
  observers.forEach(observer => {
    if (observer && typeof observer.disconnect === 'function') observer.disconnect();
  });
  STATE.currentChartResizeObserver = null;
}

function purgePlotlyChildren(container) {
  if (!container || !window.Plotly) return;
  container.querySelectorAll('.js-plotly-plot').forEach(plot => {
    try { Plotly.purge(plot); } catch (e) { console.warn('Plotly purge skipped:', e); }
  });
}

function extractPlotTitle(layout) {
  const title = layout && layout.title;
  if (!title) return '';
  if (typeof title === 'string') return title;
  if (typeof title.text === 'string') return title.text.replace(/<[^>]*>/g, '').trim();
  return '';
}

// ── Publication chart renderer (legacy, for external calls) ──
function renderPublicationChart(containerId, plotlyJson, palette, theme) {
  const container = el(containerId);
  if (!container || !window.Plotly) return;
  try {
    const fig = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
    if (!fig || !fig.data) return;
    let traces = Array.isArray(fig.data) ? fig.data : [fig.data];
    let layout = fig.layout || {};
    traces = polishTracesForPublication(traces, theme, palette);
    layout = applyThemeLayout(layout, theme);
    if (STATE.chartTitle) {
      if (typeof layout.title === 'object') layout.title.text = STATE.chartTitle;
      else layout.title = { text: STATE.chartTitle };
    }
    const defaultMargin = { l: 72, r: 48, t: 72, b: 72 };
    layout.margin = { ...defaultMargin, ...(layout.margin || {}) };
    layout = polishLayoutForPublication(layout, '', theme);
    const frameSize = fitChartPlotToFrame(container);
    layout.width = frameSize.width;
    layout.height = frameSize.height;
    layout.autosize = false;
    Plotly.newPlot(container, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
    }).then(() => {
      syncCurrentPlotlyLayoutFromDom(container);
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

    if (t.type === 'heatmap' || t.type === 'heatmapgl') {
      t.hoverongaps = false;
      if (t.colorbar) {
        t.colorbar = {
          thickness: 18, len: 0.82, outlinewidth: 0,
          tickfont: { family: theme.fontFamily, size: 10, color: ink },
          ...t.colorbar,
        };
      }
    }

    if (t.type === 'contour' || t.type === 'surface') {
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

  l.font = {
    family: family,
    color: ink,
    size: theme.tickFontSize || 12,
  };

  l.hoverlabel = {
    bgcolor: '#ffffff',
    bordercolor: theme.axisLineColor || '#D7DEE8',
    font: { family, color: ink, size: 12 },
    ...(l.hoverlabel || {}),
  };

  if (typeof l.title === 'string') {
    l.title = { text: l.title };
  }
  l.title = {
    x: 0.02,
    xanchor: 'left',
    font: { family, size: (theme.titleFontSize || 18), color: theme.titleColor || ink },
    ...(l.title || {}),
  };

  l.legend = {
    orientation: 'h',
    x: 0, y: -0.18,
    xanchor: 'left', yanchor: 'top',
    bgcolor: 'rgba(255,255,255,0)',
    borderwidth: 0,
    font: { family, size: (theme.legendFontSize || 11), color: ink },
    ...(l.legend || {}),
  };

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
