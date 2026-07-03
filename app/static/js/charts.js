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


/* ============================================================
   v2 chart sizing override: smaller, centered, leaves room for right parameters
   ============================================================ */
function getChartFrameSize(plotMount) {
  const preview = plotMount.parentElement;
  const previewStyle = preview ? getComputedStyle(preview) : null;
  const padX = previewStyle ? parseFloat(previewStyle.paddingLeft || 0) + parseFloat(previewStyle.paddingRight || 0) : 0;
  const padY = previewStyle ? parseFloat(previewStyle.paddingTop || 0) + parseFloat(previewStyle.paddingBottom || 0) : 0;
  const maxW = Math.max(420, Math.floor((preview?.clientWidth || 820) - padX));
  const maxH = Math.max(360, Math.floor((preview?.clientHeight || 560) - padY));
  const aspect = 1.58;
  const usableW = Math.max(320, Math.floor(maxW * 0.72));
  const usableH = Math.max(280, Math.floor(maxH * 0.66));
  let width = Math.floor(usableH * aspect);
  let height = usableH;
  if (width > usableW) {
    width = usableW;
    height = Math.floor(width / aspect);
  }
  const minH = Math.min(360, usableH);
  if (height < minH) {
    height = minH;
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
  plotMount.style.setProperty('margin', 'auto', 'important');
  return size;
}


/* ============================================================
   v11 visual controls alignment with jichutongji_zhong
   ============================================================ */
const __mlhighOldPolishLayoutV11 = typeof polishLayoutForPublication === 'function' ? polishLayoutForPublication : null;
function polishLayoutForPublication(layout, chartType, theme) {
  let l = __mlhighOldPolishLayoutV11 ? __mlhighOldPolishLayoutV11(layout, chartType, theme) : { ...(layout || {}) };
  const family = theme?.fontFamily || "'Arial', 'Noto Sans SC', sans-serif";
  const tickSize = Number(STATE.axisTickFontSize || 11);
  const titleSize = Number(STATE.axisTitleFontSize || 13);
  const labelSize = Number(STATE.labelFontSize || 12);
  const ink = theme?.ink || '#111827';

  const patchAxis = (axis) => {
    const a = { ...(axis || {}) };
    a.tickfont = { ...(a.tickfont || {}), family, size: tickSize, color: (a.tickfont && a.tickfont.color) || ink };
    if (typeof a.title === 'string') {
      a.title = { text: a.title, font: { family, size: titleSize, color: ink } };
    } else {
      a.title = { ...(a.title || {}), font: { ...((a.title && a.title.font) || {}), family, size: titleSize, color: ink } };
    }
    if (STATE.chartBackgroundMode === 'grid') {
      a.showgrid = true;
      a.gridcolor = 'rgba(148,163,184,.22)';
      a.gridwidth = 1;
      a.zeroline = false;
    } else {
      a.showgrid = false;
      a.zeroline = false;
    }
    return a;
  };

  Object.keys(l).forEach(k => {
    if (/^xaxis\d*$/.test(k) || /^yaxis\d*$/.test(k)) l[k] = patchAxis(l[k]);
  });
  l.xaxis = patchAxis(l.xaxis);
  l.yaxis = patchAxis(l.yaxis);

  if (l.legend) l.legend = { ...l.legend, font: { ...((l.legend && l.legend.font) || {}), family, size: labelSize, color: ink } };
  if (Array.isArray(l.annotations)) l.annotations = l.annotations.map(a => ({ ...a, font: { ...((a && a.font) || {}), family, size: labelSize, color: (a.font && a.font.color) || ink } }));

  if (STATE.chartBackgroundMode === 'transparent') {
    l.paper_bgcolor = 'rgba(0,0,0,0)';
    l.plot_bgcolor = 'rgba(0,0,0,0)';
  } else if (STATE.chartBackgroundMode === 'blank') {
    l.paper_bgcolor = '#ffffff';
    l.plot_bgcolor = '#ffffff';
  }
  return l;
}

const __mlhighOldFrameSizeV11 = typeof getChartFrameSize === 'function' ? getChartFrameSize : null;
function getChartFrameSize(plotMount) {
  const preview = plotMount.parentElement;
  const previewStyle = preview ? getComputedStyle(preview) : null;
  const padX = previewStyle ? parseFloat(previewStyle.paddingLeft || 0) + parseFloat(previewStyle.paddingRight || 0) : 0;
  const padY = previewStyle ? parseFloat(previewStyle.paddingTop || 0) + parseFloat(previewStyle.paddingBottom || 0) : 0;
  const maxW = Math.max(420, Math.floor((preview?.clientWidth || 820) - padX));
  const maxH = Math.max(320, Math.floor((preview?.clientHeight || 560) - padY));
  const aspect = 1.56;
  const usableW = Math.max(320, Math.floor(maxW * 0.80));
  const usableH = Math.max(280, Math.floor(maxH * 0.78));
  let width = Math.floor(usableH * aspect);
  let height = usableH;
  if (width > usableW) {
    width = usableW;
    height = Math.floor(width / aspect);
  }
  return { width: Math.max(320, width), height: Math.max(280, height) };
}

function fitChartPlotToFrame(plotMount) {
  const size = getChartFrameSize(plotMount);
  plotMount.style.setProperty('width', `${size.width}px`, 'important');
  plotMount.style.setProperty('height', `${size.height}px`, 'important');
  plotMount.style.setProperty('min-height', `${size.height}px`, 'important');
  plotMount.style.setProperty('max-width', '100%', 'important');
  plotMount.style.setProperty('max-height', '100%', 'important');
  plotMount.style.setProperty('margin', 'auto', 'important');
  return size;
}


/* ============================================================
   v12 final chart render fix
   - remove recursive layout override from v11 by replacing with a standalone function
   - keep controls aligned with jichutongji_zhong
   ============================================================ */
function polishLayoutForPublication(layout, chartType, theme) {
  const l = { ...(layout || {}) };
  const family = theme?.fontFamily || "'Arial', 'Noto Sans SC', sans-serif";
  const ink = theme?.ink || '#111827';
  const axisColor = theme?.axisLineColor || '#26313D';
  const tickSize = Number(STATE.axisTickFontSize || theme?.tickFontSize || 11);
  const titleSize = Number(STATE.axisTitleFontSize || theme?.axisFontSize || 13);
  const labelSize = Number(STATE.labelFontSize || 12);

  l.font = { ...(l.font || {}), family, color: ink };
  if (typeof l.title === 'string') {
    l.title = { text: l.title, font: { family, size: Math.max(18, labelSize + 6), color: ink }, x: 0.04, xanchor: 'left' };
  } else if (l.title) {
    l.title = { ...l.title, font: { ...((l.title && l.title.font) || {}), family, size: Math.max(18, labelSize + 6), color: ink } };
  }

  const patchAxis = (axis, isY = false) => {
    const prev = { ...(axis || {}) };
    if (prev.visible === false) return prev;
    const titleText = typeof prev.title === 'string' ? prev.title : (prev.title && prev.title.text) || '';
    const mode = STATE.chartBackgroundMode || 'grid';
    return {
      ...prev,
      showline: true,
      linewidth: 1.5,
      linecolor: axisColor,
      mirror: false,
      ticks: 'outside',
      ticklen: 5,
      tickwidth: 1,
      tickcolor: axisColor,
      zeroline: false,
      showgrid: mode === 'grid' && isY,
      gridcolor: 'rgba(148,163,184,.22)',
      gridwidth: 1,
      automargin: true,
      tickfont: { ...(prev.tickfont || {}), family, size: tickSize, color: (prev.tickfont && prev.tickfont.color) || ink },
      title: { ...(typeof prev.title === 'object' ? prev.title : {}), text: titleText, font: { ...((prev.title && prev.title.font) || {}), family, size: titleSize, color: ink }, standoff: 12 },
    };
  };

  Object.keys(l).forEach(k => {
    if (/^xaxis\d*$/.test(k)) l[k] = patchAxis(l[k], false);
    if (/^yaxis\d*$/.test(k)) l[k] = patchAxis(l[k], true);
  });
  l.xaxis = patchAxis(l.xaxis, false);
  l.yaxis = patchAxis(l.yaxis, true);

  l.legend = { ...(l.legend || {}), font: { ...((l.legend && l.legend.font) || {}), family, size: labelSize, color: ink }, orientation: l.legend?.orientation || 'h' };
  if (Array.isArray(l.annotations)) {
    l.annotations = l.annotations.map(a => ({ ...a, font: { ...((a && a.font) || {}), family, size: labelSize, color: (a.font && a.font.color) || ink } }));
  }

  if (STATE.chartBackgroundMode === 'transparent') {
    l.paper_bgcolor = 'rgba(0,0,0,0)';
    l.plot_bgcolor = 'rgba(0,0,0,0)';
  } else {
    l.paper_bgcolor = '#ffffff';
    l.plot_bgcolor = '#ffffff';
  }
  return l;
}

function getChartFrameSize(plotMount) {
  const preview = plotMount.parentElement;
  const previewStyle = preview ? getComputedStyle(preview) : null;
  const padX = previewStyle ? parseFloat(previewStyle.paddingLeft || 0) + parseFloat(previewStyle.paddingRight || 0) : 0;
  const padY = previewStyle ? parseFloat(previewStyle.paddingTop || 0) + parseFloat(previewStyle.paddingBottom || 0) : 0;
  const maxW = Math.max(420, Math.floor((preview?.clientWidth || 820) - padX));
  const maxH = Math.max(320, Math.floor((preview?.clientHeight || 560) - padY));
  const aspect = 1.56;
  const usableW = Math.max(320, Math.floor(maxW * 0.78));
  const usableH = Math.max(280, Math.floor(maxH * 0.76));
  let width = Math.floor(usableH * aspect);
  let height = usableH;
  if (width > usableW) {
    width = usableW;
    height = Math.floor(width / aspect);
  }
  return { width: Math.max(320, width), height: Math.max(280, height) };
}

function fitChartPlotToFrame(plotMount) {
  const size = getChartFrameSize(plotMount);
  plotMount.style.setProperty('width', `${size.width}px`, 'important');
  plotMount.style.setProperty('height', `${size.height}px`, 'important');
  plotMount.style.setProperty('min-height', `${size.height}px`, 'important');
  plotMount.style.setProperty('max-width', '100%', 'important');
  plotMount.style.setProperty('max-height', '100%', 'important');
  plotMount.style.setProperty('margin', 'auto', 'important');
  return size;
}


/* ============================================================
   v14 interactive chart switching + variable-wise colors + arrow axes
   ============================================================ */

function getActiveChartDataV14() {
  const bundle = STATE.currentChartBundle || [];
  const idx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  return bundle[idx] || null;
}

function normalizePlotlyObjV14(plotly) {
  if (!plotly) return null;
  if (typeof plotly === 'string') {
    try { return JSON.parse(plotly); } catch (_) { return null; }
  }
  return plotly;
}

function renderActiveChartV14() {
  const container = el('chartActivePlot');
  if (!container) return;
  const chart = getActiveChartDataV14();
  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A'];
  purgePlotlyChildren(container);
  container.innerHTML = '';
  if (!chart || !chart.plotly) {
    container.innerHTML = '<div class="empty-state small">暂无可显示图形</div>';
    return;
  }
  const plotMount = document.createElement('div');
  plotMount.className = 'chart-plot';
  container.appendChild(plotMount);
  renderCenteredChart(plotMount, chart.plotly, palette, theme, {
    index: STATE.activeChartIndex || 0,
    title: chart.title || 'Chart',
    role: 'active',
  });
  if (typeof renderAppearanceControls === 'function') {
    setTimeout(renderAppearanceControls, 60);
  }
}

function renderChartSwitcherV14(charts) {
  const switcher = el('chartVariantTabs');
  if (!switcher) return;
  if (!charts || charts.length <= 1) {
    switcher.innerHTML = '';
    switcher.style.display = 'none';
    return;
  }
  switcher.style.display = 'flex';
  /* Icon-style labels: numbered chart tabs */
  const icons = ['📈', '📊', '📉', '🔬', '📋', '🧪', '💠', '◆'];
  switcher.innerHTML = charts.map((c, i) => {
    const icon = icons[i] || '▪';
    const label = c.title || `图${i + 1}`;
    return `
    <button type="button" class="chart-variant-tab ${i === (STATE.activeChartIndex || 0) ? 'active' : ''}" data-chart-index="${i}" title="${escapeHtml(label)}">
      <span class="tab-icon">${icon}</span><span class="tab-num">${i + 1}</span>
    </button>
  `;}).join('');
}

function renderAllCharts(charts) {
  const container = el('chartPreviewContainer');
  if (!container) return;
  disconnectResizeObserver();
  purgePlotlyChildren(container);
  STATE.currentRenderedPlots = [];
  STATE.currentChartBundle = charts || [];
  STATE.activeChartIndex = Math.min(STATE.activeChartIndex || 0, Math.max(0, STATE.currentChartBundle.length - 1));
  container.classList.remove('chart-gallery');
  container.innerHTML = `
    <div id="chartVariantTabs" class="chart-variant-tabs"></div>
    <div id="chartActivePlot" class="chart-active-plot"></div>
  `;
  renderChartSwitcherV14(STATE.currentChartBundle);
  renderActiveChartV14();
}

document.addEventListener('click', function(ev) {
  const btn = ev.target.closest && ev.target.closest('.chart-variant-tab');
  if (!btn) return;
  ev.preventDefault();
  const idx = Number(btn.dataset.chartIndex || 0);
  STATE.activeChartIndex = idx;
  qsa('.chart-variant-tab').forEach(b => b.classList.toggle('active', Number(b.dataset.chartIndex) === idx));
  STATE.userColors = STATE.userColorsByChart?.[idx] || STATE.userColors || null;
  renderActiveChartV14();
}, true);

const __oldPolishTracesV14 = typeof polishTracesForPublication === 'function' ? polishTracesForPublication : null;
function polishTracesForPublication(traces, theme, palette) {
  let polished = __oldPolishTracesV14 ? __oldPolishTracesV14(traces, theme, palette) : (traces || []).map(t => ({...t}));
  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  const barWidth = STATE.barWidth != null ? Number(STATE.barWidth) : 0.68;
  const categoryColors = (STATE.userBarCategoryColorsByChart || {})[activeIdx] || {};
  const userOpacity = STATE.markerOpacity != null ? Number(STATE.markerOpacity) : 0.88;
  polished = polished.map((trace, ti) => {
    const t = { ...trace };
    if (t.type === 'bar') {
      t.width = barWidth;
      const xvals = Array.isArray(t.x) ? t.x.map(v => String(v)) : [];
      if (xvals.length) {
        const colors = xvals.map((x, j) => categoryColors[x] || (Array.isArray(t.marker?.color) ? t.marker.color[j] : ((STATE.userColors && STATE.userColors[ti]) || palette[ti % palette.length])));
        t.marker = { ...(t.marker || {}), color: colors, opacity: userOpacity, line: { color: '#ffffff', width: 0.8 } };
      }
    }
    return t;
  });
  return polished;
}

const __oldPolishLayoutV14 = typeof polishLayoutForPublication === 'function' ? polishLayoutForPublication : null;
function polishLayoutForPublication(layout, chartType, theme) {
  let l = __oldPolishLayoutV14 ? __oldPolishLayoutV14(layout, chartType, theme) : { ...(layout || {}) };
  // Arrowed axes using paper-coordinate annotations.
  const arrows = [
    { x: 1.018, y: 0, ax: 0.965, ay: 0, xref: 'paper', yref: 'paper', axref: 'paper', ayref: 'paper',
      showarrow: true, arrowhead: 3, arrowsize: 1.15, arrowwidth: 1.6, arrowcolor: '#111827', text: '' },
    { x: 0, y: 1.018, ax: 0, ay: 0.965, xref: 'paper', yref: 'paper', axref: 'paper', ayref: 'paper',
      showarrow: true, arrowhead: 3, arrowsize: 1.15, arrowwidth: 1.6, arrowcolor: '#111827', text: '' },
  ];
  const oldAnn = Array.isArray(l.annotations) ? l.annotations.filter(a => !a._axisArrow) : [];
  l.annotations = [...oldAnn, ...arrows.map(a => ({ ...a, _axisArrow: true }))];
  l.xaxis = { ...(l.xaxis || {}), showline: true, linecolor: '#111827', linewidth: 1.6, mirror: false };
  l.yaxis = { ...(l.yaxis || {}), showline: true, linecolor: '#111827', linewidth: 1.6, mirror: false };
  return l;
}



/* v14.1 safe arrow-axis layout override: no custom annotation fields */
const __oldPolishLayoutV141 = typeof polishLayoutForPublication === 'function' ? polishLayoutForPublication : null;
function polishLayoutForPublication(layout, chartType, theme) {
  let l = __oldPolishLayoutV141 ? __oldPolishLayoutV141(layout, chartType, theme) : { ...(layout || {}) };
  const baseAnnotations = Array.isArray(l.annotations)
    ? l.annotations.filter(a => !(a && a.text === '' && a.showarrow === true && a.xref === 'paper' && a.yref === 'paper' && (a.x > 1 || a.y > 1)))
    : [];
  const arrows = [
    { x: 1.018, y: 0, ax: 0.965, ay: 0, xref: 'paper', yref: 'paper', axref: 'paper', ayref: 'paper',
      showarrow: true, arrowhead: 3, arrowsize: 1.15, arrowwidth: 1.6, arrowcolor: '#111827', text: '' },
    { x: 0, y: 1.018, ax: 0, ay: 0.965, xref: 'paper', yref: 'paper', axref: 'paper', ayref: 'paper',
      showarrow: true, arrowhead: 3, arrowsize: 1.15, arrowwidth: 1.6, arrowcolor: '#111827', text: '' },
  ];
  l.annotations = [...baseAnnotations, ...arrows];
  l.xaxis = { ...(l.xaxis || {}), showline: true, linecolor: '#111827', linewidth: 1.6, mirror: false };
  l.yaxis = { ...(l.yaxis || {}), showline: true, linecolor: '#111827', linewidth: 1.6, mirror: false };
  return l;
}


/* ============================================================
   v15 final chart polish: strict control-to-chart mapping
   ============================================================ */
const __oldPolishTracesV15 = typeof polishTracesForPublication === 'function' ? polishTracesForPublication : null;
function polishTracesForPublication(traces, theme, palette) {
  let polished = __oldPolishTracesV15 ? __oldPolishTracesV15(traces, theme, palette) : (traces || []).map(t => ({ ...t }));
  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  const barWidth = STATE.barWidth != null ? Number(STATE.barWidth) : 0.68;
  const opacity = STATE.markerOpacity != null ? Number(STATE.markerOpacity) : 0.88;
  const markerSize = STATE.markerSize != null ? Number(STATE.markerSize) : 8;
  const lineWidth = STATE.lineWidth != null ? Number(STATE.lineWidth) : 2.5;
  const markerShape = STATE.markerShape || 'circle';
  const lineDash = STATE.lineDash || 'solid';
  const traceColors = (STATE.userTraceColorsByChart || {})[activeIdx] || [];
  const barCategoryColors = (STATE.userBarCategoryColorsByChart || {})[activeIdx] || {};
  const colorway = palette || ['#2563eb','#ef4444','#0f766e','#eab308','#7c3aed'];

  return polished.map((trace, ti) => {
    const t = { ...trace };
    const traceColor = traceColors[ti] || (t.marker && typeof t.marker.color === 'string' ? t.marker.color : (t.line && t.line.color) || colorway[ti % colorway.length]);
    if (t.type === 'bar' || t.type === 'histogram') {
      t.width = barWidth;
      const xvals = Array.isArray(t.x) ? t.x.map(v => String(v)) : [];
      const yvals = Array.isArray(t.y) ? t.y.map(v => String(v)) : [];
      const cats = xvals.length >= yvals.length ? xvals : yvals;
      const colors = cats.length ? cats.map(cat => barCategoryColors[cat] || traceColor) : traceColor;
      t.marker = { ...(t.marker || {}), color: colors, opacity, line: { color: '#ffffff', width: 0.8 } };
      return t;
    }
    if (t.type === 'scatter') {
      const mode = String(t.mode || 'markers');
      if (mode.includes('lines')) {
        t.line = { ...(t.line || {}), color: traceColor, width: lineWidth, dash: (t.line && t.line.dash) || lineDash };
      }
      if (mode.includes('markers')) {
        t.marker = { ...(t.marker || {}), color: traceColor, size: markerSize, symbol: markerShape, opacity, line: { color: '#ffffff', width: 0.6 } };
      }
      if (!mode.includes('markers') && !mode.includes('lines')) {
        t.marker = { ...(t.marker || {}), color: traceColor, size: markerSize, symbol: markerShape, opacity };
      }
      return t;
    }
    if (t.type === 'box' || t.type === 'violin') {
      t.line = { ...(t.line || {}), color: traceColor, width: lineWidth, dash: lineDash };
      t.marker = { ...(t.marker || {}), color: traceColor, size: markerSize, symbol: markerShape, opacity };
      t.fillcolor = t.fillcolor || traceColor;
      t.opacity = opacity;
      return t;
    }
    if (t.type === 'pie') {
      if (Array.isArray(t.labels)) {
        const colors = t.labels.map((lab, i) => barCategoryColors[String(lab)] || traceColors[i] || colorway[i % colorway.length]);
        t.marker = { ...(t.marker || {}), colors };
      }
      t.opacity = opacity;
      return t;
    }
    if (t.type === 'heatmap' || t.type === 'contour') {
      t.opacity = opacity;
      return t;
    }
    return t;
  });
}

const __oldPolishLayoutV15 = typeof polishLayoutForPublication === 'function' ? polishLayoutForPublication : null;
function polishLayoutForPublication(layout, chartType, theme) {
  let l = __oldPolishLayoutV15 ? __oldPolishLayoutV15(layout, chartType, theme) : { ...(layout || {}) };
  const tickFontSize = STATE.axisTickFontSize || 11;
  const titleFontSize = STATE.axisTitleFontSize || 13;
  const labelFontSize = STATE.labelFontSize || 12;
  const bgMode = STATE.chartBackgroundMode || 'grid';
  l.font = { ...(l.font || {}), size: labelFontSize, color: '#243649' };
  l.title = typeof l.title === 'string' ? { text: l.title } : { ...(l.title || {}) };
  if (l.title && !l.title.font) l.title.font = { size: Math.max(16, titleFontSize + 3), color: '#17324d' };
  const xShowLine = (l.xaxis && l.xaxis.showline === false) ? false : true;
  const yShowLine = (l.yaxis && l.yaxis.showline === false) ? false : true;
  l.xaxis = { ...(l.xaxis || {}), showline: xShowLine, linecolor: '#111827', linewidth: 1.6, tickfont: { ...((l.xaxis||{}).tickfont||{}), size: tickFontSize }, title: typeof (l.xaxis||{}).title === 'string' ? { text: l.xaxis.title, font: { size: titleFontSize } } : { ...((l.xaxis||{}).title||{}), font: { ...((((l.xaxis||{}).title)||{}).font||{}), size: titleFontSize } } };
  l.yaxis = { ...(l.yaxis || {}), showline: yShowLine, linecolor: '#111827', linewidth: 1.6, tickfont: { ...((l.yaxis||{}).tickfont||{}), size: tickFontSize }, title: typeof (l.yaxis||{}).title === 'string' ? { text: l.yaxis.title, font: { size: titleFontSize } } : { ...((l.yaxis||{}).title||{}), font: { ...((((l.yaxis||{}).title)||{}).font||{}), size: titleFontSize } } };
  if (l.polar) {
    l.polar = {
      ...(l.polar || {}),
      angularaxis: { ...((l.polar||{}).angularaxis || {}), tickfont: { ...((((l.polar||{}).angularaxis)||{}).tickfont || {}), size: tickFontSize } },
      radialaxis: { ...((l.polar||{}).radialaxis || {}), tickfont: { ...((((l.polar||{}).radialaxis)||{}).tickfont || {}), size: tickFontSize } },
    };
  }
  if (bgMode === 'transparent') {
    l.paper_bgcolor = 'rgba(0,0,0,0)';
    l.plot_bgcolor = 'rgba(0,0,0,0)';
    l.xaxis.showgrid = false;
    l.yaxis.showgrid = false;
  } else if (bgMode === 'blank') {
    l.paper_bgcolor = '#ffffff';
    l.plot_bgcolor = '#ffffff';
    l.xaxis.showgrid = false;
    l.yaxis.showgrid = false;
  } else {
    l.paper_bgcolor = '#ffffff';
    l.plot_bgcolor = '#ffffff';
    l.xaxis.gridcolor = '#e9eff6';
    l.yaxis.gridcolor = '#e9eff6';
    l.xaxis.showgrid = true;
    l.yaxis.showgrid = true;
  }
  const ann = Array.isArray(l.annotations) ? l.annotations.filter(a => !(a && a.text === '' && a.showarrow === true && a.xref === 'paper' && a.yref === 'paper' && (a.x > 1 || a.y > 1))) : [];
  l.annotations = [
    ...ann,
    { x: 1.018, y: 0, ax: 0.965, ay: 0, xref: 'paper', yref: 'paper', axref: 'paper', ayref: 'paper', showarrow: true, arrowhead: 3, arrowsize: 1.15, arrowwidth: 1.6, arrowcolor: '#111827', text: '' },
    { x: 0, y: 1.018, ax: 0, ay: 0.965, xref: 'paper', yref: 'paper', axref: 'paper', ayref: 'paper', showarrow: true, arrowhead: 3, arrowsize: 1.15, arrowwidth: 1.6, arrowcolor: '#111827', text: '' },
  ];
  return l;
}


/* ============================================================
   v18 final no-fail rendering + top chart tabs
   - chart switcher fixed at top, outside plot canvas
   - final renderer does NOT chain old risky polish functions
   - Plotly failure falls back to lightweight SVG, never "render failed"
   ============================================================ */
function deepCloneV18(obj) {
  try { return JSON.parse(JSON.stringify(obj)); } catch (_) { return obj; }
}

function arrayifyV18(v) {
  if (!v) return [];
  if (Array.isArray(v)) return v;
  if (typeof v === 'object') {
    // If a backend ever returns a typed-array-like object, degrade safely.
    if (Array.isArray(v.values)) return v.values;
    if (Array.isArray(v.data)) return v.data;
  }
  return [v];
}

function cleanTraceV18(trace, idx, palette) {
  const t = deepCloneV18(trace || {});
  t.type = t.type || 'scatter';
  if (t.x) t.x = arrayifyV18(t.x);
  if (t.y) t.y = arrayifyV18(t.y);
  if (t.labels) t.labels = arrayifyV18(t.labels);
  if (t.values) t.values = arrayifyV18(t.values);
  if (t.text && typeof t.text === 'object') t.text = arrayifyV18(t.text);
  if (t.customdata && typeof t.customdata === 'object' && !Array.isArray(t.customdata)) delete t.customdata;

  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  const traceColors = (STATE.userTraceColorsByChart || {})[activeIdx] || [];
  const barColors = (STATE.userBarCategoryColorsByChart || {})[activeIdx] || {};
  const baseColor = traceColors[idx] || (t.line && t.line.color) || (t.marker && typeof t.marker.color === 'string' ? t.marker.color : (palette[idx % palette.length] || '#2563eb'));
  const opacity = STATE.markerOpacity != null ? Number(STATE.markerOpacity) : 0.88;

  if (t.type === 'bar' || t.type === 'histogram') {
    const cats = (Array.isArray(t.x) && t.x.length ? t.x : (Array.isArray(t.y) ? t.y : [])).map(x => String(x));
    t.width = STATE.barWidth != null ? Number(STATE.barWidth) : 0.68;
    t.marker = {
      ...(t.marker || {}),
      color: cats.length ? cats.map(c => barColors[c] || baseColor) : baseColor,
      opacity,
      line: { color: '#ffffff', width: 0.8 },
    };
  } else if (t.type === 'scatter') {
    const mode = String(t.mode || 'markers');
    if (mode.includes('lines')) t.line = { ...(t.line || {}), color: baseColor, width: STATE.lineWidth != null ? Number(STATE.lineWidth) : 2.5 };
    if (mode.includes('markers') || !mode.includes('lines')) t.marker = { ...(t.marker || {}), color: baseColor, size: STATE.markerSize != null ? Number(STATE.markerSize) : 8, opacity, line: { color: '#ffffff', width: 0.6 } };
  } else if (t.type === 'box' || t.type === 'violin') {
    t.line = { ...(t.line || {}), color: baseColor, width: STATE.lineWidth != null ? Number(STATE.lineWidth) : 2.2 };
    t.marker = { ...(t.marker || {}), color: baseColor, size: STATE.markerSize != null ? Number(STATE.markerSize) : 7, opacity };
    t.opacity = opacity;
  } else if (t.type === 'pie') {
    const labels = arrayifyV18(t.labels).map(String);
    const colors = labels.map((lab, i) => barColors[lab] || traceColors[i] || palette[i % palette.length] || '#2563eb');
    t.marker = { ...(t.marker || {}), colors };
    t.opacity = opacity;
  } else if (t.type === 'heatmap' || t.type === 'contour') {
    t.opacity = opacity;
    // Avoid old Plotly incompatibility with texttemplate in heatmap.
    if (t.texttemplate && !window.Plotly?.PlotSchema) {
      delete t.texttemplate;
    }
  }
  return t;
}

function cleanLayoutV18(layout, theme, frameSize) {
  const l = deepCloneV18(layout || {});
  const tickSize = STATE.axisTickFontSize || 11;
  const titleSize = STATE.axisTitleFontSize || 13;
  const labelSize = STATE.labelFontSize || 12;
  const bgMode = STATE.chartBackgroundMode || 'grid';
  const ink = '#172033';
  const gridOn = bgMode === 'grid';

  if (STATE.chartTitle) {
    l.title = { text: STATE.chartTitle };
  } else if (typeof l.title === 'string') {
    l.title = { text: l.title };
  } else {
    l.title = { ...(l.title || {}) };
  }
  l.title.font = { ...((l.title && l.title.font) || {}), size: Math.max(16, titleSize + 3), color: ink };
  l.font = { ...(l.font || {}), size: labelSize, color: ink };
  l.paper_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
  l.plot_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
  l.margin = { l: 82, r: 46, t: 78, b: 82, ...(l.margin || {}) };
  l.width = frameSize.width;
  l.height = frameSize.height;
  l.autosize = false;
  l.bargap = l.bargap ?? 0.18;

  const patchAxis = (axis) => {
    const a = { ...(axis || {}) };
    if (typeof a.title === 'string') a.title = { text: a.title };
    else a.title = { ...(a.title || {}) };
    a.title.font = { ...((a.title && a.title.font) || {}), size: titleSize, color: ink };
    a.tickfont = { ...(a.tickfont || {}), size: tickSize, color: ink };
    if (a.showline !== false) a.showline = true;
    a.linecolor = '#111827';
    a.linewidth = 1.6;
    a.mirror = false;
    a.showgrid = gridOn;
    a.gridcolor = '#e9eff6';
    a.zeroline = false;
    return a;
  };
  l.xaxis = patchAxis(l.xaxis);
  l.yaxis = patchAxis(l.yaxis);
  for (const k of Object.keys(l)) {
    if (/^xaxis\d+$/.test(k) || /^yaxis\d+$/.test(k)) l[k] = patchAxis(l[k]);
  }

  // Remove any risky old custom annotation objects and add safe pixel-arrow axes.
  const oldAnn = Array.isArray(l.annotations)
    ? l.annotations.filter(a => !(a && a.showarrow && a.text === '' && a.xref === 'paper' && a.yref === 'paper'))
    : [];
  const isCartesian = !l.polar && !String(l.title?.text || '').includes('热图');
  l.annotations = isCartesian ? [
    ...oldAnn,
    { x: 1.015, y: 0, xref: 'paper', yref: 'paper', ax: -34, ay: 0, showarrow: true, arrowhead: 3, arrowsize: 1.1, arrowwidth: 1.5, arrowcolor: '#111827', text: '' },
    { x: 0, y: 1.015, xref: 'paper', yref: 'paper', ax: 0, ay: 34, showarrow: true, arrowhead: 3, arrowsize: 1.1, arrowwidth: 1.5, arrowcolor: '#111827', text: '' },
  ] : oldAnn;
  return l;
}

function renderSvgFallbackV18(plotMount, traces, layout) {
  const first = traces.find(t => (t.x && t.x.length) || (t.y && t.y.length)) || traces[0] || {};
  const title = (layout && layout.title && layout.title.text) || first.name || '图形预览';
  const W = 760, H = 480, L = 76, R = 28, T = 52, B = 64;
  let html = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" role="img" aria-label="${title}">
    <rect x="0" y="0" width="${W}" height="${H}" fill="#fff"/>
    <text x="${L}" y="28" font-size="18" font-weight="700" fill="#172033">${String(title).replace(/[<>&]/g, '')}</text>
    <line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#111827" stroke-width="2"/>
    <line x1="${L}" y1="${H-B}" x2="${L}" y2="${T}" stroke="#111827" stroke-width="2"/>
    <path d="M ${W-R} ${H-B} l -8 -5 l 0 10 z" fill="#111827"/>
    <path d="M ${L} ${T} l -5 8 l 10 0 z" fill="#111827"/>`;
  const x = arrayifyV18(first.x);
  const y = arrayifyV18(first.y).map(Number).filter(Number.isFinite);
  const color = '#2563eb';
  if ((first.type === 'bar' || first.type === 'histogram') && x.length && y.length) {
    const minY = Math.min(0, ...y), maxY = Math.max(...y, 1);
    const bw = (W - L - R) / Math.max(1, y.length) * 0.62;
    y.forEach((v, i) => {
      const x0 = L + (i + 0.2) * ((W - L - R) / y.length);
      const y0 = T + (maxY - v) / (maxY - minY || 1) * (H - T - B);
      const zero = T + (maxY - 0) / (maxY - minY || 1) * (H - T - B);
      const h = Math.abs(zero - y0);
      html += `<rect x="${x0}" y="${Math.min(y0,zero)}" width="${bw}" height="${h}" fill="${v>=0?color:'#ef4444'}" opacity="0.86"/>`;
      if (i < 10) html += `<text x="${x0+bw/2}" y="${H-B+18}" font-size="10" text-anchor="middle" fill="#334155">${String(x[i]).slice(0,10)}</text>`;
    });
  } else if (x.length && y.length) {
    const xs = x.map((_, i) => L + i * (W - L - R) / Math.max(1, x.length - 1));
    const minY = Math.min(...y), maxY = Math.max(...y);
    const pts = y.map((v, i) => `${xs[i]},${T + (maxY - v) / (maxY - minY || 1) * (H - T - B)}`);
    html += `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="3"/>`;
    pts.forEach(p => { const [cx,cy] = p.split(','); html += `<circle cx="${cx}" cy="${cy}" r="4" fill="${color}" opacity=".85"/>`; });
  } else {
    html += `<text x="${W/2}" y="${H/2}" font-size="16" text-anchor="middle" fill="#64748b">图形已返回，但当前浏览器无法使用 Plotly 渲染，已启用安全预览。</text>`;
  }
  html += `</svg>`;
  plotMount.innerHTML = html;
}

function getChartFrameSize(plotMount) {
  const parent = plotMount?.parentElement || plotMount;
  const maxW = Math.max(420, Math.floor((parent?.clientWidth || 900) * 0.86));
  const maxH = Math.max(320, Math.floor((parent?.clientHeight || 560) * 0.82));
  const aspect = 1.58;
  let height = maxH;
  let width = Math.floor(height * aspect);
  if (width > maxW) { width = maxW; height = Math.floor(width / aspect); }
  return { width: Math.max(420, width), height: Math.max(320, height) };
}

function fitChartPlotToFrame(plotMount) {
  const s = getChartFrameSize(plotMount);
  plotMount.style.setProperty('width', `${s.width}px`, 'important');
  plotMount.style.setProperty('height', `${s.height}px`, 'important');
  plotMount.style.setProperty('min-height', `${s.height}px`, 'important');
  plotMount.style.setProperty('margin', 'auto', 'important');
  return s;
}

function renderCenteredChart(plotMount, plotlyJson, palette, theme, options = {}) {
  if (!plotMount) return;
  const safePalette = palette && palette.length ? palette : ['#2563eb', '#ef4444', '#0f766e', '#eab308', '#7c3aed'];
  try {
    const raw = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
    const data = Array.isArray(raw?.data) ? raw.data : (raw?.data ? [raw.data] : []);
    if (!data.length) {
      plotMount.innerHTML = '<div class="empty-state small">当前方法没有返回可视化数据</div>';
      return;
    }
    let traces = data.map((t, i) => cleanTraceV18(t, i, safePalette));
    const frame = fitChartPlotToFrame(plotMount);
    let layout = cleanLayoutV18(raw.layout || {}, theme || {}, frame);
    if (layout.showlegend === undefined) layout.showlegend = traces.some(t => t.name && t.showlegend !== false);

    const trackState = options.trackState !== false;
    if (trackState) {
      STATE.currentPlotlyData = traces;
      STATE.currentPlotlyLayout = layout;
    }

    if (!window.Plotly || !Plotly.newPlot) {
      renderSvgFallbackV18(plotMount, traces, layout);
      return;
    }

    Plotly.newPlot(plotMount, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
    }).then(() => {
      if (trackState) {
        const entry = {
          el: plotMount,
          index: Number.isFinite(options.index) ? options.index : 0,
          role: options.role || 'primary',
          title: options.title || (layout.title && layout.title.text) || 'Chart',
          data: traces,
          layout,
        };
        STATE.currentRenderedPlots = [entry];
      }
      try { syncCurrentPlotlyLayoutFromDom(plotMount, trackState); } catch (_) {}
      try { installChartResizeObserver(plotMount, { trackState }); } catch (_) {}
    }).catch(err => {
      console.error('Plotly render failed; using SVG fallback:', err);
      renderSvgFallbackV18(plotMount, traces, layout);
    });
  } catch (e) {
    console.error('Chart render error; using SVG fallback:', e);
    try {
      renderSvgFallbackV18(plotMount, [], { title: { text: '安全图形预览' } });
    } catch (_) {
      plotMount.innerHTML = '<div class="empty-state small">图形已生成，但当前浏览器无法渲染；请切换其他图形或下载结果。</div>';
    }
  }
}

function renderChartSwitcherV14(charts) {
  const switcher = el('chartVariantTabs');
  if (!switcher) return;
  const list = charts || [];
  if (list.length <= 1) {
    switcher.innerHTML = '';
    switcher.style.display = 'none';
    return;
  }
  switcher.style.display = 'flex';
  switcher.innerHTML = list.map((c, i) => `
    <button type="button" class="chart-variant-tab ${i === (STATE.activeChartIndex || 0) ? 'active' : ''}" data-chart-index="${i}">
      ${escapeHtml(c.title || ('图 ' + (i + 1)))}
    </button>
  `).join('');
}

function renderAllCharts(charts) {
  const container = el('chartPreviewContainer');
  if (!container) return;
  disconnectResizeObserver();
  purgePlotlyChildren(container);
  STATE.currentRenderedPlots = [];
  STATE.currentChartBundle = charts || [];
  STATE.activeChartIndex = Math.min(STATE.activeChartIndex || 0, Math.max(0, STATE.currentChartBundle.length - 1));
  container.classList.remove('chart-gallery');
  container.innerHTML = `<div id="chartActivePlot" class="chart-active-plot"></div>`;
  renderChartSwitcherV14(STATE.currentChartBundle);
  renderActiveChartV14();
}

function renderActiveChartV14() {
  const container = el('chartActivePlot');
  if (!container) return;
  const chart = getActiveChartDataV14();
  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2563eb', '#ef4444', '#0f766e', '#eab308'];
  purgePlotlyChildren(container);
  container.innerHTML = '';
  if (!chart || !chart.plotly) {
    container.innerHTML = '<div class="empty-state small">暂无可显示图形</div>';
    return;
  }
  const plotMount = document.createElement('div');
  plotMount.className = 'chart-plot';
  container.appendChild(plotMount);
  renderCenteredChart(plotMount, chart.plotly, palette, theme, {
    index: STATE.activeChartIndex || 0,
    title: chart.title || 'Chart',
    role: 'active',
  });
  if (typeof renderAppearanceControls === 'function') setTimeout(renderAppearanceControls, 80);
}



/* ============================================================
   v19 final axis rebuild + bar-to-axis alignment
   - no chaining to any previous axis functions
   - categorical bars are converted to numeric positions
   - custom axis lines and arrowheads are drawn in data coordinates
   - native axis lines are hidden to avoid duplicated "patched-on" arrows
   ============================================================ */

function safeNumV19(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function uniqueInOrderV19(arr) {
  const out = [];
  const seen = new Set();
  (arr || []).forEach(v => {
    const s = String(v);
    if (!seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  });
  return out;
}

function parsePlotlyV19(plotlyJson) {
  if (!plotlyJson) return { data: [], layout: {} };
  const raw = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
  const data = Array.isArray(raw?.data) ? raw.data : (raw?.data ? [raw.data] : []);
  return { data, layout: raw?.layout || {} };
}

function buildBarCategoryMapsV19(data) {
  let vertical = [];
  let horizontal = [];
  (data || []).forEach(t => {
    if (!t || t.type !== 'bar') return;
    const orient = t.orientation === 'h' ? 'h' : 'v';
    if (orient === 'h') {
      const y = arrayifyV18(t.y);
      if (y.some(v => safeNumV19(v) === null)) horizontal.push(...y.map(String));
    } else {
      const x = arrayifyV18(t.x);
      if (x.some(v => safeNumV19(v) === null)) vertical.push(...x.map(String));
    }
  });
  const vLabels = uniqueInOrderV19(vertical);
  const hLabels = uniqueInOrderV19(horizontal);
  return {
    verticalLabels: vLabels,
    horizontalLabels: hLabels,
    verticalMap: new Map(vLabels.map((v, i) => [v, i])),
    horizontalMap: new Map(hLabels.map((v, i) => [v, i])),
  };
}

function cleanTraceV19(trace, idx, palette, maps) {
  const t = deepCloneV18(trace || {});
  const type = t.type || 'scatter';
  t.type = type;
  const activeIdx = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
  const traceColors = (STATE.userTraceColorsByChart || {})[activeIdx] || [];
  const categoryColors = (STATE.userBarCategoryColorsByChart || {})[activeIdx] || {};
  const baseColor = traceColors[idx] || (t.line && t.line.color) || (t.marker && typeof t.marker.color === 'string' ? t.marker.color : (palette[idx % palette.length] || '#2563eb'));
  const opacity = STATE.markerOpacity != null ? Number(STATE.markerOpacity) : 0.88;

  if (t.x) t.x = arrayifyV18(t.x);
  if (t.y) t.y = arrayifyV18(t.y);
  if (t.labels) t.labels = arrayifyV18(t.labels);
  if (t.values) t.values = arrayifyV18(t.values);

  if (type === 'bar') {
    const isH = t.orientation === 'h';
    const labels = isH ? (t.y || []).map(String) : (t.x || []).map(String);
    const map = isH ? maps.horizontalMap : maps.verticalMap;
    const hasCategory = labels.some(v => map.has(v));
    if (hasCategory) {
      if (isH) {
        t.y = labels.map(v => map.get(v));
        t.customdata = labels;
        t.hovertemplate = '变量: %{customdata}<br>值: %{x:.4f}<extra></extra>';
      } else {
        t.x = labels.map(v => map.get(v));
        t.customdata = labels;
        t.hovertemplate = '变量: %{customdata}<br>值: %{y:.4f}<extra></extra>';
      }
    }
    const cats = labels.length ? labels : ((isH ? t.y : t.x) || []).map(String);
    t.width = STATE.barWidth != null ? Number(STATE.barWidth) : 0.62;
    t.marker = {
      ...(t.marker || {}),
      color: cats.length ? cats.map(c => categoryColors[c] || baseColor) : baseColor,
      opacity,
      line: { color: '#ffffff', width: 0.8 },
    };
    return t;
  }

  if (type === 'histogram') {
    t.marker = { ...(t.marker || {}), color: baseColor, opacity, line: { color: '#ffffff', width: 0.8 } };
    return t;
  }

  if (type === 'scatter') {
    const mode = String(t.mode || 'markers');
    if (mode.includes('lines')) {
      t.line = { ...(t.line || {}), color: baseColor, width: STATE.lineWidth != null ? Number(STATE.lineWidth) : 2.5 };
    }
    if (mode.includes('markers') || !mode.includes('lines')) {
      t.marker = { ...(t.marker || {}), color: baseColor, size: STATE.markerSize != null ? Number(STATE.markerSize) : 8, opacity, line: { color: '#ffffff', width: 0.6 } };
    }
    return t;
  }

  if (type === 'box' || type === 'violin') {
    t.line = { ...(t.line || {}), color: baseColor, width: STATE.lineWidth != null ? Number(STATE.lineWidth) : 2.2 };
    t.marker = { ...(t.marker || {}), color: baseColor, size: STATE.markerSize != null ? Number(STATE.markerSize) : 7, opacity };
    t.opacity = opacity;
    return t;
  }

  if (type === 'pie') {
    const labels = arrayifyV18(t.labels).map(String);
    t.marker = { ...(t.marker || {}), colors: labels.map((lab, i) => categoryColors[lab] || traceColors[i] || palette[i % palette.length] || '#2563eb') };
    t.opacity = opacity;
    return t;
  }

  if (type === 'heatmap' || type === 'contour') {
    t.opacity = opacity;
    delete t.texttemplate;
    return t;
  }

  return t;
}

function collectNumericValuesV19(traces, key) {
  const vals = [];
  (traces || []).forEach(t => {
    arrayifyV18(t[key]).forEach(v => {
      const n = safeNumV19(v);
      if (n !== null) vals.push(n);
    });
  });
  return vals;
}

function paddedRangeV19(vals, includeZero = false) {
  let arr = (vals || []).filter(Number.isFinite);
  if (!arr.length) arr = [0, 1];
  if (includeZero) arr.push(0);
  let min = Math.min(...arr), max = Math.max(...arr);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function buildLayoutV19(rawLayout, traces, theme, frameSize, maps) {
  const l = deepCloneV18(rawLayout || {});
  const tickSize = STATE.axisTickFontSize || 11;
  const axisTitleSize = STATE.axisTitleFontSize || 13;
  const labelSize = STATE.labelFontSize || 12;
  const bgMode = STATE.chartBackgroundMode || 'grid';
  const gridOn = bgMode === 'grid';
  const ink = '#111827';

  if (STATE.chartTitle) {
    l.title = { text: STATE.chartTitle };
  } else if (typeof l.title === 'string') {
    l.title = { text: l.title };
  } else {
    l.title = { ...(l.title || {}) };
  }
  l.title.font = { ...((l.title && l.title.font) || {}), size: Math.max(16, axisTitleSize + 3), color: '#172033' };

  l.font = { ...(l.font || {}), size: labelSize, color: '#243649' };
  l.paper_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
  l.plot_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
  l.margin = { l: 72, r: 34, t: 74, b: 74, ...(l.margin || {}) };
  l.width = frameSize.width;
  l.height = frameSize.height;
  l.autosize = false;
  l.bargap = l.bargap ?? 0.18;

  const hasVerticalCats = maps.verticalLabels.length > 0;
  const hasHorizontalCats = maps.horizontalLabels.length > 0;
  const hasPie = traces.some(t => t.type === 'pie');
  const hasHeat = traces.some(t => t.type === 'heatmap' || t.type === 'contour');
  const hasPolar = !!l.polar;
  const isCartesian = !hasPie && !hasHeat && !hasPolar;

  let xVals = collectNumericValuesV19(traces, 'x');
  let yVals = collectNumericValuesV19(traces, 'y');

  if (hasVerticalCats) xVals = maps.verticalLabels.map((_, i) => i);
  if (hasHorizontalCats) yVals = maps.horizontalLabels.map((_, i) => i);

  const verticalBars = traces.some(t => t.type === 'bar' && t.orientation !== 'h');
  const horizontalBars = traces.some(t => t.type === 'bar' && t.orientation === 'h');

  let xRange = hasVerticalCats ? [-0.62, Math.max(0, maps.verticalLabels.length - 1) + 0.62] : paddedRangeV19(xVals, horizontalBars);
  let yRange = hasHorizontalCats ? [-0.62, Math.max(0, maps.horizontalLabels.length - 1) + 0.62] : paddedRangeV19(yVals, verticalBars);

  const xAxisAt = xRange[0];
  const yAxisAt = (yRange[0] <= 0 && yRange[1] >= 0) ? 0 : yRange[0];
  const yAxisX = (xRange[0] <= 0 && xRange[1] >= 0 && horizontalBars) ? 0 : xRange[0];

  const patchAxis = (axis, titleText) => {
    const a = { ...(axis || {}) };
    if (typeof a.title === 'string') a.title = { text: a.title };
    else a.title = { ...(a.title || {}) };
    if (!a.title.text && titleText) a.title.text = titleText;
    a.title.font = { ...((a.title && a.title.font) || {}), size: axisTitleSize, color: ink };
    a.tickfont = { ...(a.tickfont || {}), size: tickSize, color: ink };
    a.showline = false;
    a.mirror = false;
    a.zeroline = false;
    a.showgrid = gridOn;
    a.gridcolor = '#e9eff6';
    a.gridwidth = 1;
    a.ticks = 'outside';
    a.ticklen = 5;
    a.tickcolor = ink;
    return a;
  };

  l.xaxis = patchAxis(l.xaxis, '');
  l.yaxis = patchAxis(l.yaxis, '');

  l.xaxis.range = xRange;
  l.yaxis.range = yRange;

  if (hasVerticalCats) {
    l.xaxis.type = 'linear';
    l.xaxis.tickmode = 'array';
    l.xaxis.tickvals = maps.verticalLabels.map((_, i) => i);
    l.xaxis.ticktext = maps.verticalLabels;
  }
  if (hasHorizontalCats) {
    l.yaxis.type = 'linear';
    l.yaxis.tickmode = 'array';
    l.yaxis.tickvals = maps.horizontalLabels.map((_, i) => i);
    l.yaxis.ticktext = maps.horizontalLabels;
  }

  for (const k of Object.keys(l)) {
    if (/^xaxis\d+$/.test(k) || /^yaxis\d+$/.test(k)) l[k] = patchAxis(l[k], '');
  }

  // Remove all old axis-like shapes and annotations; rebuild once.
  const oldShapes = Array.isArray(l.shapes) ? l.shapes.filter(s => s && s.name !== 'custom_arrow_axis_v19') : [];
  const oldAnn = Array.isArray(l.annotations) ? l.annotations.filter(a => !(a && a.name === 'custom_arrow_axis_v19')) : [];
  l.shapes = oldShapes;
  l.annotations = oldAnn;

  if (isCartesian) {
    l.shapes.push(
      {
        name: 'custom_arrow_axis_v19',
        type: 'line',
        xref: 'x',
        yref: 'y',
        x0: xRange[0],
        x1: xRange[1],
        y0: yAxisAt,
        y1: yAxisAt,
        line: { color: ink, width: 2 },
        layer: 'above',
      },
      {
        name: 'custom_arrow_axis_v19',
        type: 'line',
        xref: 'x',
        yref: 'y',
        x0: yAxisX,
        x1: yAxisX,
        y0: yRange[0],
        y1: yRange[1],
        line: { color: ink, width: 2 },
        layer: 'above',
      },
    );
    l.annotations.push(
      {
        name: 'custom_arrow_axis_v19',
        x: xRange[1],
        y: yAxisAt,
        xref: 'x',
        yref: 'y',
        ax: -34,
        ay: 0,
        showarrow: true,
        arrowhead: 3,
        arrowsize: 1.25,
        arrowwidth: 2,
        arrowcolor: ink,
        text: '',
      },
      {
        name: 'custom_arrow_axis_v19',
        x: yAxisX,
        y: yRange[1],
        xref: 'x',
        yref: 'y',
        ax: 0,
        ay: 34,
        showarrow: true,
        arrowhead: 3,
        arrowsize: 1.25,
        arrowwidth: 2,
        arrowcolor: ink,
        text: '',
      },
    );
  }

  return l;
}

function renderCenteredChart(plotMount, plotlyJson, palette, theme, options = {}) {
  if (!plotMount) return;
  const safePalette = palette && palette.length ? palette : ['#2563eb', '#ef4444', '#0f766e', '#eab308', '#7c3aed'];
  try {
    const parsed = parsePlotlyV19(plotlyJson);
    if (!parsed.data.length) {
      plotMount.innerHTML = '<div class="empty-state small">当前方法没有返回可视化数据</div>';
      return;
    }
    const maps = buildBarCategoryMapsV19(parsed.data);
    const traces = parsed.data.map((t, i) => cleanTraceV19(t, i, safePalette, maps));
    const frame = fitChartPlotToFrame(plotMount);
    const layout = buildLayoutV19(parsed.layout || {}, traces, theme || {}, frame, maps);
    if (layout.showlegend === undefined) {
      layout.showlegend = traces.some(t => t.name && t.showlegend !== false) && !traces.some(t => t.type === 'bar' && (maps.verticalLabels.length || maps.horizontalLabels.length));
    }

    const trackState = options.trackState !== false;
    if (trackState) {
      STATE.currentPlotlyData = traces;
      STATE.currentPlotlyLayout = layout;
    }

    if (!window.Plotly || !Plotly.newPlot) {
      renderSvgFallbackV18(plotMount, traces, layout);
      return;
    }

    Plotly.newPlot(plotMount, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
    }).then(() => {
      if (trackState) {
        STATE.currentRenderedPlots = [{
          el: plotMount,
          index: Number.isFinite(options.index) ? options.index : 0,
          role: options.role || 'primary',
          title: options.title || (layout.title && layout.title.text) || 'Chart',
          data: traces,
          layout,
        }];
      }
      try { syncCurrentPlotlyLayoutFromDom(plotMount, trackState); } catch (_) {}
      try { installChartResizeObserver(plotMount, { trackState }); } catch (_) {}
    }).catch(err => {
      console.error('Plotly render failed; using SVG fallback:', err);
      renderSvgFallbackV18(plotMount, traces, layout);
    });
  } catch (e) {
    console.error('Chart render error; using SVG fallback:', e);
    try {
      const parsed = parsePlotlyV19(plotlyJson);
      renderSvgFallbackV18(plotMount, parsed.data || [], parsed.layout || { title: { text: '安全图形预览' } });
    } catch (_) {
      plotMount.innerHTML = '<div class="empty-state small">图形已生成，但当前浏览器无法渲染；请切换其他图形或下载结果。</div>';
    }
  }
}

function renderChartSwitcherV14(charts) {
  let switcher = document.querySelector('.result-chart-tabs-fixed') || el('chartVariantTabs');
  if (!switcher) return;
  // If an old script injected the switcher into the plot area, move it back to the top.
  const resultView = el('result-view-chart');
  if (resultView && switcher.parentElement !== resultView) {
    resultView.insertBefore(switcher, resultView.firstElementChild);
  }
  const list = charts || [];
  if (list.length <= 1) {
    switcher.innerHTML = '';
    switcher.style.display = 'none';
    return;
  }
  switcher.id = 'chartVariantTabs';
  switcher.classList.add('chart-variant-tabs', 'result-chart-tabs-fixed');
  switcher.style.display = 'flex';
  switcher.innerHTML = list.map((c, i) => `
    <button type="button" class="chart-variant-tab ${i === (STATE.activeChartIndex || 0) ? 'active' : ''}" data-chart-index="${i}">
      ${escapeHtml(c.title || ('图 ' + (i + 1)))}
    </button>
  `).join('');
}

function renderAllCharts(charts) {
  const container = el('chartPreviewContainer');
  if (!container) return;
  disconnectResizeObserver();
  purgePlotlyChildren(container);
  STATE.currentRenderedPlots = [];
  STATE.currentChartBundle = charts || [];
  STATE.activeChartIndex = Math.min(STATE.activeChartIndex || 0, Math.max(0, STATE.currentChartBundle.length - 1));
  container.classList.remove('chart-gallery');
  container.innerHTML = '<div id="chartActivePlot" class="chart-active-plot"></div>';
  renderChartSwitcherV14(STATE.currentChartBundle);
  renderActiveChartV14();
}




/* ============================================================
   v20 FINAL chart workspace + standard arrow axes
   - result chart tabs only belong to visualization view
   - chart canvas is centered and width is data-aware
   - all old custom axis arrows are neutralized
   - brand-new standard arrow axes are drawn with line+triangle shapes
   ============================================================ */

function preferredChartFrameV20(plotMount, parsed) {
  const host = plotMount.closest('.chart-preview-main') || plotMount.parentElement || plotMount;
  const hostRect = host.getBoundingClientRect ? host.getBoundingClientRect() : { width: 1200, height: 780 };
  const maxW = Math.max(640, (hostRect.width || 1200) - 28);
  const maxH = Math.max(420, (hostRect.height || 760) - 28);
  const data = Array.isArray(parsed?.data) ? parsed.data : [];
  let cats = 0;
  let isBar = false;
  data.forEach(t => {
    if (!t || t.type !== 'bar') return;
    isBar = true;
    const arr = Array.isArray(t.x) && t.x.length ? t.x : (Array.isArray(t.y) ? t.y : []);
    cats = Math.max(cats, arr.length || 0);
  });
  let width = Math.min(maxW, 1180);
  if (isBar && cats > 0) width = Math.min(maxW, Math.max(760, 150 * cats + 180));
  const height = Math.min(maxH, 720);
  plotMount.style.setProperty('width', `${width}px`, 'important');
  plotMount.style.setProperty('height', `${height}px`, 'important');
  plotMount.style.setProperty('min-height', `${height}px`, 'important');
  plotMount.style.setProperty('max-width', '100%', 'important');
  plotMount.style.setProperty('max-height', '100%', 'important');
  plotMount.style.setProperty('margin', '0 auto', 'important');
  return { width, height };
}

function axisTrianglePathV20(tipX, tipY, base1X, base1Y, base2X, base2Y) {
  return `M ${tipX},${tipY} L ${base1X},${base1Y} L ${base2X},${base2Y} Z`;
}

function buildLayoutV20(rawLayout, traces, theme, frameSize, maps) {
  const l = deepCloneV18(rawLayout || {});
  const tickSize = STATE.axisTickFontSize || 11;
  const axisTitleSize = STATE.axisTitleFontSize || 13;
  const labelSize = STATE.labelFontSize || 12;
  const bgMode = STATE.chartBackgroundMode || 'grid';
  const gridOn = bgMode === 'grid';
  const ink = '#162033';

  if (STATE.chartTitle) l.title = { text: STATE.chartTitle };
  else if (typeof l.title === 'string') l.title = { text: l.title };
  else l.title = { ...(l.title || {}) };
  l.title.font = { ...((l.title && l.title.font) || {}), size: Math.max(16, axisTitleSize + 3), color: '#1f2b3d' };

  l.font = { ...(l.font || {}), size: labelSize, color: '#243649' };
  l.paper_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
  l.plot_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
  l.width = frameSize.width;
  l.height = frameSize.height;
  l.autosize = false;
  l.margin = { l: 82, r: 28, t: 72, b: 84, ...(l.margin || {}) };
  l.bargap = l.bargap ?? 0.16;

  const hasPie = traces.some(t => t.type === 'pie');
  const hasHeat = traces.some(t => t.type === 'heatmap' || t.type === 'contour');
  const hasPolar = !!l.polar;
  const hasVerticalCats = maps.verticalLabels.length > 0;
  const hasHorizontalCats = maps.horizontalLabels.length > 0;
  const isCartesian = !hasPie && !hasHeat && !hasPolar;

  let xVals = collectNumericValuesV19(traces, 'x');
  let yVals = collectNumericValuesV19(traces, 'y');
  if (hasVerticalCats) xVals = maps.verticalLabels.map((_, i) => i);
  if (hasHorizontalCats) yVals = maps.horizontalLabels.map((_, i) => i);

  const verticalBars = traces.some(t => t.type === 'bar' && t.orientation !== 'h');
  const horizontalBars = traces.some(t => t.type === 'bar' && t.orientation === 'h');

  let xRange = hasVerticalCats ? [-0.55, Math.max(0, maps.verticalLabels.length - 1) + 0.55] : paddedRangeV19(xVals, horizontalBars);
  let yRange = hasHorizontalCats ? [-0.55, Math.max(0, maps.horizontalLabels.length - 1) + 0.55] : paddedRangeV19(yVals, verticalBars);

  const patchAxis = (axis, titleText) => {
    const a = { ...(axis || {}) };
    if (typeof a.title === 'string') a.title = { text: a.title };
    else a.title = { ...(a.title || {}) };
    if (!a.title.text && titleText) a.title.text = titleText;
    a.title.font = { ...((a.title && a.title.font) || {}), size: axisTitleSize, color: ink };
    a.tickfont = { ...(a.tickfont || {}), size: tickSize, color: ink };
    a.showline = false;
    a.mirror = false;
    a.zeroline = false;
    a.ticks = 'outside';
    a.ticklen = 5;
    a.tickcolor = ink;
    a.showgrid = gridOn;
    a.gridcolor = '#e9eff6';
    a.gridwidth = 1;
    return a;
  };

  l.xaxis = patchAxis(l.xaxis, '');
  l.yaxis = patchAxis(l.yaxis, '');
  l.xaxis.range = xRange;
  l.yaxis.range = yRange;
  if (hasVerticalCats) {
    l.xaxis.type = 'linear';
    l.xaxis.tickmode = 'array';
    l.xaxis.tickvals = maps.verticalLabels.map((_, i) => i);
    l.xaxis.ticktext = maps.verticalLabels;
  }
  if (hasHorizontalCats) {
    l.yaxis.type = 'linear';
    l.yaxis.tickmode = 'array';
    l.yaxis.tickvals = maps.horizontalLabels.map((_, i) => i);
    l.yaxis.ticktext = maps.horizontalLabels;
  }
  for (const k of Object.keys(l)) {
    if (/^xaxis\d+$/.test(k) || /^yaxis\d+$/.test(k)) l[k] = patchAxis(l[k], '');
  }

  const stripAxisAnn = a => {
    if (!a) return false;
    if (a.name && String(a.name).includes('axis')) return false;
    if (a.text === '' && a.showarrow === true && (a.xref === 'paper' || a.yref === 'paper')) return false;
    return true;
  };
  const stripAxisShape = s => {
    if (!s) return false;
    if (s.name && String(s.name).includes('axis')) return false;
    return true;
  };
  l.annotations = Array.isArray(l.annotations) ? l.annotations.filter(stripAxisAnn) : [];
  l.shapes = Array.isArray(l.shapes) ? l.shapes.filter(stripAxisShape) : [];

  if (isCartesian) {
    const xAxisY = (yRange[0] <= 0 && yRange[1] >= 0) ? 0 : yRange[0];
    const yAxisX = (xRange[0] <= 0 && xRange[1] >= 0 && horizontalBars) ? 0 : xRange[0];
    const dx = Math.max((xRange[1] - xRange[0]) * 0.018, hasVerticalCats ? 0.12 : 0.0001);
    const dy = Math.max((yRange[1] - yRange[0]) * 0.018, hasHorizontalCats ? 0.12 : 0.0001);
    const xLineEnd = xRange[1] - dx * 0.9;
    const yLineEnd = yRange[1] - dy * 0.9;

    l.shapes.push(
      {
        name: 'v20_arrow_x_axis', type: 'line', xref: 'x', yref: 'y',
        x0: xRange[0], x1: xLineEnd, y0: xAxisY, y1: xAxisY,
        line: { color: ink, width: 2.2 }, layer: 'above'
      },
      {
        name: 'v20_arrow_y_axis', type: 'line', xref: 'x', yref: 'y',
        x0: yAxisX, x1: yAxisX, y0: yRange[0], y1: yLineEnd,
        line: { color: ink, width: 2.2 }, layer: 'above'
      },
      {
        name: 'v20_arrow_x_axis_head', type: 'path', xref: 'x', yref: 'y',
        path: axisTrianglePathV20(xRange[1], xAxisY, xRange[1] - dx, xAxisY + dy * 0.55, xRange[1] - dx, xAxisY - dy * 0.55),
        fillcolor: ink, line: { color: ink, width: 1 }, layer: 'above'
      },
      {
        name: 'v20_arrow_y_axis_head', type: 'path', xref: 'x', yref: 'y',
        path: axisTrianglePathV20(yAxisX, yRange[1], yAxisX - dx * 0.55, yRange[1] - dy, yAxisX + dx * 0.55, yRange[1] - dy),
        fillcolor: ink, line: { color: ink, width: 1 }, layer: 'above'
      }
    );
  }
  return l;
}

function renderCenteredChart(plotMount, plotlyJson, palette, theme, options = {}) {
  if (!plotMount) return;
  const safePalette = palette && palette.length ? palette : ['#2563eb', '#ef4444', '#0f766e', '#eab308', '#7c3aed'];
  try {
    const parsed = parsePlotlyV19(plotlyJson);
    if (!parsed.data.length) {
      plotMount.innerHTML = '<div class="empty-state small">当前方法没有返回可视化数据</div>';
      return;
    }
    const maps = buildBarCategoryMapsV19(parsed.data);
    const traces = parsed.data.map((t, i) => cleanTraceV19(t, i, safePalette, maps));
    const frame = preferredChartFrameV20(plotMount, parsed);
    const layout = buildLayoutV20(parsed.layout || {}, traces, theme || {}, frame, maps);
    if (layout.showlegend === undefined) {
      layout.showlegend = traces.some(t => t.name && t.showlegend !== false) && !traces.some(t => t.type === 'bar' && (maps.verticalLabels.length || maps.horizontalLabels.length));
    }
    const trackState = options.trackState !== false;
    if (trackState) {
      STATE.currentPlotlyData = traces;
      STATE.currentPlotlyLayout = layout;
    }
    if (!window.Plotly || !Plotly.newPlot) {
      renderSvgFallbackV18(plotMount, traces, layout);
      return;
    }
    Plotly.newPlot(plotMount, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
    }).then(() => {
      if (trackState) {
        STATE.currentRenderedPlots = [{
          el: plotMount,
          index: Number.isFinite(options.index) ? options.index : 0,
          role: options.role || 'primary',
          title: options.title || (layout.title && layout.title.text) || 'Chart',
          data: traces,
          layout,
        }];
      }
      try { syncCurrentPlotlyLayoutFromDom(plotMount, trackState); } catch (_) {}
      try { installChartResizeObserver(plotMount, { trackState }); } catch (_) {}
    }).catch(err => {
      console.error('Plotly render failed; using SVG fallback:', err);
      renderSvgFallbackV18(plotMount, traces, layout);
    });
  } catch (e) {
    console.error('Chart render error; using SVG fallback:', e);
    try {
      const parsed = parsePlotlyV19(plotlyJson);
      renderSvgFallbackV18(plotMount, parsed.data || [], parsed.layout || { title: { text: '安全图形预览' } });
    } catch (_) {
      plotMount.innerHTML = '<div class="empty-state small">图形已生成，但当前浏览器无法渲染；请切换其他图形或下载结果。</div>';
    }
  }
}

function renderChartSwitcherV20(charts) {
  let switcher = document.querySelector('#result-view-chart > .result-chart-tabs-fixed') || el('chartVariantTabs');
  const resultView = el('result-view-chart');
  if (!switcher || !resultView) return;
  if (switcher.parentElement !== resultView) resultView.insertBefore(switcher, resultView.firstElementChild);
  const list = charts || [];
  if (list.length <= 1) {
    switcher.innerHTML = '';
    switcher.style.display = 'none';
    return;
  }
  switcher.id = 'chartVariantTabs';
  switcher.className = 'chart-variant-tabs result-chart-tabs-fixed';
  switcher.style.display = 'flex';
  switcher.innerHTML = list.map((c, i) => `
    <button type="button" class="chart-variant-tab ${i === (STATE.activeChartIndex || 0) ? 'active' : ''}" data-chart-index="${i}">${escapeHtml(c.title || ('图 ' + (i + 1)))}</button>
  `).join('');
}

function renderAllCharts(charts) {
  const container = el('chartPreviewContainer');
  if (!container) return;
  disconnectResizeObserver();
  purgePlotlyChildren(container);
  STATE.currentRenderedPlots = [];
  STATE.currentChartBundle = charts || [];
  STATE.activeChartIndex = Math.min(STATE.activeChartIndex || 0, Math.max(0, STATE.currentChartBundle.length - 1));
  container.classList.remove('chart-gallery');
  container.innerHTML = '<div id="chartActivePlot" class="chart-active-plot"></div>';
  renderChartSwitcherV20(STATE.currentChartBundle);
  renderActiveChartV14();
}




/* ============================================================
   v21 final polish
   - smaller centered chart canvas
   - diagnostics unique vs visualization
   - cleaner arrow axis integration
   ============================================================ */
function preferredChartFrameV21(plotMount, parsed) {
  const host = plotMount.closest('.chart-preview-main') || plotMount.parentElement || plotMount;
  const rect = host.getBoundingClientRect ? host.getBoundingClientRect() : { width: 1200, height: 760 };
  const maxW = Math.max(620, (rect.width || 1200) - 96);
  const maxH = Math.max(420, (rect.height || 760) - 80);
  const data = Array.isArray(parsed?.data) ? parsed.data : [];
  const isBar = data.some(t => t && t.type === 'bar');
  let catCount = 0;
  data.forEach(t => {
    if (!t) return;
    const arr = Array.isArray(t.x) && t.x.length ? t.x : (Array.isArray(t.y) ? t.y : []);
    catCount = Math.max(catCount, arr.length || 0);
  });
  let width = Math.min(Math.floor(maxW * 0.72), 860);
  if (isBar && catCount > 0) width = Math.min(Math.floor(maxW * 0.78), Math.max(620, 92 * catCount + 150), 900);
  let height = Math.min(Math.floor(maxH * 0.72), 520);
  if (height < 400) height = Math.min(maxH, 400);
  plotMount.style.setProperty('width', `${width}px`, 'important');
  plotMount.style.setProperty('height', `${height}px`, 'important');
  plotMount.style.setProperty('min-height', `${height}px`, 'important');
  plotMount.style.setProperty('max-width', '100%', 'important');
  plotMount.style.setProperty('max-height', '100%', 'important');
  plotMount.style.setProperty('margin', '0 auto', 'important');
  return { width, height };
}

function buildLayoutV21(rawLayout, traces, theme, frameSize, maps) {
  const l = buildLayoutV20(rawLayout, traces, theme, frameSize, maps);
  const hasPie = traces.some(t => t.type === 'pie');
  const hasHeat = traces.some(t => t.type === 'heatmap' || t.type === 'contour');
  const hasPolar = !!l.polar;
  const isCartesian = !hasPie && !hasHeat && !hasPolar;
  if (!isCartesian) return l;

  const ink = '#162033';
  const hasVerticalCats = maps.verticalLabels.length > 0;
  const hasHorizontalCats = maps.horizontalLabels.length > 0;
  const verticalBars = traces.some(t => t.type === 'bar' && t.orientation !== 'h');
  const horizontalBars = traces.some(t => t.type === 'bar' && t.orientation === 'h');
  const xRange = Array.isArray(l.xaxis?.range) ? l.xaxis.range : [0, 1];
  const yRange = Array.isArray(l.yaxis?.range) ? l.yaxis.range : [0, 1];
  const rawXVals = hasVerticalCats ? maps.verticalLabels.map((_, i) => i) : collectNumericValuesV19(traces, 'x');
  const rawYVals = hasHorizontalCats ? maps.horizontalLabels.map((_, i) => i) : collectNumericValuesV19(traces, 'y');
  const actualXMin = rawXVals.length ? Math.min(...rawXVals) : xRange[0];
  const actualXMax = rawXVals.length ? Math.max(...rawXVals) : xRange[1];
  const actualYMin = rawYVals.length ? Math.min(...rawYVals) : yRange[0];
  const actualYMax = rawYVals.length ? Math.max(...rawYVals) : yRange[1];

  const xAxisY = hasHorizontalCats
    ? yRange[0]
    : (verticalBars ? 0 : ((actualYMin <= 0 && actualYMax >= 0) ? 0 : yRange[0]));
  const yAxisX = hasVerticalCats
    ? xRange[0]
    : (horizontalBars ? 0 : ((actualXMin <= 0 && actualXMax >= 0) ? 0 : xRange[0]));
  const yAxisStart = hasHorizontalCats ? yRange[0] : ((actualYMin < 0) ? yRange[0] : xAxisY);

  const dx = Math.max((xRange[1] - xRange[0]) * 0.018, hasVerticalCats ? 0.16 : 0.0001);
  const dy = Math.max((yRange[1] - yRange[0]) * 0.020, hasHorizontalCats ? 0.16 : 0.0001);
  const xTip = xRange[1] - dx * 0.12;
  const yTip = yRange[1] - dy * 0.12;
  const xLineEnd = xTip - dx * 0.88;
  const yLineEnd = yTip - dy * 0.88;

  l.margin = { ...(l.margin || {}), l: Math.max(100, (l.margin && l.margin.l) || 0), r: Math.max(44, (l.margin && l.margin.r) || 0), t: Math.max(88, (l.margin && l.margin.t) || 0), b: Math.max(90, (l.margin && l.margin.b) || 0) };
  l.xaxis = { ...(l.xaxis || {}), showline: false, zeroline: false };
  l.yaxis = { ...(l.yaxis || {}), showline: false, zeroline: false };
  l.shapes = (l.shapes || []).filter(s => {
    const n = String(s?.name || '');
    return !(n.includes('arrow_x_axis') || n.includes('arrow_y_axis') || n.includes('axis_head'));
  });
  l.annotations = (l.annotations || []).filter(a => {
    const n = String(a?.name || '');
    return !(n.includes('arrow_x_axis') || n.includes('arrow_y_axis') || n.includes('axis_head'));
  });

  l.shapes.push(
    {
      name: 'v21_arrow_x_axis', type: 'line', xref: 'x', yref: 'y',
      x0: yAxisX, x1: xLineEnd, y0: xAxisY, y1: xAxisY,
      line: { color: ink, width: 2.2 }, layer: 'above'
    },
    {
      name: 'v21_arrow_y_axis', type: 'line', xref: 'x', yref: 'y',
      x0: yAxisX, x1: yAxisX, y0: yAxisStart, y1: yLineEnd,
      line: { color: ink, width: 2.2 }, layer: 'above'
    },
    {
      name: 'v21_arrow_x_axis_head', type: 'path', xref: 'x', yref: 'y',
      path: axisTrianglePathV20(xTip, xAxisY, xLineEnd, xAxisY + dy * 0.46, xLineEnd, xAxisY - dy * 0.46),
      fillcolor: ink, line: { color: ink, width: 0.8 }, layer: 'above'
    },
    {
      name: 'v21_arrow_y_axis_head', type: 'path', xref: 'x', yref: 'y',
      path: axisTrianglePathV20(yAxisX, yTip, yAxisX - dx * 0.46, yLineEnd, yAxisX + dx * 0.46, yLineEnd),
      fillcolor: ink, line: { color: ink, width: 0.8 }, layer: 'above'
    }
  );
  return l;
}

function renderCenteredChart(plotMount, plotlyJson, palette, theme, options = {}) {
  if (!plotMount) return;
  const safePalette = palette && palette.length ? palette : ['#2563eb', '#ef4444', '#0f766e', '#eab308', '#7c3aed'];
  try {
    const parsed = parsePlotlyV19(plotlyJson);
    if (!parsed.data.length) {
      plotMount.innerHTML = '<div class="empty-state small">当前方法没有返回可视化数据</div>';
      return;
    }
    const maps = buildBarCategoryMapsV19(parsed.data);
    const traces = parsed.data.map((t, i) => cleanTraceV19(t, i, safePalette, maps));
    const frame = preferredChartFrameV21(plotMount, parsed);
    const layout = buildLayoutV21(parsed.layout || {}, traces, theme || {}, frame, maps);
    if (layout.showlegend === undefined) {
      layout.showlegend = traces.some(t => t.name && t.showlegend !== false) && !traces.some(t => t.type === 'bar' && (maps.verticalLabels.length || maps.horizontalLabels.length));
    }
    const trackState = options.trackState !== false;
    if (trackState) {
      STATE.currentPlotlyData = traces;
      STATE.currentPlotlyLayout = layout;
    }
    if (!window.Plotly || !Plotly.newPlot) {
      renderSvgFallbackV18(plotMount, traces, layout);
      return;
    }
    Plotly.newPlot(plotMount, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
    }).then(() => {
      if (trackState) {
        STATE.currentRenderedPlots = [{
          el: plotMount,
          index: Number.isFinite(options.index) ? options.index : 0,
          role: options.role || 'primary',
          title: options.title || (layout.title && layout.title.text) || 'Chart',
          data: traces,
          layout,
        }];
      }
      try { syncCurrentPlotlyLayoutFromDom(plotMount, trackState); } catch (_) {}
      try { installChartResizeObserver(plotMount, { trackState }); } catch (_) {}
    }).catch(err => {
      console.error('Plotly render failed; using SVG fallback:', err);
      renderSvgFallbackV18(plotMount, traces, layout);
    });
  } catch (e) {
    console.error('Chart render error; using SVG fallback:', e);
    try {
      const parsed = parsePlotlyV19(plotlyJson);
      const maps = buildBarCategoryMapsV19(parsed.data || []);
      const traces = (parsed.data || []).map((t, i) => cleanTraceV19(t, i, safePalette, maps));
      const frame = preferredChartFrameV21(plotMount, parsed);
      const layout = buildLayoutV21(parsed.layout || { title: { text: '安全图形预览' } }, traces, theme || {}, frame, maps);
      renderSvgFallbackV18(plotMount, traces, layout);
    } catch (_) {
      plotMount.innerHTML = '<div class="empty-state small">图形已生成，但当前浏览器无法渲染；请切换其他图形或下载结果。</div>';
    }
  }
}

function renderAllDiagCharts(diagnostics) {
  const container = el('diagnosticsContainer');
  if (!container) return;
  const theme = getActiveTheme();
  const palette = getActivePalette() || theme.colorway || ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A'];
  const normalizeTitle = (s) => String(s || '').toLowerCase().replace(/[\s·—\-_:：,，.。()（）\[\]【】]/g, '');
  const extractPlotTitle = (plotlyJson) => {
    try {
      const parsed = parsePlotlyV19(plotlyJson);
      return parsed?.layout?.title?.text || parsed?.layout?.title || '';
    } catch (_) {
      return '';
    }
  };
  const primaryTitles = new Set((STATE.currentChartBundle || []).flatMap(c => [String(c?.title || '').trim(), String(extractPlotTitle(c?.plotly) || '').trim()]).map(normalizeTitle).filter(Boolean));
  const primaryPlots = new Set((STATE.currentChartBundle || []).map(c => String(c?.plotly || '').trim()).filter(Boolean));
  const uniqueDiags = (diagnostics || []).filter((d, idx, arr) => {
    const title = String(d?.title || '').trim();
    const plot = String(d?.plotly || '').trim();
    const titleNorm = normalizeTitle(title || extractPlotTitle(plot));
    if (!title && !plot) return false;
    if ((titleNorm && primaryTitles.has(titleNorm)) || primaryPlots.has(plot)) return false;
    const firstIdx = arr.findIndex(x => normalizeTitle(String(x?.title || '').trim() || extractPlotTitle(x?.plotly)) === titleNorm && String(x?.plotly || '').trim() === plot);
    return firstIdx === idx;
  });
  purgePlotlyChildren(container);
  if (!uniqueDiags.length) {
    container.innerHTML = '<div class="empty-state">当前方法暂无独立的诊断评估图；结果可视化中已展示核心图形。</div>';
    return;
  }
  container.innerHTML = uniqueDiags.map((d, i) => `
    <div class="result-card diag-card">
      <h3 class="result-card-title">${escapeHtml(d.title || ('诊断图 ' + (i + 1)))}</h3>
      <div class="result-chart-container chart-diag-container" id="diag-chart-${i}"></div>
    </div>
  `).join('');
  uniqueDiags.forEach((d, i) => {
    if (!d.plotly) return;
    setTimeout(() => {
      const cardContainer = el(`diag-chart-${i}`);
      if (!cardContainer) return;
      const plotMount = document.createElement('div');
      plotMount.className = 'chart-plot';
      cardContainer.appendChild(plotMount);
      renderCenteredChart(plotMount, d.plotly, palette, theme, {
        index: i,
        title: d.title || ('Diagnostic ' + (i + 1)),
        role: 'diagnostic',
        trackState: false,
      });
    }, i * 80);
  });
}


/* ============================================================
   v25 definitive result-chart / diagnostics repair
   - render diagnostics only with visible-tab sizing
   - smaller centered preview
   - complete arrow axes
   - no duplicate diagnostics
   - robust SVG fallback without Plotly error message wall
   ============================================================ */
(function(){
  function __safeParsePlotlyV25(plotlyJson) {
    if (!plotlyJson) return { data: [], layout: {} };
    try {
      const raw = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
      const data = Array.isArray(raw?.data) ? raw.data : (raw?.data ? [raw.data] : []);
      return { data, layout: raw?.layout || {} };
    } catch (_) {
      return { data: [], layout: {} };
    }
  }

  function __preferredChartFrameV25(plotMount, parsed, role='primary') {
    const host = (role === 'diagnostic')
      ? (plotMount.closest('.chart-diag-container') || plotMount.parentElement || plotMount)
      : (plotMount.closest('.chart-preview-main') || plotMount.parentElement || plotMount);
    const rect = host.getBoundingClientRect ? host.getBoundingClientRect() : { width: 1100, height: 700 };
    const hostW = Math.max(420, Math.floor(rect.width || 1100));
    const hostH = Math.max(320, Math.floor(rect.height || (role === 'diagnostic' ? 520 : 700)));
    const data = Array.isArray(parsed?.data) ? parsed.data : [];
    const isBarLike = data.some(t => ['bar','histogram'].includes(String(t?.type || '')));
    let points = 0;
    data.forEach(t => {
      const n = Array.isArray(t?.x) ? t.x.length : (Array.isArray(t?.y) ? t.y.length : 0);
      if (n > points) points = n;
    });
    const widthRatio = role === 'diagnostic' ? 0.58 : 0.70;
    const heightRatio = role === 'diagnostic' ? 0.62 : 0.72;
    let width = Math.min(Math.floor(hostW * widthRatio), role === 'diagnostic' ? 720 : 860);
    let height = Math.min(Math.floor(hostH * heightRatio), role === 'diagnostic' ? 360 : 470);
    if (isBarLike && points > 6) {
      width = Math.min(Math.max(width, 620), role === 'diagnostic' ? 760 : 900);
      height = Math.min(Math.max(height, 340), role === 'diagnostic' ? 390 : 500);
    }
    width = Math.max(width, role === 'diagnostic' ? 520 : 620);
    height = Math.max(height, role === 'diagnostic' ? 300 : 380);
    plotMount.style.setProperty('width', `${width}px`, 'important');
    plotMount.style.setProperty('height', `${height}px`, 'important');
    plotMount.style.setProperty('min-height', `${height}px`, 'important');
    plotMount.style.setProperty('max-width', '100%', 'important');
    plotMount.style.setProperty('margin', '0 auto', 'important');
    return { width, height };
  }

  function __collectNumericV25(traces, key) {
    const vals = [];
    (traces || []).forEach(t => {
      const arr = Array.isArray(t?.[key]) ? t[key] : [];
      arr.forEach(v => {
        const n = Number(v);
        if (Number.isFinite(n)) vals.push(n);
      });
    });
    return vals;
  }

  function __categoryMapsV25(data) {
    const vertical = [];
    const horizontal = [];
    (data || []).forEach(t => {
      if (!t || t.type !== 'bar') return;
      if (t.orientation === 'h') {
        (Array.isArray(t.y) ? t.y : []).forEach(v => { if (!Number.isFinite(Number(v))) horizontal.push(String(v)); });
      } else {
        (Array.isArray(t.x) ? t.x : []).forEach(v => { if (!Number.isFinite(Number(v))) vertical.push(String(v)); });
      }
    });
    const uniq = arr => arr.filter((v, i) => arr.indexOf(v) === i);
    const vLabels = uniq(vertical);
    const hLabels = uniq(horizontal);
    return {
      verticalLabels: vLabels,
      horizontalLabels: hLabels,
      verticalMap: new Map(vLabels.map((v, i) => [v, i])),
      horizontalMap: new Map(hLabels.map((v, i) => [v, i])),
    };
  }

  function __cleanTraceV25(trace, idx, palette, maps) {
    const t = JSON.parse(JSON.stringify(trace || {}));
    const type = t.type || 'scatter';
    const color = ((STATE.userTraceColorsByChart || {})[STATE.activeChartIndex || 0] || [])[idx]
      || (Array.isArray(palette) && palette[idx % palette.length])
      || '#2563eb';
    if (type === 'bar') {
      if (t.orientation === 'h') {
        if (Array.isArray(t.y)) t.y = t.y.map(v => Number.isFinite(Number(v)) ? Number(v) : (maps.horizontalMap.has(String(v)) ? maps.horizontalMap.get(String(v)) : v));
      } else {
        if (Array.isArray(t.x)) t.x = t.x.map(v => Number.isFinite(Number(v)) ? Number(v) : (maps.verticalMap.has(String(v)) ? maps.verticalMap.get(String(v)) : v));
      }
      t.marker = { ...(t.marker || {}), color: Array.isArray(t.marker?.color) ? t.marker.color : color, line: { color: '#ffffff', width: 0.7 }, opacity: Number(STATE.markerOpacity ?? 0.9) };
      t.textposition = t.textposition || 'outside';
      t.cliponaxis = false;
    } else if (type === 'histogram') {
      t.marker = { ...(t.marker || {}), color, line: { color: '#ffffff', width: 0.6 }, opacity: 0.72 };
    } else if (type === 'box') {
      t.marker = { ...(t.marker || {}), color, size: Math.max(4, Number(STATE.markerSize || 6)), opacity: 0.75 };
      t.line = { ...(t.line || {}), color, width: 1.6 };
      t.fillcolor = t.fillcolor || `rgba(37,99,235,0.12)`;
      t.boxmean = t.boxmean !== undefined ? t.boxmean : true;
    } else if (type === 'scatter' || type === 'scattergl') {
      const mode = String(t.mode || 'markers');
      if (mode.includes('lines')) t.line = { ...(t.line || {}), color, width: Number(STATE.lineWidth || 2.4), shape: t.line?.shape || 'linear' };
      if (mode.includes('markers') || !mode) t.marker = { ...(t.marker || {}), color, size: Math.max(5, Number(STATE.markerSize || 7)), opacity: Number(STATE.markerOpacity ?? 0.86), line: { color: '#ffffff', width: 0.8 } };
      if (!t.mode) t.mode = 'markers';
    }
    return t;
  }

  function __buildLayoutV25(rawLayout, traces, theme, frameSize, maps, role='primary') {
    const l = JSON.parse(JSON.stringify(rawLayout || {}));
    const ink = theme?.ink || '#162033';
    const family = theme?.fontFamily || "'Arial', 'Noto Sans SC', sans-serif";
    const tickSize = Number(STATE.axisTickFontSize || 11);
    const axisTitleSize = Number(STATE.axisTitleFontSize || 13);
    const labelSize = Number(STATE.labelFontSize || 12);
    const bgMode = STATE.chartBackgroundMode || 'grid';
    const gridOn = bgMode === 'grid';

    if (typeof l.title === 'string') l.title = { text: l.title };
    l.title = { ...(l.title || {}), x: 0.02, xanchor: 'left', font: { family, size: 18, color: ink } };
    l.width = frameSize.width;
    l.height = frameSize.height;
    l.autosize = false;
    l.paper_bgcolor = '#ffffff';
    l.plot_bgcolor = '#ffffff';
    l.margin = {
      l: role === 'diagnostic' ? 94 : 88,
      r: 34,
      t: role === 'diagnostic' ? 70 : 78,
      b: role === 'diagnostic' ? 86 : 92,
      ...(l.margin || {}),
    };
    l.font = { ...(l.font || {}), family, color: ink, size: tickSize };
    l.legend = { orientation: 'h', x: 0, y: -0.18, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(255,255,255,0)', borderwidth: 0, font: { family, size: labelSize, color: ink }, ...(l.legend || {}) };
    l.bargap = l.bargap ?? 0.28;

    const hasPie = traces.some(t => t.type === 'pie');
    const hasHeat = traces.some(t => ['heatmap','contour','surface'].includes(String(t.type || '')));
    const hasPolar = !!l.polar;
    const isCartesian = !(hasPie || hasHeat || hasPolar);

    const patchAxis = (axis, titleText, isY) => {
      const a = { ...(axis || {}) };
      if (typeof a.title === 'string') a.title = { text: a.title };
      a.title = { ...(a.title || {}), text: (a.title && a.title.text) || titleText || '', font: { ...((a.title && a.title.font) || {}), family, size: axisTitleSize, color: ink }, standoff: isY ? 10 : 12 };
      a.tickfont = { ...(a.tickfont || {}), family, size: tickSize, color: ink };
      a.ticks = 'outside';
      a.ticklen = 5;
      a.tickwidth = 1;
      a.tickcolor = ink;
      a.showline = false;
      a.zeroline = false;
      a.showgrid = gridOn && (isY || String(a.type || '') === 'linear');
      a.gridcolor = '#e9eff6';
      a.gridwidth = 1;
      a.automargin = true;
      return a;
    };

    l.xaxis = patchAxis(l.xaxis, '', false);
    l.yaxis = patchAxis(l.yaxis, '', true);

    if (maps.verticalLabels.length) {
      l.xaxis.type = 'linear';
      l.xaxis.tickmode = 'array';
      l.xaxis.tickvals = maps.verticalLabels.map((_, i) => i);
      l.xaxis.ticktext = maps.verticalLabels;
    }
    if (maps.horizontalLabels.length) {
      l.yaxis.type = 'linear';
      l.yaxis.tickmode = 'array';
      l.yaxis.tickvals = maps.horizontalLabels.map((_, i) => i);
      l.yaxis.ticktext = maps.horizontalLabels;
    }

    if (!isCartesian) return l;

    const rawX = maps.verticalLabels.length ? maps.verticalLabels.map((_, i) => i) : __collectNumericV25(traces, 'x');
    const rawY = maps.horizontalLabels.length ? maps.horizontalLabels.map((_, i) => i) : __collectNumericV25(traces, 'y');
    const xMin = rawX.length ? Math.min(...rawX) : 0;
    const xMax = rawX.length ? Math.max(...rawX) : 1;
    const yMin0 = rawY.length ? Math.min(...rawY) : 0;
    const yMax0 = rawY.length ? Math.max(...rawY) : 1;

    const xPad = maps.verticalLabels.length ? 0.6 : Math.max((xMax - xMin) * 0.08, 0.5);
    const yPad = maps.horizontalLabels.length ? 0.6 : Math.max((yMax0 - yMin0) * 0.12, 0.8);
    const yMin = Math.min(0, yMin0);
    const yMax = Math.max(0, yMax0);
    l.xaxis.range = Array.isArray(l.xaxis?.range) ? l.xaxis.range : [xMin - xPad, xMax + xPad];
    l.yaxis.range = Array.isArray(l.yaxis?.range) ? l.yaxis.range : [yMin - yPad * 0.15, yMax + yPad];

    const xr = l.xaxis.range, yr = l.yaxis.range;
    const axisX = xr[0];
    const axisY = (yr[0] <= 0 && yr[1] >= 0) ? 0 : yr[0];
    const dx = Math.max((xr[1] - xr[0]) * 0.022, maps.verticalLabels.length ? 0.14 : 0.0001);
    const dy = Math.max((yr[1] - yr[0]) * 0.026, maps.horizontalLabels.length ? 0.14 : 0.0001);
    const xTip = xr[1] - dx * 0.55;
    const yTip = yr[1] - dy * 0.55;
    const xLineEnd = xTip - dx * 0.92;
    const yLineEnd = yTip - dy * 0.92;

    l.shapes = (l.shapes || []).filter(s => !String(s?.name || '').includes('v25_axis_'));
    l.shapes.push(
      { name: 'v25_axis_x', type: 'line', xref: 'x', yref: 'y', x0: axisX, x1: xLineEnd, y0: axisY, y1: axisY, line: { color: ink, width: 2.0 }, layer: 'above' },
      { name: 'v25_axis_y', type: 'line', xref: 'x', yref: 'y', x0: axisX, x1: axisX, y0: axisY, y1: yLineEnd, line: { color: ink, width: 2.0 }, layer: 'above' },
      { name: 'v25_axis_x_head', type: 'path', xref: 'x', yref: 'y', path: `M ${xTip} ${axisY} L ${xLineEnd} ${axisY + dy*0.52} L ${xLineEnd} ${axisY - dy*0.52} Z`, fillcolor: ink, line: { color: ink, width: 0.8 }, layer: 'above' },
      { name: 'v25_axis_y_head', type: 'path', xref: 'x', yref: 'y', path: `M ${axisX} ${yTip} L ${axisX - dx*0.52} ${yLineEnd} L ${axisX + dx*0.52} ${yLineEnd} Z`, fillcolor: ink, line: { color: ink, width: 0.8 }, layer: 'above' }
    );

    return l;
  }

  function __escapeXmlV25(s) {
    return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function __renderSvgFallbackV25(plotMount, traces, layout, role='primary') {
    const title = layout?.title?.text || layout?.title || '图形预览';
    const frame = __preferredChartFrameV25(plotMount, { data: traces }, role);
    const W = Math.max(520, frame.width), H = Math.max(320, frame.height);
    const L = 72, R = 24, T = 42, B = 56;
    const chartW = W - L - R, chartH = H - T - B;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" role="img" aria-label="${__escapeXmlV25(title)}">`;
    svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`;
    svg += `<text x="${L}" y="24" font-size="18" font-weight="700" fill="#162033">${__escapeXmlV25(title)}</text>`;
    svg += `<line x1="${L}" y1="${H-B}" x2="${W-R-16}" y2="${H-B}" stroke="#162033" stroke-width="2"/>`;
    svg += `<line x1="${L}" y1="${H-B}" x2="${L}" y2="${T+12}" stroke="#162033" stroke-width="2"/>`;
    svg += `<path d="M ${W-R} ${H-B} L ${W-R-16} ${H-B-7} L ${W-R-16} ${H-B+7} Z" fill="#162033"/>`;
    svg += `<path d="M ${L} ${T} L ${L-7} ${T+16} L ${L+7} ${T+16} Z" fill="#162033"/>`;

    const first = traces.find(t => Array.isArray(t?.x) || Array.isArray(t?.y)) || traces[0] || {};
    const type = String(first.type || 'scatter');
    const xs = Array.isArray(first.x) ? first.x : [];
    const ys = Array.isArray(first.y) ? first.y : [];

    if ((type === 'bar' || type === 'histogram') && ys.length) {
      const nums = ys.map(v => Number(v)).filter(Number.isFinite);
      const maxY = nums.length ? Math.max(...nums) : 1;
      const step = chartW / Math.max(1, ys.length);
      const bw = Math.min(56, step * 0.52);
      ys.forEach((v, i) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return;
        const x = L + i * step + (step - bw) / 2;
        const h = (n / Math.max(maxY, 1)) * (chartH - 16);
        const y = H - B - h;
        svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" fill="#3b82f6" opacity="0.85"/>`;
        svg += `<text x="${(x + bw/2).toFixed(1)}" y="${(y - 6).toFixed(1)}" font-size="10" text-anchor="middle" fill="#334155">${Number(n).toFixed(1).replace(/\.0$/, '')}</text>`;
        if (i < 12) svg += `<text x="${(x + bw/2).toFixed(1)}" y="${H-B+18}" font-size="10" text-anchor="middle" fill="#334155">${__escapeXmlV25(xs[i] ?? i).slice(0, 12)}</text>`;
      });
    } else if ((type === 'scatter' || type === 'scattergl') && xs.length && ys.length) {
      const xNums = xs.map(v => Number(v)).filter(Number.isFinite);
      const yNums = ys.map(v => Number(v)).filter(Number.isFinite);
      const minX = xNums.length ? Math.min(...xNums) : 0;
      const maxX = xNums.length ? Math.max(...xNums) : xs.length - 1;
      const minY = yNums.length ? Math.min(...yNums) : 0;
      const maxY = yNums.length ? Math.max(...yNums) : 1;
      const normX = v => L + ((Number(v) - minX) / ((maxX - minX) || 1)) * (chartW - 6);
      const normY = v => T + ((maxY - Number(v)) / ((maxY - minY) || 1)) * (chartH - 6);
      ys.forEach((v, i) => {
        if (!Number.isFinite(Number(v)) || !Number.isFinite(Number(xs[i]))) return;
        svg += `<circle cx="${normX(xs[i]).toFixed(1)}" cy="${normY(v).toFixed(1)}" r="3.2" fill="#8b5cf6" opacity="0.88"/>`;
      });
    } else if (type === 'box' && ys.length) {
      const vals = ys.map(v => Number(v)).filter(Number.isFinite).sort((a,b)=>a-b);
      if (vals.length) {
        const q = p => vals[Math.max(0, Math.min(vals.length - 1, Math.floor((vals.length - 1) * p)))];
        const q1 = q(0.25), med = q(0.5), q3 = q(0.75), mn = vals[0], mx = vals[vals.length - 1];
        const normY = v => T + ((mx - Number(v)) / ((mx - mn) || 1)) * (chartH - 20);
        const cx = L + chartW * 0.48, bw = Math.min(120, chartW * 0.22);
        svg += `<line x1="${cx}" x2="${cx}" y1="${normY(mx)}" y2="${normY(mn)}" stroke="#2563eb" stroke-width="2"/>`;
        svg += `<rect x="${cx-bw/2}" y="${normY(q3)}" width="${bw}" height="${Math.max(8,normY(q1)-normY(q3))}" fill="rgba(37,99,235,.16)" stroke="#2563eb" stroke-width="2"/>`;
        svg += `<line x1="${cx-bw/2}" x2="${cx+bw/2}" y1="${normY(med)}" y2="${normY(med)}" stroke="#2563eb" stroke-width="2.2"/>`;
      }
    }
    svg += `</svg>`;
    plotMount.innerHTML = svg;
  }

  renderCenteredChart = function(plotMount, plotlyJson, palette, theme, options = {}) {
    if (!plotMount) return;
    const role = options.role || 'primary';
    const safePalette = palette && palette.length ? palette : ['#2563eb', '#ec4899', '#10b981', '#eab308', '#7c3aed'];
    const parsed = __safeParsePlotlyV25(plotlyJson);
    if (!parsed.data.length) {
      plotMount.innerHTML = '<div class="empty-state small">当前方法没有返回可视化数据</div>';
      return;
    }
    const maps = __categoryMapsV25(parsed.data);
    const traces = parsed.data.map((t, i) => __cleanTraceV25(t, i, safePalette, maps));
    const frame = __preferredChartFrameV25(plotMount, parsed, role);
    const layout = __buildLayoutV25(parsed.layout || {}, traces, theme || {}, frame, maps, role);
    const trackState = options.trackState !== false;
    if (trackState) {
      STATE.currentPlotlyData = traces;
      STATE.currentPlotlyLayout = layout;
    }

    const finishTrack = () => {
      if (!trackState) return;
      STATE.currentRenderedPlots = [{
        el: plotMount,
        index: Number.isFinite(options.index) ? options.index : 0,
        role,
        title: options.title || (layout.title && layout.title.text) || 'Chart',
        data: traces,
        layout,
      }];
    };

    if (!window.Plotly || !Plotly.newPlot) {
      __renderSvgFallbackV25(plotMount, traces, layout, role);
      finishTrack();
      return;
    }

    Plotly.newPlot(plotMount, traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'sendDataToCloud'],
    }).then(() => {
      finishTrack();
      try { syncCurrentPlotlyLayoutFromDom(plotMount, trackState); } catch (_) {}
      try { installChartResizeObserver(plotMount, { trackState }); } catch (_) {}
    }).catch(err => {
      console.error('Plotly render failed; switching to safe SVG preview', err);
      __renderSvgFallbackV25(plotMount, traces, layout, role);
      finishTrack();
    });
  };

  renderAllDiagCharts = function(diagnostics) {
    const container = el('diagnosticsContainer');
    if (!container) return;
    const theme = getActiveTheme();
    const palette = getActivePalette() || theme.colorway || ['#2563eb', '#ec4899', '#10b981', '#eab308'];
    const normalize = s => String(s || '').toLowerCase().replace(/[\s·—\-_:：,，.。()（）\[\]【】]/g, '');
    const chartTitles = new Set((STATE.currentChartBundle || []).map(c => normalize(c?.title || '')).filter(Boolean));
    const chartPlots = new Set((STATE.currentChartBundle || []).map(c => String(c?.plotly || '').trim()).filter(Boolean));
    const unique = (diagnostics || []).filter((d, idx, arr) => {
      const t = normalize(d?.title || '');
      const p = String(d?.plotly || '').trim();
      if (!t && !p) return false;
      if (t && chartTitles.has(t)) return false;
      if (p && chartPlots.has(p)) return false;
      const firstIdx = arr.findIndex(x => normalize(x?.title || '') === t && String(x?.plotly || '').trim() === p);
      return firstIdx === idx;
    });
    STATE.currentDiagnosticsBundle = unique;
    purgePlotlyChildren(container);
    if (!unique.length) {
      container.innerHTML = '<div class="empty-state">当前方法暂无独立的诊断评估图；结果可视化中已展示核心图形。</div>';
      return;
    }
    container.innerHTML = unique.map((d, i) => `
      <div class="result-card diag-card">
        <h3 class="result-card-title">${escapeHtml(d.title || ('诊断图 ' + (i + 1)))}</h3>
        <div class="result-chart-container chart-diag-container" id="diag-chart-${i}"></div>
      </div>`).join('');
    unique.forEach((d, i) => {
      setTimeout(() => {
        const card = el(`diag-chart-${i}`);
        if (!card) return;
        card.innerHTML = '';
        const plotMount = document.createElement('div');
        plotMount.className = 'chart-plot';
        card.appendChild(plotMount);
        renderCenteredChart(plotMount, d.plotly, palette, theme, {
          index: i,
          title: d.title || ('Diagnostic ' + (i + 1)),
          role: 'diagnostic',
          trackState: false,
        });
      }, i * 40);
    });
  };
})();

/* ============================================================
   v26 stronger preview / axis / diagnostics fixes
   ============================================================ */
(function(){
  function v26Parse(plotlyJson){
    try {
      const raw = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : (plotlyJson || {});
      return { data: Array.isArray(raw.data) ? raw.data : (raw.data ? [raw.data] : []), layout: raw.layout || {} };
    } catch(e){ return { data: [], layout: {} }; }
  }
  function v26Frame(plotMount, parsed, role='primary'){
    const host = role === 'diagnostic' ? (plotMount.closest('.chart-diag-container') || plotMount.parentElement || plotMount)
                                       : (plotMount.closest('.chart-preview-main') || plotMount.parentElement || plotMount);
    const r = host.getBoundingClientRect ? host.getBoundingClientRect() : { width: 1200, height: 700 };
    let width = role === 'diagnostic' ? Math.min(640, Math.floor((r.width||1000) * 0.54)) : Math.min(760, Math.floor((r.width||1200) * 0.63));
    let height = role === 'diagnostic' ? Math.min(320, Math.floor((r.height||500) * 0.50)) : Math.min(430, Math.floor((r.height||650) * 0.66));
    width = Math.max(width, role === 'diagnostic' ? 540 : 620);
    height = Math.max(height, role === 'diagnostic' ? 280 : 360);
    plotMount.style.setProperty('width', `${width}px`, 'important');
    plotMount.style.setProperty('height', `${height}px`, 'important');
    plotMount.style.setProperty('min-height', `${height}px`, 'important');
    plotMount.style.setProperty('max-width', '100%', 'important');
    plotMount.style.setProperty('margin', '0 auto', 'important');
    return { width, height };
  }
  function v26Num(arr){ return (arr||[]).map(v => Number(v)).filter(v => Number.isFinite(v)); }
  function v26Maps(data){
    const vx=[], hy=[];
    (data||[]).forEach(t=>{
      if(String(t?.type||'')==='bar'){
        if(String(t.orientation||'')==='h') (Array.isArray(t.y)?t.y:[]).forEach(v=>{ if(!Number.isFinite(Number(v))) hy.push(String(v)); });
        else (Array.isArray(t.x)?t.x:[]).forEach(v=>{ if(!Number.isFinite(Number(v))) vx.push(String(v)); });
      }
    });
    const uniq=a=>a.filter((v,i)=>a.indexOf(v)===i);
    const v=uniq(vx), h=uniq(hy);
    return {verticalLabels:v, horizontalLabels:h, verticalMap:new Map(v.map((x,i)=>[x,i])), horizontalMap:new Map(h.map((x,i)=>[x,i]))};
  }
  function v26Trace(t, idx, palette, maps){
    const x = JSON.parse(JSON.stringify(t || {}));
    const type = String(x.type || 'scatter');
    const color = ((STATE.userTraceColorsByChart||{})[STATE.activeChartIndex||0]||[])[idx] || (palette[idx % palette.length]);
    if(type==='bar'){
      if(String(x.orientation||'')==='h') x.y = (Array.isArray(x.y)?x.y:[]).map(v=>Number.isFinite(Number(v))?Number(v):(maps.horizontalMap.has(String(v))?maps.horizontalMap.get(String(v)):v));
      else x.x = (Array.isArray(x.x)?x.x:[]).map(v=>Number.isFinite(Number(v))?Number(v):(maps.verticalMap.has(String(v))?maps.verticalMap.get(String(v)):v));
      x.marker = {...(x.marker||{}), color:Array.isArray(x.marker?.color)?x.marker.color:color, line:{color:'#ffffff',width:0.8}, opacity:Number(STATE.markerOpacity ?? 0.88)};
      x.textposition = x.textposition || 'outside'; x.cliponaxis = false;
      if(STATE.barWidth != null) x.width = Number(STATE.barWidth);
    } else if(type==='histogram'){
      x.marker = {...(x.marker||{}), color, line:{color:'#ffffff',width:0.6}, opacity:0.72};
    } else if(type==='scatter' || type==='scattergl'){
      const mode = String(x.mode || 'markers');
      if(mode.includes('lines')) x.line = {...(x.line||{}), color, width:Number(STATE.lineWidth||2.4)};
      if(mode.includes('markers') || !mode) x.marker = {...(x.marker||{}), color, size:Math.max(5,Number(STATE.markerSize||8)), opacity:Number(STATE.markerOpacity ?? 0.88), line:{color:'#ffffff',width:0.7}};
      if(!x.mode) x.mode = 'markers';
    } else if(type==='box'){
      x.marker = {...(x.marker||{}), color, size:Math.max(4,Number(STATE.markerSize||7)), opacity:0.72};
      x.line = {...(x.line||{}), color, width:1.6};
      x.boxmean = true;
    }
    return x;
  }
  function v26Layout(raw, traces, theme, frame, maps, role='primary'){
    const L = JSON.parse(JSON.stringify(raw || {}));
    const ink = theme?.ink || '#162033';
    const family = theme?.fontFamily || "'Arial', 'Noto Sans SC', sans-serif";
    const bgMode = STATE.chartBackgroundMode || 'grid';
    const gridOn = bgMode === 'grid';
    if(typeof L.title === 'string') L.title = {text:L.title};
    L.title = {...(L.title||{}), x:0.02, xanchor:'left', font:{family,size:18,color:ink}};
    L.width = frame.width; L.height = frame.height; L.autosize = false;
    L.paper_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
    L.plot_bgcolor = bgMode === 'transparent' ? 'rgba(0,0,0,0)' : '#ffffff';
    L.margin = {l:96, r:34, t:86, b:88, ...(L.margin||{})};
    L.font = {...(L.font||{}), family, color:ink, size:Number(STATE.axisTickFontSize||11)};
    L.legend = {orientation:'h', x:0, y:-0.17, xanchor:'left', yanchor:'top', bgcolor:'rgba(255,255,255,0)', font:{family,size:Number(STATE.labelFontSize||12),color:ink}, ...(L.legend||{})};
    L.bargap = L.bargap ?? 0.18;
    const hasPie = traces.some(t=>t.type==='pie');
    const hasPolar = !!L.polar; const hasHeat = traces.some(t=>['heatmap','contour','surface'].includes(String(t.type||'')));
    if(hasPie || hasPolar || hasHeat) return L;

    function patchAxis(a, titleText){
      const axis = {...(a||{})};
      if(typeof axis.title === 'string') axis.title={text:axis.title};
      axis.title = {...(axis.title||{}), text:(axis.title&&axis.title.text)||titleText||'', font:{...((axis.title&&axis.title.font)||{}), family, size:Number(STATE.axisTitleFontSize||13), color:ink}, standoff:12};
      axis.tickfont = {...(axis.tickfont||{}), family, size:Number(STATE.axisTickFontSize||11), color:ink};
      axis.ticks='outside'; axis.ticklen=5; axis.tickwidth=1; axis.tickcolor=ink;
      axis.showline=false; axis.zeroline=false; axis.showgrid=gridOn; axis.gridcolor='#e8edf5'; axis.gridwidth=1; axis.automargin=true;
      return axis;
    }
    L.xaxis = patchAxis(L.xaxis, '');
    L.yaxis = patchAxis(L.yaxis, '');
    if(maps.verticalLabels.length){ L.xaxis.type='linear'; L.xaxis.tickmode='array'; L.xaxis.tickvals=maps.verticalLabels.map((_,i)=>i); L.xaxis.ticktext=maps.verticalLabels; }
    if(maps.horizontalLabels.length){ L.yaxis.type='linear'; L.yaxis.tickmode='array'; L.yaxis.tickvals=maps.horizontalLabels.map((_,i)=>i); L.yaxis.ticktext=maps.horizontalLabels; }

    const xs = maps.verticalLabels.length ? maps.verticalLabels.map((_,i)=>i) : traces.flatMap(t=>Array.isArray(t.x)?t.x:[]).map(v=>Number(v)).filter(Number.isFinite);
    const ys = maps.horizontalLabels.length ? maps.horizontalLabels.map((_,i)=>i) : traces.flatMap(t=>Array.isArray(t.y)?t.y:[]).map(v=>Number(v)).filter(Number.isFinite);
    const xMin = xs.length ? Math.min(...xs) : 0, xMax = xs.length ? Math.max(...xs) : 1;
    const yMin0 = ys.length ? Math.min(...ys) : 0, yMax0 = ys.length ? Math.max(...ys) : 1;
    const xPad = maps.verticalLabels.length ? 0.55 : Math.max((xMax-xMin)*0.08, 0.5);
    const positiveOnly = yMin0 >= 0;
    const xRange = [xMin - xPad, xMax + xPad];
    const yRange = positiveOnly ? [0, yMax0 + Math.max(yMax0*0.15, 0.8)] : [yMin0 - Math.max((yMax0-yMin0)*0.08,0.6), yMax0 + Math.max((yMax0-yMin0)*0.12,0.8)];
    if(!Array.isArray(L.xaxis.range)) L.xaxis.range = xRange;
    if(!Array.isArray(L.yaxis.range)) L.yaxis.range = yRange;
    const xr=L.xaxis.range, yr=L.yaxis.range;
    const originX = xr[0], originY = (yr[0] <= 0 && yr[1] >= 0) ? 0 : yr[0];
    const xEnd = xr[1] - (xr[1]-xr[0])*0.015;
    const yEnd = yr[1] - (yr[1]-yr[0])*0.018;
    L.annotations = (L.annotations||[]).filter(a=>!String(a?.name||'').startsWith('v26_axis_'));
    L.annotations.push(
      {name:'v26_axis_x', x:xEnd, y:originY, ax:originX, ay:originY, xref:'x', yref:'y', axref:'x', ayref:'y', showarrow:true, arrowhead:3, arrowsize:1.2, arrowwidth:2, arrowcolor:ink, text:''},
      {name:'v26_axis_y', x:originX, y:yEnd, ax:originX, ay:originY, xref:'x', yref:'y', axref:'x', ayref:'y', showarrow:true, arrowhead:3, arrowsize:1.2, arrowwidth:2, arrowcolor:ink, text:''}
    );
    L.shapes = (L.shapes || []).filter(s => !String(s?.name||'').includes('axis'));
    return L;
  }
  function v26Escape(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function v26Svg(plotMount, traces, layout, role='primary'){
    const frame = v26Frame(plotMount, {data:traces}, role); const W=frame.width, H=frame.height;
    const title=(layout?.title?.text||layout?.title||'图形预览');
    const L=72,R=24,T=44,B=54, CW=W-L-R, CH=H-T-B;
    let svg=`<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" role="img" aria-label="${v26Escape(title)}">`;
    svg+=`<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`;
    svg+=`<text x="${L}" y="22" font-size="17" font-weight="700" fill="#162033">${v26Escape(title)}</text>`;
    const ox=L, oy=H-B, xTip=W-R, yTip=T+4;
    svg+=`<line x1="${ox}" y1="${oy}" x2="${xTip-16}" y2="${oy}" stroke="#162033" stroke-width="2"/>`;
    svg+=`<line x1="${ox}" y1="${oy}" x2="${ox}" y2="${yTip+16}" stroke="#162033" stroke-width="2"/>`;
    svg+=`<path d="M ${xTip} ${oy} L ${xTip-16} ${oy-7} L ${xTip-16} ${oy+7} Z" fill="#162033"/>`;
    svg+=`<path d="M ${ox} ${yTip} L ${ox-7} ${yTip+16} L ${ox+7} ${yTip+16} Z" fill="#162033"/>`;
    const first=traces[0]||{}; const type=String(first.type||'scatter');
    if((type==='bar'||type==='histogram') && Array.isArray(first.y)){
      const ys=first.y.map(v=>Number(v)).filter(Number.isFinite); const xs=Array.isArray(first.x)?first.x:ys.map((_,i)=>String(i+1));
      const maxY=ys.length?Math.max(...ys):1; const step=CW/Math.max(1,ys.length); const bw=Math.min(54,step*0.52);
      ys.forEach((v,i)=>{ const h=(v/Math.max(maxY,1))*(CH-20); const x=ox+i*step+(step-bw)/2; const y=oy-h; const fill=Array.isArray(first.marker?.color)?first.marker.color[i%first.marker.color.length]:(first.marker?.color||'#3b82f6'); svg+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}" opacity="0.88"/>`; svg+=`<text x="${(x+bw/2).toFixed(1)}" y="${(y-6).toFixed(1)}" font-size="10" text-anchor="middle" fill="#334155">${Number(v).toFixed(1).replace(/\.0$/,'')}</text>`; if(i<10) svg+=`<text x="${(x+bw/2).toFixed(1)}" y="${oy+16}" font-size="10" text-anchor="middle" fill="#334155">${v26Escape(xs[i]).slice(0,12)}</text>`; });
    } else if((type==='scatter'||type==='scattergl') && Array.isArray(first.x) && Array.isArray(first.y)){
      const xs=v26Num(first.x), ys=v26Num(first.y); const minX=xs.length?Math.min(...xs):0, maxX=xs.length?Math.max(...xs):1, minY=ys.length?Math.min(...ys):0, maxY=ys.length?Math.max(...ys):1;
      const fx=v=>ox+((Number(v)-minX)/((maxX-minX)||1))*(CW-10), fy=v=>T+((maxY-Number(v))/((maxY-minY)||1))*(CH-10);
      for(let i=0;i<Math.min(first.x.length, first.y.length);i++){ const xv=Number(first.x[i]), yv=Number(first.y[i]); if(!Number.isFinite(xv)||!Number.isFinite(yv)) continue; svg+=`<circle cx="${fx(xv).toFixed(1)}" cy="${fy(yv).toFixed(1)}" r="3.1" fill="#8b5cf6" opacity="0.88"/>`; }
    }
    svg+='</svg>'; plotMount.innerHTML=svg;
  }

  renderCenteredChart = function(plotMount, plotlyJson, palette, theme, options={}){
    if(!plotMount) return;
    const role = options.role || 'primary';
    const parsed = v26Parse(plotlyJson);
    if(!parsed.data.length){ plotMount.innerHTML='<div class="empty-state small">当前方法没有返回可视化数据</div>'; return; }
    const safePalette = palette && palette.length ? palette : ['#2563eb','#ec4899','#10b981','#eab308','#7c3aed'];
    const maps = v26Maps(parsed.data);
    const traces = parsed.data.map((t,i)=>v26Trace(t,i,safePalette,maps));
    const frame = v26Frame(plotMount, parsed, role);
    const layout = v26Layout(parsed.layout||{}, traces, theme||{}, frame, maps, role);
    if(options.trackState !== false){ STATE.currentPlotlyData=traces; STATE.currentPlotlyLayout=layout; }
    if(role === 'diagnostic'){
      v26Svg(plotMount, traces, layout, role);
      return;
    }
    if(!window.Plotly || !Plotly.newPlot){ v26Svg(plotMount, traces, layout, role); return; }
    Plotly.newPlot(plotMount, traces, layout, {responsive:true, displaylogo:false, displayModeBar:false, modeBarButtonsToRemove:['lasso2d','select2d','sendDataToCloud']})
      .then(()=>{ try{syncCurrentPlotlyLayoutFromDom(plotMount, options.trackState !== false);}catch(_){} try{installChartResizeObserver(plotMount,{trackState:options.trackState !== false});}catch(_){} })
      .catch((err)=>{ console.error('primary plotly render failed', err); v26Svg(plotMount, traces, layout, role); });
  };

  renderAllCharts = function(charts){
    const preview = el('chartPreviewMain');
    const tabs = el('chartVariantTabs');
    if(!preview || !tabs) return;
    const theme=getActiveTheme(); const palette=getActivePalette() || theme.colorway || ['#2563eb','#ec4899','#10b981','#eab308'];
    STATE.currentChartBundle = charts || [];
    if(!charts || !charts.length){ preview.innerHTML='<div class="empty-state">运行分析后将在这里显示结果可视化。</div>'; tabs.innerHTML=''; return; }
    tabs.innerHTML=(charts||[]).map((c,i)=>`<button class="chip-tab ${i===(STATE.activeChartIndex||0)?'active':''}" data-chart-variant="${i}">${escapeHtml(c.title||('图 '+(i+1)))}</button>`).join('');
    qsa('[data-chart-variant]', tabs).forEach(btn=>btn.addEventListener('click',()=>{ STATE.activeChartIndex=Number(btn.dataset.chartVariant)||0; renderAllCharts(STATE.currentChartBundle); try{renderAppearanceControls();}catch(_){} }));
    const active = Math.max(0, Math.min(STATE.activeChartIndex||0, charts.length-1));
    STATE.activeChartIndex = active;
    const chart = charts[active];
    preview.innerHTML='<div class="chart-active-plot"><div class="chart-plot"></div></div>';
    const mount = preview.querySelector('.chart-plot');
    renderCenteredChart(mount, chart.plotly, palette, theme, {index:active, title:chart.title, role:'primary', trackState:true});
  };

  renderAllDiagCharts = function(diagnostics){
    const container = el('diagnosticsContainer'); if(!container) return;
    const theme=getActiveTheme(); const palette=getActivePalette() || theme.colorway || ['#2563eb','#ec4899','#10b981','#eab308'];
    const norm=s=>String(s||'').toLowerCase().replace(/[\s·—\-_:：,，.。()（）\[\]【】]/g,'');
    const mainTitles=new Set((STATE.currentChartBundle||[]).map(c=>norm(c?.title||'')).filter(Boolean));
    const unique=[]; const seen=new Set();
    (diagnostics||[]).forEach(d=>{ const key=norm(d?.title||''); if(!key) return; if(mainTitles.has(key)) return; if(seen.has(key)) return; seen.add(key); unique.push(d); });
    STATE.currentDiagnosticsBundle = unique.slice(0, 3);
    if(!STATE.currentDiagnosticsBundle.length){ container.innerHTML='<div class="empty-state">当前方法暂无独立诊断评估图。</div>'; return; }
    container.innerHTML=STATE.currentDiagnosticsBundle.map((d,i)=>`<div class="result-card diag-card"><h3 class="result-card-title">${escapeHtml(d.title||('诊断图 '+(i+1)))}</h3><div class="result-chart-container chart-diag-container" id="diag-chart-${i}"></div></div>`).join('');
    STATE.currentDiagnosticsBundle.forEach((d,i)=>{ const card=el(`diag-chart-${i}`); if(!card) return; card.innerHTML='<div class="chart-plot"></div>'; const mount=card.querySelector('.chart-plot'); renderCenteredChart(mount, d.plotly, palette, theme, {index:i, title:d.title, role:'diagnostic', trackState:false}); });
  };
})();


/* ============================================================
   v27 definitive repair
   Root cause fixed:
   - v26 wrote to #chartPreviewMain, but the real container is #chartPreviewContainer.
   - Diagnostics were rendered while hidden / with duplicated plots.
   - Axis arrows used several competing old implementations.
   This block is intentionally appended last and overrides prior versions.
   ============================================================ */
(function(){
  const $ = (id) => document.getElementById(id);
  const safeHtml = (s) => (window.escapeHtml ? escapeHtml(s) : String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])));

  function parsePlotly(plotlyJson) {
    try {
      const raw = typeof plotlyJson === "string" ? JSON.parse(plotlyJson) : (plotlyJson || {});
      return { data: Array.isArray(raw.data) ? raw.data : (raw.data ? [raw.data] : []), layout: raw.layout || {} };
    } catch (e) {
      return { data: [], layout: {} };
    }
  }

  function uniq(arr) { return arr.filter((v, i) => arr.indexOf(v) === i); }

  function catMaps(data) {
    const vx = [], hy = [];
    const verticalSeries = [];
    const horizontalSeries = [];
    (data || []).forEach(t => {
      if (!t || t.type !== "bar") return;
      if (t.orientation === "h") {
        const labels = (Array.isArray(t.y) ? t.y : []).map(String);
        horizontalSeries.push(labels);
        labels.forEach(v => { if (!Number.isFinite(Number(v))) hy.push(String(v)); });
      } else {
        const labels = (Array.isArray(t.x) ? t.x : []).map(String);
        verticalSeries.push(labels);
        labels.forEach(v => { if (!Number.isFinite(Number(v))) vx.push(String(v)); });
      }
    });
    const v = uniq(vx), h = uniq(hy);
    const sameLabels = (series) => series.length > 1
      && series.every(labels => labels.length > 0)
      && series.every(labels => labels.join("\u0001") === series[0].join("\u0001"));
    const singleBarPerTrace = (series) => series.length > 1
      && series.every(labels => labels.length === 1);
    return {
      verticalLabels: v,
      horizontalLabels: h,
      verticalMap: new Map(v.map((x, i) => [x, i])),
      horizontalMap: new Map(h.map((x, i) => [x, i])),
      groupedVerticalBars: sameLabels(verticalSeries),
      groupedHorizontalBars: sameLabels(horizontalSeries),
      singleVerticalBarPerTrace: singleBarPerTrace(verticalSeries),
      singleHorizontalBarPerTrace: singleBarPerTrace(horizontalSeries),
    };
  }

  function frameSize(mount, role) {
    let width = role === "diagnostic"
      ? Math.min(Number(STATE.chartWidth || 760), 820)
      : Number(STATE.chartWidth || 760);
    let height = role === "diagnostic"
      ? Math.min(Number(STATE.chartHeight || 600), 520)
      : Number(STATE.chartHeight || 600);
    width = Math.max(480, Math.min(1400, width));
    height = Math.max(360, Math.min(1000, height));
    mount.style.setProperty("width", `${width}px`, "important");
    mount.style.setProperty("height", `${height}px`, "important");
    mount.style.setProperty("min-height", `${height}px`, "important");
    mount.style.setProperty("max-width", "none", "important");
    mount.style.setProperty("max-height", "none", "important");
    mount.style.setProperty("flex", "0 0 auto", "important");
    mount.style.setProperty("margin", "0", "important");
    mount.style.setProperty("contain", "layout size", "important");
    mount.dataset.chartWidth = String(width);
    mount.dataset.chartHeight = String(height);
    return { width, height };
  }

  function arrNums(v) { return (Array.isArray(v) ? v : []).map(Number).filter(Number.isFinite); }

  function activePalette() {
    const theme = typeof getActiveTheme === "function" ? getActiveTheme() : {};
    const custom = typeof getActivePalette === "function" ? getActivePalette() : null;
    return (custom && custom.length ? custom : theme.colorway) || ["#2563eb", "#dc2626", "#059669", "#d97706"];
  }

  function paletteColor(index, palette = activePalette()) {
    return palette[index % Math.max(palette.length, 1)] || "#2563eb";
  }

  function activeChartColorState() {
    const chartIndex = Number.isFinite(STATE.activeChartIndex) ? STATE.activeChartIndex : 0;
    return {
      chartIndex,
      series: (STATE.userTraceColorsByChart || {})[chartIndex] || [],
      categories: (STATE.userBarCategoryColorsByChart || {})[chartIndex] || {},
    };
  }

  function traceDisplayColor(traceIndex, palette = activePalette()) {
    const colors = activeChartColorState();
    return colors.series[traceIndex] || paletteColor(traceIndex, palette);
  }

  function categoryDisplayColor(label, fallback) {
    const colors = activeChartColorState();
    return colors.categories[String(label)] || fallback;
  }

  function clampOpacity(value, fallback = 0.88) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.max(0.05, Math.min(1, numeric));
  }

  function flatNumericMatrix(values) {
    if (!Array.isArray(values)) return [];
    return values.flatMap(row => Array.isArray(row) ? row : [row]).map(Number).filter(Number.isFinite);
  }

  function sameLabels(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || !a.length) return false;
    return a.every((value, index) => String(value) === String(b[index]));
  }

  function titleText(layout) {
    const title = layout && layout.title;
    if (!title) return "";
    return typeof title === "string" ? title : String(title.text || "");
  }

  function isCorrelationHeatmap(trace, layout) {
    const text = `${titleText(layout)} ${trace.name || ""}`.toLowerCase();
    if (/相关|correlation|corr|rg/.test(text)) return true;
    const values = flatNumericMatrix(trace.z);
    return values.length > 0
      && sameLabels(trace.x, trace.y)
      && values.every(value => value >= -1.000001 && value <= 1.000001);
  }

  function themeHeatmapColorscale(trace, layout) {
    const theme = typeof getActiveTheme === "function" ? getActiveTheme() : {};
    if (isCorrelationHeatmap(trace || {}, layout || {}) && Array.isArray(theme.divergentScale)) {
      return { scale: theme.divergentScale, divergent: true };
    }
    if (theme && theme.sequentialScale && Array.isArray(theme.sequentialScale)) {
      return { scale: theme.sequentialScale, divergent: false };
    }
    return { scale: null, divergent: false };
  }

  window.getChartPaletteDisplayColor = (index) => paletteColor(Number(index) || 0);
  window.getChartTraceDisplayColor = (index) => traceDisplayColor(Number(index) || 0);
  window.getChartCategoryDisplayColor = (label, fallback) => categoryDisplayColor(label, fallback || "#2563eb");

  function cleanTrace(trace, i, palette, maps, rawLayout) {
    const t = JSON.parse(JSON.stringify(trace || {}));
    const type = String(t.type || "scatter");
    const color = traceDisplayColor(i, palette);
    const barStyle = STATE.barStyle || "solid";
    const patternMap = { slash: "/", backslash: "\\", cross: "x", dot: "." };
    const patternShape = patternMap[barStyle] || "";
    const opacity = Number(STATE.markerOpacity ?? 0.88);
    const labelSize = Number(STATE.labelFontSize || 12);
    if (Number.isFinite(opacity)) t.opacity = opacity;

    if (type === "bar") {
      const isH = t.orientation === "h";
      const rawLabels = (isH ? (Array.isArray(t.y) ? t.y : []) : (Array.isArray(t.x) ? t.x : [])).map(String);

      const useTraceColor = isH
        ? (maps.groupedHorizontalBars || maps.singleHorizontalBarPerTrace)
        : (maps.groupedVerticalBars || maps.singleVerticalBarPerTrace);
      const colors = useTraceColor
        ? color
        : (rawLabels.length ? rawLabels.map((l, j) => categoryDisplayColor(l, paletteColor(j, palette))) : color);
      const outlineColors = Array.isArray(colors) ? colors : color;
      const fillColors = barStyle === "outline"
        ? (Array.isArray(colors) ? colors.map(() => "rgba(255,255,255,0)") : "rgba(255,255,255,0)")
        : colors;
      t.marker = {
        ...(t.marker || {}),
        color: fillColors,
        opacity,
        line: {
          color: barStyle === "outline" ? outlineColors : "#ffffff",
          width: barStyle === "outline" ? 2 : 0.75,
        },
        pattern: patternShape ? { shape: patternShape, fillmode: "overlay", solidity: 0.28 } : undefined,
      };
      t.width = STATE.barWidth != null ? Number(STATE.barWidth) : 0.55;
      t.textposition = t.textposition || "outside";
      if (t.text || t.texttemplate) {
        t.textfont = { ...(t.textfont || {}), size: labelSize };
        t.insidetextfont = { ...(t.insidetextfont || {}), size: labelSize };
        t.outsidetextfont = { ...(t.outsidetextfont || {}), size: labelSize };
      }
      t.cliponaxis = false;
    } else if (type === "scatter" || type === "scattergl") {
      t.type = "scatter";
      const mode = String(t.mode || "markers");
      if (mode.includes("lines")) {
        t.line = {
          ...(t.line || {}),
          color,
          width: Number(STATE.lineWidth || 2.2),
          dash: (t.line && t.line.dash) || STATE.lineDash || "solid",
        };
      }
      if (t.stackgroup || String(t.fill || "").startsWith("to")) {
        const areaAlpha = Math.max(0.18, Math.min(0.58, opacity * 0.55));
        t.fillcolor = withAlpha(color, areaAlpha);
        t.line = {
          ...(t.line || {}),
          color,
          width: Math.max(1.2, Number(STATE.lineWidth || 2.2) * 0.72),
        };
      }
      if (mode.includes("markers") || !mode.includes("lines")) {
        const rawMarkerColor = t.marker && t.marker.color;
        const pointLabels = Array.isArray(t.x) ? t.x.map(String) : [];
        const textLabels = Array.isArray(t.text) && t.text.length === pointLabels.length
          ? t.text.map(label => String(label || "").trim())
          : [];
        const markerLabels = textLabels.some(Boolean) ? textLabels : pointLabels;
        const hasCategoricalPointLabels = pointLabels.length > 1
          && (textLabels.some(Boolean) || pointLabels.some(label => !Number.isFinite(Number(label))));
        t.marker = {
          ...(t.marker || {}),
          color: hasCategoricalPointLabels
            ? markerLabels.map((label, j) => categoryDisplayColor(label, paletteColor(j, palette)))
            : (Array.isArray(rawMarkerColor) ? rawMarkerColor : color),
          size: Math.max(5, Number(STATE.markerSize || 7)),
          symbol: STATE.markerShape || "circle",
          opacity,
          line: { color: "#ffffff", width: 0.7 },
        };
      }
      if (!t.mode) t.mode = "markers";
      if (t.text || t.texttemplate) {
        t.textfont = { ...(t.textfont || {}), size: labelSize };
      }
    } else if (type === "histogram") {
      t.marker = {
        ...(t.marker || {}),
        color: barStyle === "outline" ? "rgba(255,255,255,0)" : color,
        opacity,
        line: { color: barStyle === "outline" ? color : "#ffffff", width: barStyle === "outline" ? 2 : 0.6 },
        pattern: patternShape ? { shape: patternShape, fillmode: "overlay", solidity: 0.28 } : undefined,
      };
      const bins = Math.max(5, Math.min(80, Number(STATE.histogramBins || 24)));
      if (Array.isArray(t.y) && !Array.isArray(t.x)) t.nbinsy = bins;
      else t.nbinsx = bins;
    } else if (type === "box" || type === "violin") {
      t.marker = {
        ...(t.marker || {}),
        color,
        size: Math.max(4, Number(STATE.markerSize || 6)),
        symbol: STATE.markerShape || "circle",
        opacity,
      };
      t.line = { ...(t.line || {}), color, width: Number(STATE.lineWidth || 1.6) };
      if (type === "box") {
        t.boxmean = true;
        t.boxpoints = String(STATE.boxPoints) === "false" ? false : (STATE.boxPoints || "outliers");
      } else {
        t.points = String(STATE.violinPoints) === "false" ? false : (STATE.violinPoints || "outliers");
        t.box = { ...(t.box || {}), visible: true };
        t.meanline = { ...(t.meanline || {}), visible: true };
      }
    } else if (type === "pie") {
      t.hole = Number(STATE.pieHole || 0);
      t.textposition = t.textposition || "auto";
      const labels = (Array.isArray(t.labels) ? t.labels : []).map(String);
      t.textfont = { ...(t.textfont || {}), size: labelSize };
      t.marker = {
        ...(t.marker || {}),
        colors: labels.map((label, index) => categoryDisplayColor(label, paletteColor(index, palette))),
        line: { color: "#ffffff", width: 1 },
      };
    } else if (type === "heatmap" || type === "heatmapgl" || type === "contour" || type === "surface") {
      // Heatmaps are theme-linked; correlation matrices use the theme's divergent scale.
      const themeScale = themeHeatmapColorscale(t, rawLayout);
      t.opacity = clampOpacity(STATE.heatmapOpacity, opacity);
      if (themeScale.scale) {
        t.colorscale = themeScale.scale;
        if (themeScale.divergent) {
          t.zmid = 0;
          if (t.zmin == null) t.zmin = -1;
          if (t.zmax == null) t.zmax = 1;
        }
      } else {
        t.colorscale = t.colorscale || "Blues";
      }
    } else if (type === "sankey") {
      const labels = (t.node && Array.isArray(t.node.label) ? t.node.label : []).map(String);
      t.node = {
        ...(t.node || {}),
        pad: Math.max(4, Number(STATE.sankeyNodePad || 16)),
        thickness: Math.max(6, Number(STATE.sankeyNodeThickness || 18)),
        color: labels.map((label, index) => categoryDisplayColor(label, paletteColor(index, palette))),
        line: { color: "#ffffff", width: 0.8 },
      };
      t.textfont = { ...(t.textfont || {}), size: labelSize };
      t.node.font = { ...(t.node.font || {}), size: labelSize };
    }
    if (t.text || t.texttemplate) {
      t.textfont = { ...(t.textfont || {}), size: labelSize };
    }
    if (t.error_x) {
      t.error_x = { ...(t.error_x || {}), color };
    }
    if (t.error_y) {
      t.error_y = { ...(t.error_y || {}), color };
    }
    return t;
  }

  function buildLayout(rawLayout, traces, theme, size, maps, role) {
    const layout = JSON.parse(JSON.stringify(rawLayout || {}));
    const ink = (theme && theme.ink) || "#162033";
    const family = (theme && theme.fontFamily) || "'Arial','Noto Sans SC',sans-serif";
    const bgColor = (theme && theme.bgColor) || "#ffffff";
    const plotBgColor = (theme && theme.plotBgColor) || "#ffffff";
    const gridColor = (theme && theme.gridColor) || "#e8edf5";
    const bgMode = STATE.chartBackgroundMode || "grid";
    const gridOn = bgMode === "grid";

    if (typeof layout.title === "string") layout.title = { text: layout.title };
    layout.title = {
      ...(layout.title || {}),
      text: STATE.chartTitle || (layout.title && layout.title.text) || "",
      x: 0.02,
      xanchor: "left",
      font: { family, size: Number(STATE.chartTitleFontSize || 18), color: (theme && theme.titleColor) || ink },
    };
    layout.width = size.width;
    layout.height = size.height;
    layout.autosize = false;
    layout.paper_bgcolor = bgMode === "transparent" ? "rgba(0,0,0,0)" : bgColor;
    layout.plot_bgcolor = bgMode === "transparent" ? "rgba(0,0,0,0)" : plotBgColor;
    const legendCount = traces.filter(t => t.showlegend !== false && String(t.name || "").trim()).length;
    const showLegend = layout.showlegend !== false && legendCount > 0;
    const rawMargin = layout.margin || {};
    layout.margin = {
      l: Math.max(64, Math.min(108, Number(rawMargin.l) || 82)),
      r: Math.max(28, Math.min(68, Number(rawMargin.r) || 34)),
      t: Math.max(58, Math.min(86, Number(rawMargin.t) || 66)),
      b: showLegend ? Math.max(72, Math.min(106, Number(rawMargin.b) || 82)) : 58,
    };
    const alignTitleToPlot = () => {
      layout.title.x = layout.margin.l / size.width;
      layout.title.xanchor = "left";
    };
    alignTitleToPlot();
    layout.font = { ...(layout.font || {}), family, size: Number(STATE.axisTickFontSize || 11), color: ink };
    layout.dragmode = false;
    layout.hovermode = layout.hovermode || "closest";
    layout.showlegend = showLegend;
    layout.legend = {
      ...(layout.legend || {}),
      orientation: "h", x: 0, y: -0.13, xanchor: "left", yanchor: "top",
      bgcolor: "rgba(255,255,255,0)", borderwidth: 0,
      font: { family, size: Number(STATE.labelFontSize || 12), color: ink },
    };

    const traceTypes = traces.map(t => String(t.type || "scatter"));
    const hasDomainTrace = traceTypes.some(type => [
      "pie", "sankey", "sunburst", "treemap", "funnelarea", "indicator", "table", "surface"
    ].includes(type)) || !!layout.polar;
    const hasMatrixTrace = traceTypes.some(type => ["heatmap", "contour"].includes(type));
    const patchAxis = (a, isY) => {
      const axis = { ...(a || {}) };
      if (typeof axis.title === "string") axis.title = { text: axis.title };
      axis.title = {
        ...(axis.title || {}),
        font: { ...((axis.title && axis.title.font) || {}), family, size: Number(STATE.axisTitleFontSize || 13), color: ink },
        standoff: isY ? 10 : 12
      };
      axis.tickfont = { ...(axis.tickfont || {}), family, size: Number(STATE.axisTickFontSize || 11), color: ink };
      axis.ticks = "outside"; axis.ticklen = 5; axis.tickcolor = ink; axis.tickwidth = 1;
      axis.showline = false; axis.zeroline = false;
      axis.showgrid = gridOn; axis.gridcolor = gridColor; axis.gridwidth = 1; axis.automargin = true;
      if (isY && axis.nticks == null) axis.nticks = 8;
      return axis;
    };
    layout.xaxis = patchAxis(layout.xaxis, false);
    layout.yaxis = patchAxis(layout.yaxis, true);
    const stripInjectedAxisLayer = () => {
      layout.annotations = (layout.annotations || []).filter(a => {
        const name = String((a && a.name) || "");
        if (name.startsWith("v27_axis_")) return false;
        return !(a && a.text === "" && a.showarrow === true && a.xref === "paper" && a.yref === "paper");
      });
      layout.shapes = (layout.shapes || []).filter(s => {
        const name = String((s && s.name) || "");
        return !name.startsWith("v27_axis_");
      });
    };
    const axisHeadPath = (points) => `M ${points.map(p => `${p[0]} ${p[1]}`).join(" L ")} Z`;
    const applyStandardAxes = (arrows = true) => {
      layout.xaxis.side = "bottom";
      layout.yaxis.side = "left";
      layout.xaxis.mirror = false;
      layout.yaxis.mirror = false;
      stripInjectedAxisLayer();
      if (arrows) {
        // Draw the axes as one continuous L-shaped path plus filled arrowheads.
        // This avoids the visual seams caused by mixing native lines and annotations.
        layout.xaxis.showline = false;
        layout.yaxis.showline = false;
        const axisWidth = 1.65;
        const headDepth = 0.022;
        const headHalf = 0.010;
        layout.shapes = [
          ...(layout.shapes || []),
          {
            name: "v27_axis_path",
            type: "path",
            xref: "x domain",
            yref: "y domain",
            path: "M 0 1 L 0 0 L 1 0",
            line: { color: ink, width: axisWidth },
            layer: "above",
          },
          {
            name: "v27_axis_head_x",
            type: "path",
            xref: "x domain",
            yref: "y domain",
            path: axisHeadPath([[1, 0], [1 - headDepth, headHalf], [1 - headDepth, -headHalf]]),
            fillcolor: ink,
            line: { color: ink, width: 0 },
            layer: "above",
          },
          {
            name: "v27_axis_head_y",
            type: "path",
            xref: "x domain",
            yref: "y domain",
            path: axisHeadPath([[0, 1], [-headHalf, 1 - headDepth], [headHalf, 1 - headDepth]]),
            fillcolor: ink,
            line: { color: ink, width: 0 },
            layer: "above",
          },
        ];
      } else {
        layout.xaxis.showline = true;
        layout.yaxis.showline = true;
        layout.xaxis.linecolor = ink;
        layout.yaxis.linecolor = ink;
        layout.xaxis.linewidth = 1.4;
        layout.yaxis.linewidth = 1.4;
      }
    };
    const approxSame = (a, b) => Math.abs(Number(a) - Number(b)) < 1e-9;
    const isFullSpan = (a, b) => approxSame(a, 0) && approxSame(b, 1);
    const isGeneratedAxisShape = (shape) => {
      const name = String((shape && shape.name) || "");
      return name.startsWith("v27_axis_")
        || name.startsWith("v25_axis_")
        || name.startsWith("v21_arrow_")
        || name.startsWith("v20_arrow_")
        || name === "custom_arrow_axis_v19"
        || /(^|_)axis(_|$)/i.test(name);
    };
    const isSolidZeroBaseline = (shape) => {
      const dash = String((shape && shape.line && shape.line.dash) || "solid").toLowerCase();
      if (dash && dash !== "solid") return false;
      const xref = String((shape && shape.xref) || "");
      const yref = String((shape && shape.yref) || "");
      const isHorizontalZero = approxSame(shape && shape.y0, 0)
        && approxSame(shape && shape.y1, 0)
        && isFullSpan(shape && shape.x0, shape && shape.x1)
        && /paper|domain/.test(xref);
      const isVerticalZero = approxSame(shape && shape.x0, 0)
        && approxSame(shape && shape.x1, 0)
        && isFullSpan(shape && shape.y0, shape && shape.y1)
        && /paper|domain/.test(yref);
      return isHorizontalZero || isVerticalZero;
    };
    const isAdjustableReferenceLine = (shape) => {
      if (!shape || shape.type !== "line" || isGeneratedAxisShape(shape)) return false;
      return !isSolidZeroBaseline(shape);
    };
    const applyReferenceLineStyles = () => {
      const refColor = STATE.referenceLineColor || (theme && theme.axisLineColor) || "#64748b";
      const refWidth = Math.max(0.5, Number(STATE.referenceLineWidth || 1.8));
      const refDash = STATE.referenceLineDash || "dash";
      layout.shapes = (layout.shapes || []).map(shape => {
        if (!isAdjustableReferenceLine(shape)) return shape;
        return {
          ...shape,
          line: {
            ...((shape && shape.line) || {}),
            color: refColor,
            width: refWidth,
            dash: refDash,
          },
        };
      });
    };

    if (hasDomainTrace) {
      layout.margin = {
        l: Math.min(layout.margin.l, 58),
        r: Math.min(layout.margin.r, 48),
        t: layout.margin.t,
        b: showLegend ? 76 : 42,
      };
      alignTitleToPlot();
      return layout;
    }
    if (hasMatrixTrace) {
      applyStandardAxes(false);
      applyReferenceLineStyles();
      if (!Array.isArray(layout.yaxis.range)) layout.yaxis.autorange = "reversed";
      return layout;
    }

    const valuesWithError = (axisName) => {
      const values = [];
      traces.forEach(trace => {
        const raw = arrNums(trace[axisName]);
        values.push(...raw);
        const err = trace[`error_${axisName}`] || {};
        const plus = arrNums(err.array);
        const minus = arrNums(err.arrayminus);
        raw.forEach((value, index) => {
          if (Number.isFinite(plus[index])) values.push(value + plus[index]);
          if (Number.isFinite(minus[index])) values.push(value - minus[index]);
          else if (err.symmetric !== false && Number.isFinite(plus[index])) values.push(value - plus[index]);
        });
      });
      return values;
    };
    const rangeFor = (values, options = {}) => {
      if (options.categoryCount) return [-0.55, options.categoryCount - 0.45];
      if (!values.length) return null;
      let min = Math.min(...values);
      let max = Math.max(...values);
      const magnitude = Math.max(Math.abs(min), Math.abs(max), 1);
      const span = Math.max(max - min, magnitude * 0.08, 0.1);
      if (options.zeroBaseline && min >= 0) {
        min = 0;
        max += span * 0.12;
      } else if (options.zeroBaseline && max <= 0) {
        max = 0;
        min -= span * 0.12;
      } else {
        min -= span * 0.09;
        max += span * 0.11;
      }
      return [min, max];
    };
    const hasVerticalBar = traces.some(t => String(t.type || "") === "bar" && t.orientation !== "h");
    const hasHorizontalBar = traces.some(t => String(t.type || "") === "bar" && t.orientation === "h");
    const hasZeroFilledY = traces.some(t => {
      const type = String(t.type || "scatter");
      const fill = String(t.fill || "");
      return ["scatter", "scattergl"].includes(type)
        && (t.stackgroup || fill === "tozeroy");
    });
    const xVals = valuesWithError("x");
    const yVals = valuesWithError("y");
    if (!Array.isArray(layout.xaxis.range)) {
      const range = rangeFor(xVals, {
        zeroBaseline: hasHorizontalBar,
      });
      if (range) layout.xaxis.range = range;
    }
    if (!Array.isArray(layout.yaxis.range)) {
      const range = rangeFor(yVals, {
        zeroBaseline: hasVerticalBar || hasZeroFilledY || layout.yaxis.rangemode === "tozero",
      });
      if (range) layout.yaxis.range = range;
    }

    const xr = Array.isArray(layout.xaxis.range) ? layout.xaxis.range : null;
    const yr = Array.isArray(layout.yaxis.range) ? layout.yaxis.range : null;
    if (!xr || !yr || !xr.every(Number.isFinite) || !yr.every(Number.isFinite)) {
      applyStandardAxes();
      applyReferenceLineStyles();
      if (hasVerticalBar && !yr) layout.yaxis.rangemode = "tozero";
      if (hasHorizontalBar && !xr) layout.xaxis.rangemode = "tozero";
      if (hasHorizontalBar && !Array.isArray(layout.yaxis.range)) {
        layout.yaxis.autorange = "reversed";
      }
      return layout;
    }

    stripInjectedAxisLayer();
    applyStandardAxes();
    applyReferenceLineStyles();
    if (hasHorizontalBar && !Array.isArray(layout.yaxis.range)) {
      layout.yaxis.autorange = "reversed";
    }
    return layout;
  }

  function simpleSvg(mount, traces, layout, role) {
    const size = frameSize(mount, role);
    const W = size.width, H = size.height;
    const title = (layout.title && layout.title.text) || layout.title || "图形预览";
    const L = 70, R = 25, T = 42, B = 55;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<rect width="${W}" height="${H}" fill="#fff"/><text x="${L}" y="24" font-size="16" font-weight="700" fill="#162033">${safeHtml(title)}</text>`;
    svg += `<line x1="${L}" y1="${H-B}" x2="${W-R-15}" y2="${H-B}" stroke="#162033" stroke-width="1.8"/><path d="M ${W-R} ${H-B} L ${W-R-15} ${H-B-6} L ${W-R-15} ${H-B+6} Z" fill="#162033"/>`;
    svg += `<line x1="${L}" y1="${H-B}" x2="${L}" y2="${T+15}" stroke="#162033" stroke-width="1.8"/><path d="M ${L} ${T} L ${L-6} ${T+15} L ${L+6} ${T+15} Z" fill="#162033"/>`;
    const t = traces[0] || {};
    if (String(t.type || "") === "bar" && Array.isArray(t.y)) {
      const ys = t.y.map(Number).filter(Number.isFinite);
      const xs = Array.isArray(t.x) ? t.x : ys.map((_, i) => i + 1);
      const maxY = Math.max(...ys, 1);
      const step = (W - L - R) / Math.max(ys.length, 1);
      const bw = Math.min(50, step * 0.50);
      ys.forEach((v, i) => {
        const h = (v / maxY) * (H - T - B - 18);
        const x = L + i * step + (step - bw) / 2;
        const y = H - B - h;
        svg += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" fill="#2563eb" opacity="0.86"/>`;
        if (i < 8) svg += `<text x="${x+bw/2}" y="${H-B+16}" font-size="10" text-anchor="middle" fill="#334155">${safeHtml(xs[i]).slice(0,10)}</text>`;
      });
    }
    svg += `</svg>`;
    mount.innerHTML = svg;
  }

  window.renderCenteredChart = function(mount, plotlyJson, palette, theme, options = {}) {
    if (!mount) return;
    const role = options.role || "primary";
    const parsed = parsePlotly(plotlyJson);
    if (!parsed.data.length) { mount.innerHTML = '<div class="empty-state small">当前图形无可渲染数据</div>'; return; }
    const maps = catMaps(parsed.data);
    const safePalette = palette && palette.length ? palette : ["#2563eb", "#ec4899", "#10b981", "#eab308", "#7c3aed"];
    const traces = parsed.data.map((t, i) => cleanTrace(t, i, safePalette, maps, parsed.layout || {}));
    const size = frameSize(mount, role);
    const layout = buildLayout(parsed.layout || {}, traces, theme || {}, size, maps, role);

    if (options.trackState !== false) { STATE.currentPlotlyData = traces; STATE.currentPlotlyLayout = layout; }

    if (!window.Plotly || !Plotly.newPlot) { simpleSvg(mount, traces, layout, role); return; }
    Plotly.newPlot(mount, traces, layout, {
      responsive: false,
      displaylogo: false,
      displayModeBar: false,
      editable: false,
      scrollZoom: false,
      doubleClick: false,
      showTips: false,
    })
      .then(() => { try { if (options.trackState !== false) syncCurrentPlotlyLayoutFromDom(mount, true); } catch(e){} })
      .catch(err => { console.error("Plotly render failed", err); simpleSvg(mount, traces, layout, role); });
  };

  window.renderAllCharts = function(charts) {
    const container = $("chartPreviewContainer");
    const tabs = $("chartVariantTabs");
    if (!container) return;
    STATE.currentChartBundle = charts || [];
    if (!STATE.currentChartBundle.length) {
      container.innerHTML = '<div class="empty-state">运行分析后将在这里显示结果可视化。</div>';
      if (tabs) tabs.innerHTML = "";
      return;
    }
    STATE.activeChartIndex = Math.max(0, Math.min(STATE.activeChartIndex || 0, STATE.currentChartBundle.length - 1));
    if (tabs) {
      tabs.innerHTML = STATE.currentChartBundle.map((c, i) => `<button class="chart-variant-tab ${i === STATE.activeChartIndex ? "active" : ""}" data-chart-index="${i}">${safeHtml(c.title || ("图 " + (i + 1)))}</button>`).join("");
      tabs.querySelectorAll("[data-chart-index]").forEach(btn => btn.addEventListener("click", () => { STATE.activeChartIndex = Number(btn.dataset.chartIndex) || 0; renderAllCharts(STATE.currentChartBundle); try { renderAppearanceControls(); } catch(e){} }));
      tabs.style.display = STATE.currentChartBundle.length > 1 ? "flex" : "none";
    }
    container.innerHTML = '<div id="chartActivePlot" class="chart-active-plot"><div class="chart-plot"></div></div>';
    const mount = container.querySelector(".chart-plot");
    const theme = getActiveTheme();
    const palette = getActivePalette() || theme.colorway || ["#2563eb", "#ec4899", "#10b981", "#eab308"];
    const active = STATE.currentChartBundle[STATE.activeChartIndex];
    renderCenteredChart(mount, active.plotly, palette, theme, { index: STATE.activeChartIndex, title: active.title, role: "primary", trackState: true });
  };

  window.renderAllDiagCharts = function(diagnostics) {
    const container = $("diagnosticsContainer");
    if (!container) return;
    const norm = s => String(s || "").toLowerCase().replace(/[\s·—\-_:：,，.。()（）\[\]【】]/g, "");
    const mainTitles = new Set((STATE.currentChartBundle || []).map(c => norm(c && c.title)).filter(Boolean));
    const unique = [];
    const seen = new Set();
    (diagnostics || []).forEach(d => {
      const key = norm(d && d.title);
      if (!key || seen.has(key) || mainTitles.has(key)) return;
      seen.add(key); unique.push(d);
    });
    STATE.currentDiagnosticsBundle = unique.slice(0, 3);
    if (!STATE.currentDiagnosticsBundle.length) {
      container.innerHTML = '<div class="empty-state">当前方法暂无独立诊断评估图。</div>';
      return;
    }
    const theme = getActiveTheme();
    const palette = getActivePalette() || theme.colorway || ["#2563eb", "#ec4899", "#10b981", "#eab308"];
    container.innerHTML = STATE.currentDiagnosticsBundle.map((d, i) => `<div class="result-card diag-card"><h3 class="result-card-title">${safeHtml(d.title || ("诊断图 " + (i + 1)))}</h3><div class="result-chart-container chart-diag-container" id="diag-chart-${i}"><div class="chart-plot"></div></div></div>`).join("");
    STATE.currentDiagnosticsBundle.forEach((d, i) => {
      const mount = $(`diag-chart-${i}`).querySelector(".chart-plot");
      renderCenteredChart(mount, d.plotly, palette, theme, { index: i, title: d.title, role: "diagnostic", trackState: false });
    });
  };
})();
