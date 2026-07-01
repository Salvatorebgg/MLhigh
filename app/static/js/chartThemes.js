/* ── Chart Themes — Publication-quality palettes ─────────── */

const CHART_THEMES = {
  cnsTheme: {
    name: '临床研究通用出版',
    fontFamily: "'Arial', 'Helvetica Neue', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(31, 41, 55, 0.08)',
    zeroLineColor: 'rgba(31, 41, 55, 0.18)',
    titleColor: '#111827',
    axisColor: '#1F2937',
    axisLineColor: '#27313D',
    ink: '#111827',
    colorway: ['#2E6F9E', '#D95F59', '#2A9D8F', '#E9A93A', '#6F5AA7', '#7C8B52', '#C776A5', '#4A5568', '#8F6B43', '#5BA4CF', '#3D8B6E', '#D4A05A', '#8B5E83', '#5A8F7B', '#C9684E', '#6B8DB5'],
    sequentialScale: [[0, '#F7FBFF'], [0.25, '#D9EAF7'], [0.5, '#87BBD8'], [0.75, '#2E6F9E'], [1, '#14395B']],
    divergentScale: [[0, '#B64C4C'], [0.5, '#F8FAFC'], [1, '#246B80']],
    mapScale: [[0, '#F7FBFF'], [0.2, '#C6DBEF'], [0.45, '#6BAED6'], [0.7, '#2171B5'], [1, '#08306B']],
    markerLine: '#FFFFFF',
    opacity: 0.88,
    titleFontSize: 18,
    axisFontSize: 13,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  clinicalTheme: {
    name: 'CHARLS 临床',
    fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(20, 42, 46, 0.07)',
    zeroLineColor: '#DCE8E3',
    titleColor: '#142A2E',
    axisColor: '#142A2E',
    axisLineColor: '#3A5A5E',
    ink: '#142A2E',
    colorway: ['#0E7C7B', '#B34D3E', '#274C77', '#D59F32', '#665C9E', '#4D86B8', '#5F8D4E', '#C76E52', '#59788E', '#A5688F', '#3BA89A', '#D47A6E', '#3868A0', '#E2B84A', '#7E72B8', '#68A06A'],
    sequentialScale: [[0, '#F0FAF9'], [0.3, '#A8DCD9'], [0.6, '#4AABA9'], [1, '#0E7C7B']],
    divergentScale: [[0, '#B34D3E'], [0.5, '#F8FAFC'], [1, '#0E7C7B']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  journalTheme: {
    name: '经典期刊',
    fontFamily: "'Arial', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(0,0,0,0.06)',
    zeroLineColor: '#333333',
    titleColor: '#000000',
    axisColor: '#222222',
    axisLineColor: '#222222',
    ink: '#000000',
    colorway: ['#1B4F72', '#922B21', '#1E8449', '#B7950B', '#6C3483', '#1A5276', '#784212', '#17202A', '#4A235A', '#0E6655', '#2E86C1', '#C0392B', '#27AE60', '#D4AC0D', '#8E44AD', '#1F618D'],
    sequentialScale: [[0, '#F0F5FA'], [0.3, '#A0C4D8'], [0.6, '#4A7FA0'], [1, '#1B4F72']],
    divergentScale: [[0, '#922B21'], [0.5, '#F8F8F8'], [1, '#1B4F72']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 16,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  natureStyleTheme: {
    name: 'Nature 系配色',
    fontFamily: "'Arial', 'Helvetica Neue', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(17,24,39,0.07)',
    zeroLineColor: 'rgba(17,24,39,0.18)',
    titleColor: '#111111',
    axisColor: '#202124',
    axisLineColor: '#202124',
    ink: '#111111',
    colorway: ['#3C5488', '#E64B35', '#00A087', '#4DBBD5', '#F39B7F', '#8491B4', '#91D1C2', '#7E6148', '#B09C85', '#DC0000', '#5A7DB5', '#F07355', '#2DBFA5', '#6DCDE0', '#E6AFA0', '#9AA8C0'],
    sequentialScale: [[0, '#F7FBFF'], [0.35, '#BBD7EA'], [0.7, '#4F93B7'], [1, '#164A73']],
    divergentScale: [[0, '#B2182B'], [0.5, '#F7F7F7'], [1, '#2166AC']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 17,
    axisFontSize: 13,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  lancetTheme: {
    name: 'Lancet',
    fontFamily: "'Arial', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(0,0,0,0.06)',
    zeroLineColor: 'rgba(0,0,0,0.15)',
    titleColor: '#00468B',
    axisColor: '#1a1a1a',
    axisLineColor: '#1a1a1a',
    ink: '#1a1a1a',
    colorway: ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A', '#ADB6B6', '#1B1919', '#F0E685', '#1A6EC7', '#FF3030', '#58D158', '#22C0DA', '#B080C0', '#FFC8A0'],
    sequentialScale: [[0, '#F0F4FA'], [0.3, '#A8C4E0'], [0.6, '#4A8CC2'], [1, '#00468B']],
    divergentScale: [[0, '#ED0000'], [0.5, '#F8F8F8'], [1, '#00468B']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  nejmTheme: {
    name: 'NEJM',
    fontFamily: "'Arial', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(0,0,0,0.06)',
    zeroLineColor: 'rgba(0,0,0,0.15)',
    titleColor: '#1a1a1a',
    axisColor: '#1a1a1a',
    axisLineColor: '#1a1a1a',
    ink: '#1a1a1a',
    colorway: ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91', '#EE4C97', '#8C564B', '#2CA02C', '#D85A4A', '#2892D0', '#F0A050', '#38A068', '#9088C0', '#88B0C0'],
    sequentialScale: [[0, '#FFF5F0'], [0.3, '#FCBBA1'], [0.6, '#EF6548'], [1, '#A50F15']],
    divergentScale: [[0, '#0072B5'], [0.5, '#F8F8F8'], [1, '#BC3C29']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  scienceTheme: {
    name: 'Science',
    fontFamily: "'Arial', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(0,0,0,0.06)',
    zeroLineColor: 'rgba(0,0,0,0.15)',
    titleColor: '#1a1a1a',
    axisColor: '#1a1a1a',
    axisLineColor: '#1a1a1a',
    ink: '#1a1a1a',
    colorway: ['#3B4992', '#EE0000', '#008B45', '#631879', '#008280', '#BB0021', '#5F559B', '#A20056', '#808180', '#1B195E', '#5868B0', '#FF3030', '#20B060', '#8030A0', '#20A8A0', '#DD1040'],
    sequentialScale: [[0, '#F0F0F8'], [0.3, '#B8B8D8'], [0.6, '#6868A8'], [1, '#3B4992']],
    divergentScale: [[0, '#EE0000'], [0.5, '#F8F8F8'], [1, '#3B4992']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  warmTheme: {
    name: '暖色调',
    fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(80,60,40,0.06)',
    zeroLineColor: '#DDD0C0',
    titleColor: '#3D2E1E',
    axisColor: '#4A3828',
    axisLineColor: '#4A3828',
    ink: '#3D2E1E',
    colorway: ['#C5741A', '#B34D3E', '#D59F32', '#8A5A30', '#E08050', '#A05040', '#C09050', '#906040', '#D4A574', '#7A4420', '#D88A30', '#C86050', '#E0B848', '#A07040', '#F09868', '#B86858'],
    sequentialScale: [[0, '#FFF8F0'], [0.3, '#F5D4A8'], [0.6, '#D89850'], [1, '#8A5A30']],
    divergentScale: [[0, '#B34D3E'], [0.5, '#FFF8F0'], [1, '#C5741A']],
    markerLine: '#FFFFFF',
    opacity: 0.88,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  coolTheme: {
    name: '冷色调',
    fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(30,50,80,0.06)',
    zeroLineColor: 'rgba(30,50,80,0.15)',
    titleColor: '#1a2a40',
    axisColor: '#1a2a40',
    axisLineColor: '#2a3a50',
    ink: '#1a2a40',
    colorway: ['#2E5090', '#4A90D9', '#1B7A8A', '#6B8FB5', '#3A6B9F', '#5BAEB5', '#2D6B7F', '#7AAFC8', '#1F4E6F', '#4ECDC4', '#3A68B0', '#62A8E8', '#2598A8', '#80A8D0', '#4A82B8', '#70C0D0'],
    sequentialScale: [[0, '#F0F5FA'], [0.3, '#A8C8E8'], [0.6, '#4A8AB8'], [1, '#1A3A5C']],
    divergentScale: [[0, '#4ECDC4'], [0.5, '#F8F8F8'], [1, '#2E5090']],
    markerLine: '#FFFFFF',
    opacity: 0.88,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  pastelTheme: {
    name: '柔和马卡龙',
    fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(0,0,0,0.05)',
    zeroLineColor: 'rgba(0,0,0,0.10)',
    titleColor: '#3a3a3a',
    axisColor: '#4a4a4a',
    axisLineColor: '#4a4a4a',
    ink: '#3a3a3a',
    colorway: ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD', '#A8E0D8', '#FFFFC8', '#D0D0E8', '#FCA0A0', '#A0C8E0', '#FFD088'],
    sequentialScale: [[0, '#F8FDFC'], [0.3, '#C8EDE7'], [0.6, '#8DD3C7'], [1, '#4AA89A']],
    divergentScale: [[0, '#FB8072'], [0.5, '#FEFEFE'], [1, '#8DD3C7']],
    markerLine: '#FFFFFF',
    opacity: 0.92,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  darkMutedTheme: {
    name: '深色低调',
    fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif",
    bgColor: '#1E2328',
    plotBgColor: '#252A30',
    gridColor: 'rgba(120,130,140,0.15)',
    zeroLineColor: 'rgba(160,170,180,0.25)',
    titleColor: '#E0E4E8',
    axisColor: '#BCC3C9',
    axisLineColor: '#6A7580',
    ink: '#E0E4E8',
    colorway: ['#8AC6B4', '#D08070', '#6090B8', '#D0A850', '#9888B8', '#68A0C8', '#78B098', '#D09078', '#7898B0', '#B88898', '#A0D8C8', '#E09888', '#78A8D0', '#E0B868', '#B098C8', '#80B8D8'],
    sequentialScale: [[0, '#2A3038'], [0.3, '#3A5A60'], [0.6, '#5A9A90'], [1, '#8AC6B4']],
    divergentScale: [[0, '#D08070'], [0.5, '#3A4048'], [1, '#8AC6B4']],
    markerLine: 'rgba(0,0,0,0.3)',
    opacity: 0.88,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },

  monoTheme: {
    name: '单色灰阶',
    fontFamily: "'Arial', 'Noto Sans SC', sans-serif",
    bgColor: '#FFFFFF',
    plotBgColor: '#FFFFFF',
    gridColor: 'rgba(0,0,0,0.06)',
    zeroLineColor: 'rgba(0,0,0,0.15)',
    titleColor: '#1a1a1a',
    axisColor: '#1a1a1a',
    axisLineColor: '#1a1a1a',
    ink: '#1a1a1a',
    colorway: ['#1a1a1a', '#4a4a4a', '#7a7a7a', '#a0a0a0', '#c0c0c0', '#3a3a3a', '#5a5a5a', '#8a8a8a', '#b0b0b0', '#d0d0d0', '#2a2a2a', '#606060', '#959595', '#b8b8b8', '#484848', '#6e6e6e'],
    sequentialScale: [[0, '#F8F8F8'], [0.3, '#C0C0C0'], [0.6, '#707070'], [1, '#1a1a1a']],
    divergentScale: [[0, '#1a1a1a'], [0.5, '#F0F0F0'], [1, '#1a1a1a']],
    markerLine: '#FFFFFF',
    opacity: 0.90,
    titleFontSize: 17,
    axisFontSize: 12,
    tickFontSize: 11,
    legendFontSize: 11,
  },
};

const COLOR_PALETTES = {
  default: null,
  tableau10: ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'],
  set2: ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3'],
  paired: ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A'],
  dark2: ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'],
  accent: ['#7FC97F', '#BEAED4', '#FDC086', '#FFFF99', '#386CB0', '#F0027F', '#BF5B17', '#666666'],
  viridis: ['#440154', '#482878', '#3E4989', '#31688E', '#26828E', '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725'],
  plasma: ['#0D0887', '#4B03A1', '#7D03A8', '#A82296', '#CB4679', '#E56B5D', '#F89441', '#FDC328', '#F0F921'],
};

function getActiveTheme() {
  return CHART_THEMES[STATE.chartTheme] || CHART_THEMES.cnsTheme;
}

function getActivePalette() {
  const paletteName = STATE.chartPalette || 'default';
  if (paletteName === 'custom' && STATE.customPalette && STATE.customPalette.length > 0) {
    return STATE.customPalette;
  }
  if (paletteName !== 'default' && COLOR_PALETTES[paletteName]) {
    return COLOR_PALETTES[paletteName];
  }
  return null;
}

function applyThemeLayout(layout, theme) {
  const titleObj = typeof layout.title === 'string'
    ? { text: layout.title }
    : (layout.title || { text: '' });

  const normAxisTitle = (ax) => {
    if (!ax) return {};
    if (typeof ax.title === 'string') return { text: ax.title };
    return ax.title || {};
  };

  const themedAxis = (axisConfig) => ({
    ...axisConfig,
    gridcolor: theme.gridColor,
    zerolinecolor: theme.zeroLineColor,
    linecolor: theme.axisLineColor,
    tickfont: { family: theme.fontFamily, color: theme.axisColor, size: theme.tickFontSize },
    title: { ...normAxisTitle(axisConfig), font: { family: theme.fontFamily, color: theme.axisColor, size: theme.axisFontSize } },
  });

  if (!layout.xaxis && !layout.grid && !layout.geo) layout.xaxis = {};
  if (!layout.yaxis && !layout.grid && !layout.geo) layout.yaxis = {};

  const result = {
    ...layout,
    font: { family: theme.fontFamily, color: theme.axisColor, size: theme.axisFontSize },
    title: {
      ...titleObj,
      x: titleObj.x ?? 0.02,
      xanchor: titleObj.xanchor || 'left',
      font: { family: theme.fontFamily, color: theme.titleColor, size: theme.titleFontSize },
    },
    paper_bgcolor: theme.bgColor,
    plot_bgcolor: theme.plotBgColor,
    legend: {
      ...layout.legend,
      font: { family: theme.fontFamily, color: theme.axisColor, size: theme.legendFontSize },
    },
  };

  for (const key of Object.keys(layout)) {
    if (/^xaxis\d*$/.test(key)) result[key] = themedAxis(layout[key] || {});
    if (/^yaxis\d*$/.test(key)) result[key] = themedAxis(layout[key] || {});
  }

  return result;
}
