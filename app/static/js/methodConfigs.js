/* ── MLhigh Method Configurations ────────────────────── */
/* Method catalog — drives both the UI and backend routing */

const METHOD_CATALOG = {};

// Fetched from backend at init
async function loadMethodCatalog() {
  try {
    const resp = await apiGet('/api/methods');
    resp.methods.forEach(m => { METHOD_CATALOG[m.id] = m; });
    return resp.methods;
  } catch (e) {
    console.error('Failed to load method catalog', e);
    return [];
  }
}

function getMethodConfig(methodId) {
  return METHOD_CATALOG[methodId] || null;
}

function getMethodsByCategory(category) {
  return Object.values(METHOD_CATALOG).filter(m => m.category === category);
}

const METHOD_CATEGORIES = [
  { id: 'advanced_stats', name: '高级统计', icon: 'Σ' },
  { id: 'ml_models', name: '机器学习', icon: 'ML' },
  { id: 'tools', name: '综合工具', icon: '⚙' },
];
