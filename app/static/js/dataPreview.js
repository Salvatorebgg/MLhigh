/* ── Data Preview Module ───────────────────────────────── */
/* Data preview rendering is in tableGenerator.js (renderDataPreview).
   This module provides selector-refresh utilities.              */

function updateAllSelectors() {
  setTimeout(() => {
    if (typeof buildMethodVarControls === 'function') {
      buildMethodVarControls();
    }
    if (typeof buildTableVarControls === 'function') {
      buildTableVarControls();
    }
  }, 100);
}
