# Clinical Method Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four clinically relevant methods and expand tuning controls while keeping the current method-first workflow intact.

**Architecture:** Extend existing registries instead of reshaping the app. Add data generators in `sample_service.py`, method metadata and validation in `main.py`, runners in `stats_service.py` and `ml_service.py`, and smoke coverage in `tests/smoke.py`.

**Tech Stack:** FastAPI, pandas, numpy, scipy, statsmodels, scikit-learn, Plotly, pytest-style smoke tests.

---

### Task 1: Tests

**Files:**
- Modify: `tests/smoke.py`

- [ ] Add assertions for the four new method IDs, categories, examples, role recommendation, and parameter counts.
- [ ] Add focused role-validation cases for `target_trial_emulation`, `doubly_robust_aipw`, `ml_stacking`, and `ml_mlp`.
- [ ] Run `python tests\smoke.py`; expected first result before implementation: failure because the new catalog entries are missing.

### Task 2: Method Catalog And Validation

**Files:**
- Modify: `app/main.py`

- [ ] Add four catalog entries with unique example datasets and at least three parameters each.
- [ ] Add `METHOD_EXAMPLE_MAP` entries.
- [ ] Add parameter boosts for new and existing methods.
- [ ] Extend `_derive_params_from_roles`, `_validate_method_roles`, and `_recommend_roles_for_method`.

### Task 3: Example Data

**Files:**
- Modify: `app/services/sample_service.py`

- [ ] Add synthetic clinical examples for target trial emulation, doubly robust AIPW, stacking, and MLP.
- [ ] Register each generator in `EXAMPLE_MAKERS`.
- [ ] Let application startup create the corresponding CSV files if missing.

### Task 4: Advanced Statistics Runners

**Files:**
- Modify: `app/services/stats_service.py`

- [ ] Implement `run_target_trial_emulation`.
- [ ] Implement `run_doubly_robust_aipw`.
- [ ] Register both in `STATS_ROUTER`.
- [ ] Return tables, charts, diagnostics, and discussion.

### Task 5: Machine Learning Runners

**Files:**
- Modify: `app/services/ml_service.py`

- [ ] Implement `run_ml_stacking`.
- [ ] Implement `run_ml_mlp`.
- [ ] Register both in `ML_ROUTER`.
- [ ] Return tables, charts, diagnostics, and discussion.

### Task 6: Verification

**Files:**
- Modify: generated `data/examples/*.csv` as needed by application startup.

- [ ] Run `python tests\smoke.py`.
- [ ] Run targeted syntax checks for modified Python files.
- [ ] Inspect `git diff -- tests/smoke.py app/main.py app/services/sample_service.py app/services/stats_service.py app/services/ml_service.py`.
