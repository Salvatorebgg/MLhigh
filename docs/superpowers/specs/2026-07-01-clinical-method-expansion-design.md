# Clinical Method Expansion Design

## Goal

Expand the method-first MLhigh workflow with a small set of current, clinically relevant advanced statistics and machine learning methods while preserving the existing user flow: choose method, load example or upload data, auto-recommend variables, tune parameters, run analysis, inspect tables, charts, diagnostics, and discussion.

## Classification

The current "machine learning" group contains both models and ML workflow tools. Logistic regression, Lasso, Ridge, KNN, XGBoost, random forest, SVM, decision tree, and CNN are modeling methods. Feature engineering, model comparison, dimensionality reduction, and clustering are ML workflow or unsupervised-learning methods. They should remain under `ml_models` because the application currently exposes two top-level categories only: `advanced_stats` and `ml_models`.

## New Advanced Statistics Methods

1. `target_trial_emulation`
   - Purpose: emulate a pragmatic target trial in observational clinical data.
   - Inputs: treatment/exposure, outcome, follow-up time, baseline covariates.
   - Output: eligibility summary, stabilized IPTW balance, risk difference/risk ratio, weighted outcome model, cumulative incidence chart, propensity overlap chart, covariate balance chart, and interpretation.

2. `doubly_robust_aipw`
   - Purpose: estimate average treatment effects using a doubly robust AIPW style estimator.
   - Inputs: binary treatment, binary or continuous outcome, baseline covariates.
   - Output: ATE/ATT-style estimates, propensity diagnostics, outcome model diagnostics, ITE distribution, and interpretation.

## New Machine Learning Methods

1. `ml_stacking`
   - Purpose: clinical prediction with a stacked ensemble across logistic regression, random forest, gradient boosting, and SVM-style learners.
   - Inputs: target and feature variables.
   - Output: model comparison table, final stacking metrics, ROC/PR/calibration/confusion matrix, feature importance surrogate, and interpretation.

2. `ml_mlp`
   - Purpose: tabular deep learning style classification using scikit-learn MLP.
   - Inputs: target and feature variables.
   - Output: classification metrics, training loss curve, ROC/PR/calibration/confusion matrix, permutation importance, and interpretation.

## Parameter Expansion

Existing ML methods should expose practical tuning controls already honored by service wrappers or easy to wire into wrappers: split ratio, random seed, threshold, class weight, C/penalty, tree counts, depth, learning rate, SVM kernel/gamma, KNN weights, and decision tree leaf size. Existing advanced statistics methods should expose conservative controls: confidence level, bootstrap iterations, stabilized weights, trimming percentile, matching caliper, correlation structure, prior scale, and analysis horizon.

## Data Validation

Each new method must reject incompatible datasets through `_validate_method_roles` with a clear reason. The recommendation endpoint must find suitable role mappings for each method example and for user uploads that contain the necessary variables.

## Verification

The smoke suite must prove that every registered method has a unique example dataset, at least three parameters, a successful role recommendation, non-empty tables, non-empty charts, and a substantial discussion.
