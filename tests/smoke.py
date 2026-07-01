"""Smoke tests for the simplified method-first MLhigh workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import UPLOADS_DIR
from app.main import METHOD_CATALOG, app, run_analysis, _validate_method_roles
from app.schemas import MethodRequest
from app.services.io_service import get_example_datasets
from app.services.sample_service import EXAMPLE_MAKERS
from app.services.variable_service import classify_variables, summarize_dataset


NEW_METHOD_IDS = {
    "target_trial_emulation",
    "doubly_robust_aipw",
    "ml_stacking",
    "ml_mlp",
    "ml_elastic_net",
    "ml_deepsurv",
    "ml_deephit",
    "marginal_structural_model",
}


def test_method_catalog_is_method_first():
    methods = list(METHOD_CATALOG.values())
    assert methods
    assert {m["category"] for m in methods} <= {"advanced_stats", "ml_models"}
    assert NEW_METHOD_IDS.issubset(METHOD_CATALOG)
    assert METHOD_CATALOG["target_trial_emulation"]["category"] == "advanced_stats"
    assert METHOD_CATALOG["doubly_robust_aipw"]["category"] == "advanced_stats"
    assert METHOD_CATALOG["marginal_structural_model"]["category"] == "advanced_stats"
    assert METHOD_CATALOG["ml_stacking"]["category"] == "ml_models"
    assert METHOD_CATALOG["ml_mlp"]["category"] == "ml_models"
    assert METHOD_CATALOG["ml_elastic_net"]["category"] == "ml_models"
    assert METHOD_CATALOG["ml_deepsurv"]["category"] == "ml_models"
    assert METHOD_CATALOG["ml_deephit"]["category"] == "ml_models"
    assert all(m.get("example_dataset") and m["example_dataset"] != "comprehensive_example" for m in methods)
    assert len({m["example_dataset"] for m in methods}) == len(methods)
    assert METHOD_CATALOG["feature_engineering"]["category"] == "ml_models"
    assert METHOD_CATALOG["model_comparison"]["category"] == "ml_models"
    assert METHOD_CATALOG["dim_reduction"]["category"] == "ml_models"
    assert METHOD_CATALOG["cluster"]["category"] == "ml_models"
    assert all(len(m.get("params") or []) >= 3 for m in methods)


def test_example_registry_has_only_method_examples():
    assert "comprehensive_example" not in EXAMPLE_MAKERS
    examples = get_example_datasets()
    names = {item["name"] for item in examples}
    required = {cfg["example_dataset"] for cfg in METHOD_CATALOG.values()}
    assert "comprehensive_example" not in names
    assert required.issubset(names), f"Missing method examples: {sorted(required - names)}"


def test_sample_generation_and_variable_summary():
    for name, maker in EXAMPLE_MAKERS.items():
        df = maker()
        assert len(df) > 0, f"{name}: empty dataframe"
        assert len(df.columns) > 0, f"{name}: no columns"
    df = EXAMPLE_MAKERS[METHOD_CATALOG["gee"]["example_dataset"]]()
    types = classify_variables(df)
    summary = summarize_dataset(df, types)
    assert summary["sample_size"] > 0
    assert summary["variable_count"] > 0
    assert "continuous" in types


def test_each_method_runs_with_its_own_example():
    for method_id, cfg in METHOD_CATALOG.items():
        result = run_analysis(MethodRequest(
            method_id=method_id,
            use_demo=True,
            dataset_name=cfg["example_dataset"],
            params={},
            role_vars={},
        ))
        assert result.get("tables"), f"{method_id}: no tables"
        assert result.get("charts"), f"{method_id}: no charts"
        assert len(result.get("discussion") or "") > 800, f"{method_id}: discussion too short"
        assert result["method_id"] == method_id


def test_uploaded_csv_http_workflow():
    client = TestClient(app)
    rows = ["x1,x2,label"]
    rows.extend(f"{index / 10:.1f},{(index % 7) / 3:.3f},{index % 2}" for index in range(48))
    response = client.post(
        "/api/upload",
        files={"file": ("clinical.csv", "\n".join(rows).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text
    upload = response.json()
    upload_id = upload["upload_id"]
    try:
        recommendation = client.post("/api/recommend-roles", json={
            "method_id": "ml_lr",
            "upload_id": upload_id,
            "use_demo": False,
        })
        assert recommendation.status_code == 200, recommendation.text
        recommended = recommendation.json()
        assert recommended["available"], recommended.get("reason")
        assert recommended["roles"]["outcome_vars"] == ["label"]
        assert {"x1", "x2"}.issubset(set(recommended["roles"]["research_vars"]))

        analysis = client.post("/api/analyze", json={
            "method_id": "ml_lr",
            "upload_id": upload_id,
            "use_demo": False,
            "params": {
                **recommended["params"],
                "split_ratio": "0.7",
                "random_state": 42,
                "threshold": 0.5,
            },
            "role_vars": recommended["roles"],
        })
        assert analysis.status_code == 200, analysis.text
        result = analysis.json()
        assert result["tables"] and result["charts"]

        deleted = client.delete(f"/api/upload/{upload_id}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["deleted"] >= 1
        assert not any(UPLOADS_DIR.glob(f"{upload_id}_*"))
    finally:
        for path in UPLOADS_DIR.glob(f"{upload_id}_*"):
            path.unlink()


def test_problem_methods_auto_roles_validate_and_run():
    role_cases = {
        "sensitivity_analysis": {
            "research_vars": ["sex"],
            "covar_vars": ["baseline_score1", "age", "bmi"],
            "outcome_vars": ["followup_score1"],
        },
        "markov_model": {
            "research_vars": ["state"],
            "covar_vars": ["subject_id", "month"],
            "outcome_vars": [],
        },
        "nhanes_analysis": {
            "research_vars": [],
            "covar_vars": ["survey_weight", "strata", "psu"],
            "outcome_vars": ["sbp"],
        },
        "ldsc": {
            "research_vars": ["trait"],
            "covar_vars": ["h2_se"],
            "outcome_vars": ["h2"],
        },
        "ml_lasso": {
            "research_vars": ["age", "bmi", "biomarker_001", "biomarker_002", "biomarker_003"],
            "covar_vars": ["sex"],
            "outcome_vars": ["outcome"],
        },
        "ml_ridge": {
            "research_vars": ["age", "bmi", "biomarker_001", "biomarker_002", "biomarker_003"],
            "covar_vars": ["sex"],
            "outcome_vars": ["renal_metabolic_index"],
        },
        "target_trial_emulation": {
            "research_vars": ["treatment"],
            "covar_vars": ["age", "sex", "bmi", "baseline_risk", "followup_months"],
            "outcome_vars": ["event_12m"],
        },
        "doubly_robust_aipw": {
            "research_vars": ["treatment"],
            "covar_vars": ["age", "sex", "bmi", "baseline_score", "comorbidity_count"],
            "outcome_vars": ["outcome"],
        },
        "ml_stacking": {
            "research_vars": ["age", "bmi", "sbp", "hba1c", "egfr", "nlr"],
            "covar_vars": ["sex"],
            "outcome_vars": ["adverse_event"],
        },
        "ml_mlp": {
            "research_vars": ["age", "bmi", "heart_rate", "spo2", "creatinine", "lactate"],
            "covar_vars": ["sex"],
            "outcome_vars": ["icu_transfer"],
        },
        "ml_elastic_net": {
            "research_vars": ["age", "bmi", "sbp", "hba1c", "egfr", "albumin"],
            "covar_vars": ["sex"],
            "outcome_vars": ["renal_decline_score"],
        },
        "ml_deepsurv": {
            "research_vars": ["age", "bmi", "albumin", "ldh", "nlr", "tumor_size_cm"],
            "covar_vars": ["time_months"],
            "outcome_vars": ["event"],
        },
        "ml_deephit": {
            "research_vars": ["age", "bmi", "egfr", "hba1c", "albumin", "nlr"],
            "covar_vars": ["time_months"],
            "outcome_vars": ["event_type"],
        },
        "marginal_structural_model": {
            "research_vars": ["treatment"],
            "covar_vars": ["age", "sex", "baseline_severity", "time_varying_severity", "month"],
            "outcome_vars": ["outcome"],
        },
    }
    for method_id, roles in role_cases.items():
        df = EXAMPLE_MAKERS[METHOD_CATALOG[method_id]["example_dataset"]]()
        ok, reason, derived = _validate_method_roles(df, method_id, roles)
        assert ok, f"{method_id}: auto roles rejected: {reason}; derived={derived}"
        result = run_analysis(MethodRequest(
            method_id=method_id,
            use_demo=True,
            dataset_name=METHOD_CATALOG[method_id]["example_dataset"],
            params={},
            role_vars=roles,
        ))
        assert result.get("tables"), f"{method_id}: no tables"
        assert result.get("charts"), f"{method_id}: no charts"


def test_recommend_roles_endpoint_for_each_method_example():
    client = TestClient(app)
    for method_id, cfg in METHOD_CATALOG.items():
        response = client.post("/api/recommend-roles", json={
            "method_id": method_id,
            "use_demo": True,
            "dataset_name": cfg["example_dataset"],
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["available"], f"{method_id}: {payload.get('reason')}"
        roles = payload["roles"]
        assert any(roles.values()), f"{method_id}: no recommended roles"


def test_frontend_uses_only_the_simple_four_step_flow():
    index = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    app_js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "app" / "static" / "styles.css").read_text(encoding="utf-8")
    assert "table-export-pdf-1" in index
    assert 'data-tab="method"' in index
    assert 'data-tab="data"' in index
    assert 'data-tab="variables"' in index
    assert 'data-tab="result"' in index
    assert 'data-tab="run"' not in index
    assert "综合工具" not in index
    assert "methodConfigs.js" not in index
    assert "upload.js" not in index
    assert "variableSelect.js" not in index
    assert "tableGenerator.js" not in index
    assert "dataPreview.js" not in index
    assert "cloud-topbar" not in index
    assert "breadcrumb-bar" not in index
    assert "side-status" not in index
    assert "muted-group" not in index
    assert "breadcrumbCurrent" not in app_js
    assert "cloud-topbar" not in styles
    assert "breadcrumb-bar" not in styles
    assert "side-status" not in styles
    assert "muted-group" not in styles
    assert "roleValidationMessage" in index
    assert "methodDetail" in index
    assert "methodNextBtn" in index
    assert "dataNextBtn" in index
    assert "variableMethodName" in index
    assert "uploadedFileNameInput" in index
    assert "cancelUploadBtn" in index
    assert "validateCurrentRoles" in app_js
    assert "recommendCurrentRoles" in app_js
    assert "/api/recommend-roles" in app_js
    assert "PARAM_CN_LABELS" in app_js
    assert "PARAM_EN_LABELS" in app_js
    assert "METHOD_FORMULAS" in app_js
    assert "METHOD_CONCEPTS" in app_js
    assert "METHOD_CONCEPT_TYPES" in app_js
    assert "renderMethodDetail" in app_js
    assert "renderFormulaBox" in app_js
    assert "renderConceptSvg" in app_js
    assert "bindMethodNextAction" in app_js
    assert "renderDataNextControl" in app_js
    assert "renderDiscussionFlow" in app_js
    assert "exportResultTable" in app_js
    assert "exportChartViaBackend" in app_js
    assert "/api/export/chart-image" in app_js
    assert "data-table-export" in app_js
    assert "setActiveTab('data');" in app_js
    assert "setActiveTab('variables');" in app_js
    assert "formula-math" in app_js
    assert "concept-svg" in app_js
    assert "2. 选择数据" not in index
    assert "3. 变量与参数" not in index
    assert "4. 结果" not in index
    assert "formula-scroll" in styles
    assert ".concept-svg" in styles
    assert ".data-next-bar" in styles
    assert ".discussion-flow" in styles
    assert ".discussion-card" in styles
    assert ".table-export-btn" in styles
    assert ".export-row .export-btn" in styles
    assert "#tab-method.tab-panel.active" in styles
    assert "#tab-method .method-detail-head" in styles
    assert "#tab-method .method-grid" in styles
    assert "#tab-method .method-card.active" in styles
    assert "method.icon" not in app_js
    assert "method-icon" not in app_js
    assert "method-icon" not in styles
    assert ".csv</em>" not in app_js
    assert "renderUploadFileControl" in app_js
    assert "deleteUploadedFile" in app_js
    assert "cancelUploadedDataset" in app_js
    assert "isVariableParam" in app_js
    assert ".filter(param => !isVariableParam(param))" in app_js
    assert "comprehensive_example" not in app_js
    assert "chartWidthInput" in app_js and "chartHeightInput" in app_js and "chartSizeLink" in app_js
    assert ".simple-mlhigh-app" in styles


if __name__ == "__main__":
    print("Running smoke tests...")
    test_method_catalog_is_method_first()
    print("  [PASS] Method catalog is method-first")
    test_example_registry_has_only_method_examples()
    print("  [PASS] Example registry has no comprehensive example")
    test_sample_generation_and_variable_summary()
    print("  [PASS] Example generation and variable summary")
    test_each_method_runs_with_its_own_example()
    print("  [PASS] Every method runs with its own example")
    test_uploaded_csv_http_workflow()
    print("  [PASS] Uploaded CSV HTTP workflow")
    test_problem_methods_auto_roles_validate_and_run()
    print("  [PASS] Problem-method auto roles validate and run")
    test_recommend_roles_endpoint_for_each_method_example()
    print("  [PASS] Recommendation endpoint works for each method example")
    test_frontend_uses_only_the_simple_four_step_flow()
    print("  [PASS] Frontend uses the simple four-step flow")
    print("All smoke tests passed!")
