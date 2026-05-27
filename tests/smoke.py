"""Smoke test for Clinical Advanced Statistics & ML Platform core services."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from app.services.sample_service import EXAMPLE_MAKERS
from app.services.variable_service import classify_variables, summarize_dataset
from app.services.table_service import build_baseline_table
from app.services.io_service import get_example_datasets


def test_sample_generation():
    """All example datasets should generate without errors."""
    for name, fn in EXAMPLE_MAKERS.items():
        df = fn()
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) == "string":
                df[col] = df[col].astype(object)
        assert len(df) > 0, f"{name}: empty dataframe"
        assert len(df.columns) > 0, f"{name}: no columns"
    print(f"  [PASS] All {len(EXAMPLE_MAKERS)} example datasets generated")


def test_variable_classification():
    """Variable classification should return valid results."""
    df = EXAMPLE_MAKERS["propensity_score_example"]()
    types = classify_variables(df)
    summary = summarize_dataset(df, types)
    assert summary["sample_size"] > 0
    assert summary["variable_count"] > 0
    assert "continuous" in types
    assert "categorical" in types


def test_table_service():
    """Table generators should produce valid results."""
    df = EXAMPLE_MAKERS["propensity_score_example"]()
    baseline = build_baseline_table(df, "treatment")
    assert len(baseline["rows"]) > 0
    assert "columns" in baseline


def test_io_service():
    """Example dataset listing should work."""
    examples = get_example_datasets()
    assert len(examples) >= 20, f"Expected >=20 examples, got {len(examples)}"


if __name__ == "__main__":
    print("Running smoke tests...")
    test_sample_generation()
    test_variable_classification()
    print("  [PASS] Variable classification")
    test_table_service()
    print("  [PASS] Table service")
    test_io_service()
    print("  [PASS] IO service")
    print("All smoke tests passed!")
