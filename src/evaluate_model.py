from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET_CANDIDATES = ("target", "label", "y", "diagnosis")


def prepare_features(
    X: pd.DataFrame, preprocessing: dict[str, Any]
) -> pd.DataFrame:
    processed = X.reindex(columns=preprocessing["feature_columns"]).copy()

    for column in preprocessing["numeric_columns"]:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed[column] = processed[column].fillna(
            preprocessing["numeric_fill_values"].get(column, 0)
        )

    for column in preprocessing["categorical_columns"]:
        fill_value = preprocessing["categorical_fill_values"][column]
        mapping = preprocessing["categorical_mappings"][column]
        filled = processed[column].astype(str).replace("nan", fill_value).fillna(fill_value)
        processed[column] = filled.map(mapping).fillna(-1).astype(int)

    return processed[preprocessing["feature_columns"]]


def _predict_positive_class(model: Any, preprocessing: dict[str, Any]):
    def predict_fn(values):
        frame = pd.DataFrame(values, columns=preprocessing["feature_columns"])
        frame = prepare_features(frame, preprocessing)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(frame)[:, 1]
        return model.predict(frame)

    return predict_fn


def explain_model(model: Any, preprocessing: dict[str, Any], X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if X.empty:
        raise ValueError("The dataset at data/processed/data.csv is empty.")

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    prepared = prepare_features(X, preprocessing)
    background = prepared.sample(min(len(prepared), 50), random_state=42)
    samples = prepared.sample(min(len(prepared), 100), random_state=42)
    explainer = shap.Explainer(
        _predict_positive_class(model, preprocessing), background
    )
    shap_values = explainer(samples)

    plt.figure()
    shap.summary_plot(shap_values, samples, show=False)
    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT_DIR / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def _load_feature_matrix(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Update DEFAULT_DATA_PATH or place the dataset there."
        )

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError(f"The dataset at {data_path} is empty.")
    feature_columns = [col for col in data.columns if col.lower() not in TARGET_CANDIDATES]
    if not feature_columns:
        raise ValueError("No feature columns found after excluding target candidates.")
    return data[feature_columns]


if __name__ == "__main__":
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {DEFAULT_MODEL_PATH}. Train and save the model before evaluation."
        )

    bundle = joblib.load(DEFAULT_MODEL_PATH)
    X = _load_feature_matrix(DEFAULT_DATA_PATH)
    explain_model(bundle["model"], bundle["preprocessing"], X)
