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


def explain_model(model: Any, X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
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
    feature_columns = [col for col in data.columns if col.lower() not in TARGET_CANDIDATES]
    if not feature_columns:
        raise ValueError("No feature columns found after excluding target candidates.")
    return data[feature_columns]


if __name__ == "__main__":
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {DEFAULT_MODEL_PATH}. Train and save the model before evaluation."
        )

    model = joblib.load(DEFAULT_MODEL_PATH)
    X = _load_feature_matrix(DEFAULT_DATA_PATH)
    explain_model(model, X)
