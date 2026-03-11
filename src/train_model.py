from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
TARGET_CANDIDATES = ("target", "label", "y", "diagnosis")


def _get_target_column(data: pd.DataFrame) -> str:
    for column in data.columns:
        if column.lower() in TARGET_CANDIDATES:
            return column
    raise ValueError(
        "A target column is required in data.csv. Use one of: target, label, y, diagnosis."
    )


def train_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> Path:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError(f"The dataset at {data_path} is empty.")

    target_column = _get_target_column(data)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.empty:
        raise ValueError("No feature columns found after removing the target column.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


if __name__ == "__main__":
    saved_model_path = train_model()
    print(f"Model saved to {saved_model_path}")
