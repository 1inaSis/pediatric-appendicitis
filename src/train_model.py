"""
Train and compare multiple ML models for pediatric appendicitis diagnosis.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
TARGET_CANDIDATES = ("diagnosis", "target", "label", "y")


def _get_target_column(data: pd.DataFrame) -> str:
    for column in data.columns:
        if column.lower() in TARGET_CANDIDATES:
            return column
    raise ValueError(
        "No target column found. Expected one of: diagnosis, target, label, y."
    )


def _encode_target(target: pd.Series) -> pd.Series:
    normalized = target.astype(str).str.strip().str.lower()
    mapping = {
        "appendicitis": 1,
        "no appendicitis": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "positive": 1,
        "negative": 0,
        "1": 1,
        "0": 0,
    }

    mapped = normalized.map(mapping)
    if mapped.isnull().any():
        encoder = LabelEncoder()
        return pd.Series(encoder.fit_transform(normalized), index=target.index)
    return mapped.astype(int)


def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    if DATA_PATH.stat().st_size == 0:
        print("Dataset is empty. Run data_processing.py first.")
        raise ValueError(
            "Dataset is empty. Run data_processing.py first. The dataset must be generated before training."
        )

    data = pd.read_csv("data/processed/data.csv")
    if data.empty:
        print("Dataset is empty. Run data_processing.py first.")
        raise ValueError(
            "Dataset is empty. Run data_processing.py first. The dataset must be generated before training."
        )

    print(f"Dataset loaded successfully: {data.shape}")

    target_col = _get_target_column(data)
    data = data.dropna(subset=[target_col]).copy()
    if data.empty:
        raise ValueError("The dataset has no rows left after dropping missing targets.")

    X = data.drop(columns=[target_col]).copy()
    y = _encode_target(data[target_col])
    return X, y


def prepare_features(
    X: pd.DataFrame,
    preprocessing: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    processed = X.copy()

    if preprocessing is None:
        feature_columns = processed.columns.tolist()
        numeric_columns = processed.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_columns = [
            column for column in feature_columns if column not in numeric_columns
        ]
        numeric_fill_values = processed[numeric_columns].median(numeric_only=True).to_dict()
        categorical_fill_values = {}
        categorical_mappings = {}

        for column in categorical_columns:
            filled = processed[column].astype(str).replace("nan", "__missing__").fillna("__missing__")
            mode = filled.mode(dropna=False)
            fill_value = mode.iloc[0] if not mode.empty else "__missing__"
            categories = sorted(filled.fillna(fill_value).unique().tolist())
            categorical_fill_values[column] = fill_value
            categorical_mappings[column] = {
                category: index for index, category in enumerate(categories)
            }

        preprocessing = {
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "numeric_fill_values": numeric_fill_values,
            "categorical_fill_values": categorical_fill_values,
            "categorical_mappings": categorical_mappings,
        }
    else:
        feature_columns = preprocessing["feature_columns"]
        numeric_columns = preprocessing["numeric_columns"]
        categorical_columns = preprocessing["categorical_columns"]
        numeric_fill_values = preprocessing["numeric_fill_values"]
        categorical_fill_values = preprocessing["categorical_fill_values"]
        categorical_mappings = preprocessing["categorical_mappings"]
        processed = processed.reindex(columns=feature_columns)

    for column in numeric_columns:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed[column] = processed[column].fillna(numeric_fill_values.get(column, 0))

    for column in categorical_columns:
        fill_value = categorical_fill_values[column]
        mapping = categorical_mappings[column]
        filled = processed[column].astype(str).replace("nan", fill_value).fillna(fill_value)
        processed[column] = filled.map(mapping).fillna(-1).astype(int)

    return processed[feature_columns], preprocessing


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    estimators = {
        "SVM": SVC(probability=True, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(random_state=42, verbose=0),
    }

    trained_models: dict[str, dict[str, object]] = {}
    results: list[dict[str, float | str]] = []
    X_train_prepared, preprocessing = prepare_features(X_train)
    X_test_prepared, _ = prepare_features(X_test, preprocessing)

    for name, estimator in estimators.items():
        estimator.fit(X_train_prepared, y_train)
        trained_models[name] = {
            "model": estimator,
            "preprocessing": preprocessing,
        }

        y_pred = estimator.predict(X_test_prepared)
        if hasattr(estimator, "predict_proba"):
            y_scores = estimator.predict_proba(X_test_prepared)[:, 1]
        else:
            y_scores = y_pred

        results.append(
            {
                "Model": name,
                "accuracy_score": accuracy_score(y_test, y_pred),
                "roc_auc_score": roc_auc_score(y_test, y_scores),
                "precision_score": precision_score(y_test, y_pred, zero_division=0),
                "recall_score": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results).set_index("Model").sort_values(
        by="roc_auc_score", ascending=False
    )
    return trained_models, results_df


def save_best_model(
    trained_models: dict[str, dict[str, object]], results_df: pd.DataFrame
) -> str:
    best_model_name = results_df["roc_auc_score"].idxmax()
    best_model = trained_models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print("Best model saved to models/best_model.pkl")

    return best_model_name


def main() -> None:
    X, y = load_processed_data()

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    trained_models, results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    print(results_df.round(4).to_string())

    best_model_name = save_best_model(trained_models, results_df)
    print(f"Best model by ROC-AUC: {best_model_name}")


if __name__ == "__main__":
    main()
