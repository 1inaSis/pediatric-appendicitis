"""
Modèles entraînés : SVM, Random Forest, LightGBM, CatBoost
Métrique de sélection : ROC-AUC
Auteur : Lina BENADDI (ML Engineer)
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


# ── Chemins des fichiers ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"      # Dataset traité par Sara
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"           # Meilleur modèle sauvegardé

# Noms possibles de la colonne cible dans le dataset
TARGET_CANDIDATES = ("diagnosis", "target", "label", "y")


def _get_target_column(data: pd.DataFrame) -> str:
    """
    Détecte automatiquement la colonne cible dans le DataFrame.
    Recherche parmi les noms candidats : diagnosis, target, label, y.

    Args:
        data : DataFrame contenant les données

    Returns:
        Nom de la colonne cible trouvée

    Raises:
        ValueError : si aucune colonne cible n'est trouvée
    """
    for column in data.columns:
        if column.lower() in TARGET_CANDIDATES:
            return column
    raise ValueError(
        "No target column found. Expected one of: diagnosis, target, label, y."
    )


def _encode_target(target: pd.Series) -> pd.Series:
    """
    Encode la colonne cible en valeurs binaires (0 ou 1).
    Convertit les valeurs texte en minuscules avant le mapping
    pour éviter les erreurs dues aux différences de casse.

    Mapping utilisé :
        'appendicitis' / 'yes' / 'true' / 'positive' / '1' → 1
        'no appendicitis' / 'no' / 'false' / 'negative' / '0' → 0

    Si des valeurs ne sont pas reconnues, utilise LabelEncoder automatiquement.

    Args:
        target : Series contenant les valeurs brutes de la cible

    Returns:
        Series encodée en entiers (0 ou 1)
    """
    # Normalisation : supprime les espaces et convertit en minuscules
    normalized = target.astype(str).str.strip().str.lower()

    # Mapping explicite pour tous les cas possibles
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

    # Si des valeurs ne sont pas dans le mapping → encodage automatique
    if mapped.isnull().any():
        encoder = LabelEncoder()
        return pd.Series(encoder.fit_transform(normalized), index=target.index)
    return mapped.astype(int)


def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge le dataset traité par Sara depuis data/processed/data.csv.
    Vérifie que le fichier existe et n'est pas vide avant de charger.
    Détecte automatiquement la colonne cible et l'encode en 0/1.

    Returns:
        X : DataFrame des features
        y : Series de la cible encodée en 0/1

    Raises:
        FileNotFoundError : si le fichier data.csv n'existe pas
        ValueError : si le fichier est vide ou ne contient plus de lignes après nettoyage
    """
    # Vérifie que le fichier existe et n'est pas vide
    if not DATA_PATH.exists():
        print("Dataset missing or empty. Run data_processing.py first.")
        raise FileNotFoundError(
            "Dataset missing or empty. Run data_processing.py first."
        )
    if DATA_PATH.stat().st_size == 0:
        print("Dataset missing or empty. Run data_processing.py first.")
        raise ValueError("Dataset missing or empty. Run data_processing.py first.")

    data = pd.read_csv("data/processed/data.csv")

    # Vérifie que le DataFrame n'est pas vide après lecture
    if data.empty:
        print("Dataset missing or empty. Run data_processing.py first.")
        raise ValueError("Dataset missing or empty. Run data_processing.py first.")

    print(f"Dataset loaded successfully: {data.shape}")

    # Détecte et supprime les lignes où la cible est manquante
    target_col = _get_target_column(data)
    data = data.dropna(subset=[target_col]).copy()
    if data.empty:
        raise ValueError("The dataset has no rows left after dropping missing targets.")

    # Sépare les features et la cible
    X = data.drop(columns=[target_col]).copy()
    y = _encode_target(data[target_col])
    return X, y


def prepare_features(
    X: pd.DataFrame,
    preprocessing: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Prépare les features pour l'entraînement ou la prédiction.
    - Impute les valeurs manquantes (médiane pour numériques, mode pour catégorielles)
    - Encode les colonnes catégorielles avec un mapping entier
    - Si preprocessing=None, calcule les paramètres depuis X (mode entraînement)
    - Si preprocessing fourni, applique les mêmes transformations (mode test/prédiction)

    Args:
        X            : DataFrame des features brutes
        preprocessing: dictionnaire des paramètres de transformation (optionnel)

    Returns:
        processed    : DataFrame transformé prêt pour le modèle
        preprocessing: dictionnaire des paramètres utilisés (pour réutilisation)
    """
    processed = X.copy()

    if preprocessing is None:
        # ── Mode entraînement : calcule les paramètres depuis X ───────────
        feature_columns = processed.columns.tolist()
        numeric_columns = processed.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_columns = [
            column for column in feature_columns if column not in numeric_columns
        ]

        # Calcule les valeurs de remplacement pour les numériques (médiane)
        numeric_fill_values = processed[numeric_columns].median(numeric_only=True).to_dict()
        categorical_fill_values = {}
        categorical_mappings = {}

        # Calcule les valeurs de remplacement et mappings pour les catégorielles
        for column in categorical_columns:
            filled = processed[column].astype(str).replace("nan", "__missing__").fillna("__missing__")
            mode = filled.mode(dropna=False)
            fill_value = mode.iloc[0] if not mode.empty else "__missing__"
            categories = sorted(filled.fillna(fill_value).unique().tolist())
            categorical_fill_values[column] = fill_value
            categorical_mappings[column] = {
                category: index for index, category in enumerate(categories)
            }

        # Sauvegarde tous les paramètres pour réutilisation sur le test set
        preprocessing = {
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "numeric_fill_values": numeric_fill_values,
            "categorical_fill_values": categorical_fill_values,
            "categorical_mappings": categorical_mappings,
        }
    else:
        # ── Mode test/prédiction : utilise les paramètres existants ───────
        feature_columns = preprocessing["feature_columns"]
        numeric_columns = preprocessing["numeric_columns"]
        categorical_columns = preprocessing["categorical_columns"]
        numeric_fill_values = preprocessing["numeric_fill_values"]
        categorical_fill_values = preprocessing["categorical_fill_values"]
        categorical_mappings = preprocessing["categorical_mappings"]

        # Réaligne les colonnes sur celles vues lors de l'entraînement
        processed = processed.reindex(columns=feature_columns)

    # ── Imputation des valeurs manquantes numériques (médiane) ────────────
    for column in numeric_columns:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed[column] = processed[column].fillna(numeric_fill_values.get(column, 0))

    # ── Encodage des colonnes catégorielles (mapping entier) ──────────────
    for column in categorical_columns:
        fill_value = categorical_fill_values[column]
        mapping = categorical_mappings[column]
        filled = processed[column].astype(str).replace("nan", fill_value).fillna(fill_value)
        # Les valeurs inconnues reçoivent -1
        processed[column] = filled.map(mapping).fillna(-1).astype(int)

    return processed[feature_columns], preprocessing


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    """
    Entraîne et évalue les 4 modèles ML sur les données d'entraînement et de test.
    Calcule les métriques : Accuracy, ROC-AUC, Precision, Recall, F1-Score.

    Args:
        X_train : features d'entraînement
        X_test  : features de test
        y_train : cible d'entraînement
        y_test  : cible de test

    Returns:
        trained_models : dictionnaire {nom_modèle: {model, preprocessing}}
        results_df     : DataFrame des métriques trié par ROC-AUC décroissant
    """
    # Définition des 4 modèles à comparer
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

    # Prépare les features (calcule les paramètres sur X_train, les applique sur X_test)
    X_train_prepared, preprocessing = prepare_features(X_train)
    X_test_prepared, _ = prepare_features(X_test, preprocessing)

    for name, estimator in estimators.items():
        # Entraînement du modèle
        estimator.fit(X_train_prepared, y_train)

        # Sauvegarde du modèle avec ses paramètres de preprocessing
        trained_models[name] = {
            "model": estimator,
            "preprocessing": preprocessing,
        }

        # Prédictions sur le set de test
        y_pred = estimator.predict(X_test_prepared)

        # Probabilités pour le calcul du ROC-AUC
        if hasattr(estimator, "predict_proba"):
            y_scores = estimator.predict_proba(X_test_prepared)[:, 1]
        else:
            y_scores = y_pred

        # Calcul et stockage des métriques
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

    # Trie les résultats par ROC-AUC décroissant
    results_df = pd.DataFrame(results).set_index("Model").sort_values(
        by="roc_auc_score", ascending=False
    )
    return trained_models, results_df


def save_best_model(
    trained_models: dict[str, dict[str, object]], results_df: pd.DataFrame
) -> str:
    """
    Sélectionne et sauvegarde le meilleur modèle selon le ROC-AUC.
    Crée le dossier models/ si nécessaire.
    Sauvegarde le modèle ET ses paramètres de preprocessing dans best_model.pkl.

    Args:
        trained_models : dictionnaire des modèles entraînés
        results_df     : DataFrame des métriques

    Returns:
        Nom du meilleur modèle sélectionné
    """
    # Sélectionne le modèle avec le ROC-AUC le plus élevé
    best_model_name = results_df["roc_auc_score"].idxmax()
    best_model = trained_models[best_model_name]

    # Crée le dossier models/ si nécessaire et sauvegarde
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print("Best model saved to models/best_model.pkl")

    return best_model_name


def main() -> None:
    """
    Pipeline complet d'entraînement :
    1. Charge les données traitées par Sara
    2. Vérifie que la target contient au moins 2 classes
    3. Sépare en train (80%) / test (20%) avec stratification
    4. Entraîne et évalue les 4 modèles
    5. Affiche le tableau comparatif des métriques
    6. Sauvegarde le meilleur modèle dans models/best_model.pkl
    """
    # ── 1. Chargement des données ──────────────────────────────────────────
    X, y = load_processed_data()

    # ── 2. Vérification que la target contient 2 classes ──────────────────
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    # ── 3. Séparation Train (80%) / Test (20%) avec stratification ─────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 4. Entraînement et évaluation des 4 modèles ────────────────────────
    trained_models, results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    # ── 5. Affichage du tableau comparatif ────────────────────────────────
    print(results_df.round(4).to_string())

    # ── 6. Sauvegarde du meilleur modèle ──────────────────────────────────
    best_model_name = save_best_model(trained_models, results_df)
    print(f"Best model by ROC-AUC: {best_model_name}")


if __name__ == "__main__":
    main()