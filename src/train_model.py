"""
Script pour entraîner et comparer plusieurs modèles de Machine Learning
pour le diagnostic d'appendicite pédiatrique.
Utilise les données traitées par data_processing.py (Sara).
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import catboost as cb


def load_processed_data(filepath, target_col='Diagnosis'):
    """
    Charge les données déjà traitées par Sara (data_processing.py).
    Les données sont déjà nettoyées, équilibrées et sans outliers.

    Args:
        filepath   : chemin vers le fichier traité par Sara
        target_col : nom de la colonne cible

    Returns:
        X : features prêtes pour le modèle
        y : target encodée en 0/1
    """

    # ── 1. Chargement du fichier traité par Sara ───────────────
    extension = filepath.split('.')[-1].lower()

    if extension == 'csv':
        df = pd.read_csv(filepath)
    elif extension in ['xlsx', 'xls']:
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"❌ Format non supporté : {extension}")

    print(f"✅ Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # ── 2. Supprime les lignes où la cible est manquante ───────
    df = df.dropna(subset=[target_col])
    print(f"✅ Après nettoyage : {df.shape[0]} lignes")

    # ── 3. Sépare X et y ───────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # ── 4. Encode la target en 0/1 ────────────────────────────
    print(f"📊 Valeurs uniques target : {y.unique()}")

    # Conversion forcée en string puis mapping
    y = y.astype(str).str.strip().str.lower()

    mapping = {
        'appendicitis':    1,
        'no appendicitis': 0,
        'yes':             1,    'no':       0,
        'true':            1,    'false':    0,
        'positive':        1,    'negative': 0,
        '1':               1,    '0':        0,
    }

    y_mapped = y.map(mapping)

    if y_mapped.isnull().any():
        print("⚠️ Encodage automatique LabelEncoder")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    else:
        y = y_mapped

    y = y.fillna(0).astype(int)
    print(f"✅ Distribution target : {dict(y.value_counts())}")

    # ── 5. Encode les colonnes texte dans X ───────────────────
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    if len(cat_cols) > 0:
        print(f"🔄 Encodage de {len(cat_cols)} colonnes texte...")
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))

    # ── 6. Gère les valeurs manquantes restantes dans X ───────
    num_cols = X.select_dtypes(include=[np.number]).columns
    missing_count = int(X[num_cols].isnull().sum().sum())

    if missing_count > 0:
        print(f"🔄 Traitement de {missing_count} valeurs manquantes...")
        imputer = SimpleImputer(strategy='median')
        X[num_cols] = imputer.fit_transform(X[num_cols])

    print(f"✅ Données prêtes : X={X.shape}, y={y.shape}")

    return X, y


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entraîne les 4 modèles ML et évalue leurs performances.
    """

    models = {
        'SVM':           SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LightGBM':      lgb.LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost':      cb.CatBoostClassifier(random_state=42, verbose=0)
    }

    results        = []
    trained_models = {}

    print("\n🔄 Début de l'entraînement des modèles...")
    print("="*50)

    for name, model in models.items():
        print(f"\n⚙️  Entraînement : {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        metrics = {
            'Modèle':    name,
            'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'ROC-AUC':   round(roc_auc_score(y_test, y_proba), 4),
            'Precision': round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            'F1-Score':  round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4)
        }

        results.append(metrics)
        print(f"  ✅ Accuracy  : {metrics['Accuracy']}")
        print(f"  ✅ ROC-AUC   : {metrics['ROC-AUC']}")
        print(f"  ✅ Precision : {metrics['Precision']}")
        print(f"  ✅ Recall    : {metrics['Recall']}")
        print(f"  ✅ F1-Score  : {metrics['F1-Score']}")

    results_df = pd.DataFrame(results).set_index('Modèle')
    return trained_models, results_df


def save_best_model(trained_models, results_df, X):
    """
    Sélectionne et sauvegarde le meilleur modèle selon le ROC-AUC.
    """

    os.makedirs('models', exist_ok=True)

    best_model_name = results_df['ROC-AUC'].idxmax()
    best_model      = trained_models[best_model_name]

    print(f"\n🏆 Meilleur modèle : {best_model_name}")
    print(f"   ROC-AUC : {results_df.loc[best_model_name, 'ROC-AUC']}")

    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"💾 Modèle sauvegardé : models/best_model.pkl")

    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) \
                    else [f"feature_{i}" for i in range(X.shape[1])]

    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"💾 Features sauvegardées : models/feature_names.pkl")

    return best_model, best_model_name


def main():
    """
    Pipeline complet :
    1. Charge les données traitées par Sara
    2. Sépare train/test
    3. Entraîne les 4 modèles
    4. Compare les performances
    5. Sauvegarde le meilleur modèle
    """

    # ⚠️ Utilise le fichier traité par Sara
    data_path  = 'data/data_processed_and_balanced.xlsx'
    target_col = 'Diagnosis'

    # ── 1. Chargement ──────────────────────────────────────────
    print(f"\n📂 Chargement des données traitées : {data_path}")

    # Si Sara n'a pas encore pushé son fichier → fallback sur l'original
    if not os.path.exists(data_path):
        print(f"⚠️  Fichier de Sara introuvable → fallback : data/app_data.xlsx")
        data_path = 'data/app_data.xlsx'

    try:
        X, y = load_processed_data(data_path, target_col=target_col)
        print(f"✅ Données chargées : {X.shape[0]} patients, {X.shape[1]} features")
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return

    # ── 2. Vérification 2 classes ──────────────────────────────
    if len(y.unique()) < 2:
        print(f"❌ Erreur : la target n'a qu'une seule classe {y.unique()}")
        return

    # ── 3. Séparation Train / Test ─────────────────────────────
    print("\n✂️  Séparation Train (80%) / Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    print(f"   Train : {X_train.shape[0]} patients")
    print(f"   Test  : {X_test.shape[0]} patients")

    # ── 4. Entraînement et évaluation ─────────────────────────
    trained_models, results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    # ── 5. Tableau comparatif ──────────────────────────────────
    print("\n" + "="*60)
    print("📊 TABLEAU COMPARATIF DES 4 MODÈLES")
    print("="*60)
    print(results_df.to_string())
    print("="*60)

    # ── 6. Sauvegarde ──────────────────────────────────────────
    best_model, best_name = save_best_model(trained_models, results_df, X)

    print("\n✅ Pipeline terminé avec succès !")
    print(f"   Lance l'app avec : streamlit run app/app.py")


if __name__ == "__main__":
    main()
