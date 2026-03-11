"""
Script pour évaluer le meilleur modèle ML sauvegardé.
- Matrice de confusion
- Courbe ROC
- Classification report
- (SHAP - partie de P4)

Auteur : Person 3 (ML Engineer)
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour éviter les erreurs d'affichage

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ══════════════════════════════════════════════════════
#  1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════

def load_data(target_col='Diagnosis'):
    """
    Charge les données traitées par Sara si disponibles,
    sinon utilise le dataset original.
    """

    processed_path = 'data/data_processed_and_balanced.xlsx'
    original_path  = 'data/app_data.xlsx'

    if os.path.exists(processed_path):
        print(f"📂 Chargement : {processed_path}")
        df = pd.read_excel(processed_path, engine='openpyxl')
    else:
        print(f"⚠️  Fichier traité introuvable → fallback : {original_path}")
        df = pd.read_excel(original_path, engine='openpyxl')

    print(f"✅ Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Supprime les lignes où la cible est manquante
    df = df.dropna(subset=[target_col])

    # Sépare X et y
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Encode la target en 0/1
    y = y.astype(str).str.strip().str.lower()
    mapping = {
        'appendicitis':    1,
        'no appendicitis': 0,
        'yes': 1, 'no': 0,
        '1': 1,   '0': 0,
    }
    y = y.map(mapping).fillna(0).astype(int)

    # Encode les colonnes texte
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Gère les valeurs manquantes
    num_cols = X.select_dtypes(include=[np.number]).columns
    if X[num_cols].isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='median')
        X[num_cols] = imputer.fit_transform(X[num_cols])

    print(f"✅ Données prêtes : X={X.shape}, y={y.shape}")
    print(f"   Distribution : {dict(y.value_counts())}")

    return X, y


# ══════════════════════════════════════════════════════
#  2. MATRICE DE CONFUSION
# ══════════════════════════════════════════════════════

def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Génère et sauvegarde la matrice de confusion.
    """

    os.makedirs('results', exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['No Appendicitis', 'Appendicitis']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False)

    ax.set_title(
        f'Confusion Matrix — {model_name}',
        fontsize=14, fontweight='bold', pad=15
    )

    plt.tight_layout()
    path = 'results/confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Matrice de confusion sauvegardée : {path}")

    # Affiche les détails
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📊 Détails matrice de confusion :")
    print(f"   ✅ Vrais Positifs  (TP) : {tp}  — appendicite détectée correctement")
    print(f"   ✅ Vrais Négatifs  (TN) : {tn}  — non-appendicite détectée correctement")
    print(f"   ⚠️  Faux Positifs  (FP) : {fp}  — fausse alarme")
    print(f"   ❌ Faux Négatifs  (FN) : {fn}  — appendicite manquée (dangereux !)")


# ══════════════════════════════════════════════════════
#  3. COURBE ROC
# ══════════════════════════════════════════════════════

def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Génère et sauvegarde la courbe ROC.
    """

    os.makedirs('results', exist_ok=True)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr,
            color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1],
            color='navy', lw=1,
            linestyle='--', label='Random classifier')

    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(
        f'ROC Curve — {model_name}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'results/roc_curve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Courbe ROC sauvegardée : {path}")
    print(f"   ROC-AUC : {roc_auc:.4f}")

    return roc_auc


# ══════════════════════════════════════════════════════
#  4. CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════

def print_classification_report(y_test, y_pred):
    """
    Affiche le rapport de classification détaillé.
    """

    print("\n📋 CLASSIFICATION REPORT")
    print("="*55)
    report = classification_report(
        y_test, y_pred,
        target_names=['No Appendicitis', 'Appendicitis']
    )
    print(report)
    print("="*55)

    # Sauvegarde le rapport dans un fichier texte
    os.makedirs('results', exist_ok=True)
    with open('results/classification_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*55 + "\n")
        f.write(report)

    print("💾 Rapport sauvegardé : results/classification_report.txt")


# ══════════════════════════════════════════════════════
#  5. SECTION SHAP — À COMPLÉTER PAR P4
# ══════════════════════════════════════════════════════

def plot_shap(model, X_test, feature_names):
    """
    Génère les graphiques SHAP pour expliquer les prédictions.
    ⚠️ Cette fonction sera complétée par P4 (SHAP + UI)
    """
    print("\n🔍 SHAP analysis — à compléter par P4...")
    pass


# ══════════════════════════════════════════════════════
#  6. MAIN
# ══════════════════════════════════════════════════════

def main():
    """
    Pipeline complet d'évaluation :
    1. Charge les données
    2. Charge le meilleur modèle sauvegardé
    3. Génère la matrice de confusion
    4. Génère la courbe ROC
    5. Affiche le classification report
    """

    print("\n" + "="*55)
    print("📊 ÉVALUATION DU MEILLEUR MODÈLE")
    print("="*55)

    # ── 1. Charge les données ──────────────────────────
    X, y = load_data(target_col='Diagnosis')

    # ── 2. Séparation Train / Test ─────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    print(f"\n✂️  Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    # ── 3. Charge le meilleur modèle ──────────────────
    model_path = 'models/best_model.pkl'
    names_path = 'models/feature_names.pkl'

    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        print("   Lance d'abord : python src/train_model.py")
        return

    model         = joblib.load(model_path)
    feature_names = joblib.load(names_path)
    model_name    = type(model).__name__

    print(f"\n✅ Modèle chargé : {model_name}")

    # ── 4. Prédictions ────────────────────────────────
    y_pred = model.predict(X_test)

    # ── 5. Matrice de confusion ───────────────────────
    print("\n📊 Génération matrice de confusion...")
    plot_confusion_matrix(y_test, y_pred, model_name)

    # ── 6. Courbe ROC ─────────────────────────────────
    print("\n📈 Génération courbe ROC...")
    roc_auc = plot_roc_curve(model, X_test, y_test, model_name)

    # ── 7. Classification report ──────────────────────
    print_classification_report(y_test, y_pred)

    # ── 8. SHAP (P4) ──────────────────────────────────
    plot_shap(model, X_test, feature_names)

    print("\n✅ Évaluation terminée !")
    print("   Résultats sauvegardés dans : results/")
    print("   ├── confusion_matrix.png")
    print("   ├── roc_curve.png")
    print("   └── classification_report.txt")


if __name__ == "__main__":
    main()