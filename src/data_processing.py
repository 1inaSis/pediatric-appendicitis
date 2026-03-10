import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def prepare_data(filepath):
    """
    Charge et prépare les données pour le modèle ML.
    """
    df = pd.read_excel(filepath, engine='openpyxl')
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Sépare features et target
    # ⚠️ Adapte 'diagnosis' selon le vrai nom de la colonne cible
    target_col = 'Diagnosis'
    # Supprime les lignes où la cible est manquante
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode la target si c'est du texte
    # Encode manuellement les valeurs texte en 0/1
    y = y.map({'appendicitis': 1, 'no appendicitis': 0})

    # Gère les valeurs manquantes
    num_cols = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # Encode les colonnes texte
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    print(f"✅ Données prêtes : X={X.shape}, y={y.shape}")
    return X, y


def optimize_memory(df):
    """
    Optimise la mémoire du DataFrame.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df