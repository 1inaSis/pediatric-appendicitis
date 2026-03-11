import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# Charger le dataset Excel contenant les données cliniques des patients
data=pd.read_excel("data/app_data.xlsx",engine="openpyxl")

# Indiquer les variables avec leur nombre de valeurs manquantes
print(data.isnull().sum())

# Supprimer les colonnes qui contiennent trop de valeurs manquantes ou qui ne sont pas suffisamment fiables pour l'analyse
data = data.drop(columns=[
"Segmented_Neutrophils",
"Appendix_Wall_Layers",
"Target_Sign",
"Appendicolith",
"Perfusion",
"Perforation",
"Surrounding_Tissue_Reaction",
"Appendicular_Abscess",
"Abscess_Location",
"Pathological_Lymph_Nodes",
"Lymph_Nodes_Location",
"Bowel_Wall_Thickening",
"Conglomerate_of_Bowel_Loops",
"Ileus",
"Coprostasis",
"Meteorism",
"Enteritis",
"Gynecological_Findings"
])

# Supprimer les lignes (patients) pour lesquelles certaines variables essentielles au diagnostic de l'appendicite sont manquantes
data = data.dropna(subset=[
"Age",
"Sex",
"Body_Temperature",
"WBC_Count",
"Neutrophil_Percentage",
"CRP",
"Lower_Right_Abd_Pain",
"Diagnosis"
])

# Sélectionner les colonnes numériques
num_cols = data.select_dtypes(include=['float64','int64']).columns

# appliquer winsorization (ex: couper les 5% les plus extrêmes) pour éliminer les valeurs outliers
for col in num_cols:
    data[col] = winsorize(data[col], limits=[0.05, 0.05])

# Séparer les variables explicatives  de la variable cible

# Variables contient toutes les caractéristiques cliniques utilisées pour prédire la maladie
Variables = data.drop("Diagnosis", axis=1)

#  Contenir la variable cible que le modèle devra prédire, ici le diagnostic d'appendicite
ValeurCible = data["Diagnosis"]

# Vérifier combien d'éléments pour les deux catégories du diagnostic: appendicite et non appendicite
print(ValeurCible.value_counts())
# On trouve qu'on a 398 pour appendicite et 274 pour non appendicite : c'et un déséquilibre à régler

#Séparation des deux catégories
appendicitis = data[ValeurCible == 'appendicitis']
non_appendicitis = data[ValeurCible != 'appendicitis']

# On opte pour du sur-échantillonage pour avoir la même taille de catégories
max_len = max(len(appendicitis), len(non_appendicitis))
appendicitis_balanced = appendicitis.sample(max_len, replace=True, random_state=42)
non_appendicitis_balanced = non_appendicitis.sample(max_len, replace=True, random_state=42)
data_processed_and_balanced = pd.concat([appendicitis_balanced, non_appendicitis_balanced])
data_processed_and_balanced = data_processed_and_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Vérifier la nouvelle distribution
print(data_processed_and_balanced['Diagnosis'].value_counts())

def optimize_memory(df):
    """
    Cette fonction réduit l'utilisation mémoire d'un DataFrame
    en convertissant les types numériques vers des types plus petits.

    Par exemple :
    int64  → int32 ou int16
    float64 → float32

    
    """

    # Calcul de la mémoire utilisée avant optimisation (en MB)
    start_mem = df.memory_usage().sum() / 1024**2

    # sélectionner uniquement les colonnes numériques
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Parcours de toutes les colonnes numériques du DataFrame
    for col in num_cols:
            col_type = df[col].dtype
    
            # Trouver la valeur minimale et maximale de la colonne
            c_min = df[col].min()
            c_max = df[col].max()

            # Traitement des colonnes entières
            if str(col_type)[:3] == "int":

                # Si les valeurs tiennent dans int8
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

                # Sinon vérifier int16
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)

                # Sinon vérifier int32
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            # Traitement des colonnes flottantes
            else:
                # Conversion vers float32 pour réduire la mémoire
                df[col] = df[col].astype(np.float32)

    # Calcul de la mémoire utilisée après optimisation
    end_mem = df.memory_usage().sum() / 1024**2

    # Affichage des résultats
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    print(f"Memory usage after optimization: {end_mem:.2f} MB")

    # Retourner le DataFrame optimisé
    return df

#Optimisation de la data traitée
data_processed_and_balanced = optimize_memory(data_processed_and_balanced)


#Sauvegarder la data optimisée en mémoire, traitée et équilibrée
data_processed_and_balanced.to_excel("data/data_processed_and_balanced.xlsx", index=False, engine="openpyxl")
