import pandas as pd
import numpy as np

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

#Sauvegarder la data traitée et équilibrée
data_processed_and_balanced.to_excel("data/data_processed_and_balanced.xlsx", index=False, engine="openpyxl")
