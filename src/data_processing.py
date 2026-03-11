from pathlib import Path

import pandas as pd
from scipy.stats.mstats import winsorize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "app_data.xlsx",
    PROJECT_ROOT / "data" / "app_data.csv",
]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
DROP_COLUMNS = [
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
    "Gynecological_Findings",
]
REQUIRED_COLUMNS = [
    "Age",
    "Sex",
    "Body_Temperature",
    "WBC_Count",
    "Neutrophil_Percentage",
    "CRP",
    "Lower_Right_Abd_Pain",
    "Diagnosis",
]


def find_raw_dataset() -> Path:
    for path in RAW_DATA_CANDIDATES:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "Raw dataset not found. Add data/app_data.xlsx or data/app_data.csv before running data_processing.py."
    )


def load_raw_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        data = pd.read_excel(path, engine="openpyxl")

    if data.empty:
        raise ValueError(f"The raw dataset at {path} is empty.")

    print(f"Raw dataset loaded successfully: {data.shape}")
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    missing_required = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in raw dataset: {', '.join(missing_required)}"
        )

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).copy()
    if cleaned.empty:
        raise ValueError("No rows remain after dropping records with missing required fields.")

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        cleaned[column] = winsorize(cleaned[column], limits=[0.05, 0.05])

    target = cleaned["Diagnosis"].astype(str).str.strip().str.lower()
    appendicitis = cleaned[target == "appendicitis"]
    non_appendicitis = cleaned[target != "appendicitis"]

    if appendicitis.empty or non_appendicitis.empty:
        raise ValueError("The dataset must contain at least two diagnosis classes.")

    max_len = max(len(appendicitis), len(non_appendicitis))
    appendicitis_balanced = appendicitis.sample(max_len, replace=True, random_state=42)
    non_appendicitis_balanced = non_appendicitis.sample(
        max_len, replace=True, random_state=42
    )

    processed = pd.concat([appendicitis_balanced, non_appendicitis_balanced])
    processed = processed.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Processed dataset shape:", processed.shape)
    print(processed["Diagnosis"].value_counts())
    return processed


def save_processed_dataset(data: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Processed dataset saved to {OUTPUT_PATH}")


def main() -> None:
    raw_path = find_raw_dataset()
    raw_data = load_raw_dataset(raw_path)
    processed_data = preprocess_data(raw_data)
    save_processed_dataset(processed_data)


if __name__ == "__main__":
    main()
