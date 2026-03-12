from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
SHAP_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "shap_summary.png"
SHAP_BAR_PATH = PROJECT_ROOT / "outputs" / "shap_bar.png"
TARGET_CANDIDATES = ("target", "label", "y", "diagnosis")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_columns() -> list[str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if data.empty:
        raise ValueError(f"Dataset at {DATA_PATH} is empty.")

    return [column for column in data.columns if column.lower() not in TARGET_CANDIDATES]


def predict(model_artifact, input_df: pd.DataFrame):
    model = model_artifact["model"] if isinstance(model_artifact, dict) else model_artifact
    return model.predict(input_df)


def render_shap_images() -> None:
    if SHAP_SUMMARY_PATH.exists():
        st.image(str(SHAP_SUMMARY_PATH), caption="Feature Importance (SHAP)")
    else:
        st.info("SHAP summary image not found.")

    if SHAP_BAR_PATH.exists():
        st.image(str(SHAP_BAR_PATH), caption="Global Feature Impact")
    else:
        st.info("SHAP bar image not found.")


def main() -> None:
    st.set_page_config(page_title="Pediatric Appendicitis Prediction App", layout="wide")
    st.title("Pediatric Appendicitis Prediction App")

    try:
        model_artifact = load_model()
        feature_columns = load_feature_columns()
    except Exception as exc:
        st.error(str(exc))
        return

    st.header("Model prediction")

    inputs = {}
    column_left, column_right = st.columns(2)
    for index, column_name in enumerate(feature_columns):
        target_column = column_left if index % 2 == 0 else column_right
        with target_column:
            inputs[column_name] = st.number_input(column_name, value=0.0)

    input_df = pd.DataFrame([inputs])

    if st.button("Predict"):
        try:
            prediction = predict(model_artifact, input_df)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.header("Model explanation (SHAP)")
    render_shap_images()


if __name__ == "__main__":
    main()
