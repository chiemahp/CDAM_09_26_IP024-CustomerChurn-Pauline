"""Streamlit deployment for the customer churn model from Tasks 4–5."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "churn_model_bundle.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


def risk_category(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "High"
    return "Critical"


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
    st.title("Customer Churn Risk Predictor")
    st.caption("Task 6 deployment • Logistic Regression model from Tasks 4–5")

    if not BUNDLE_PATH.exists():
        st.error("Model bundle not found. Run `python build_model.py` before starting the app.")
        st.stop()
    bundle = load_bundle()

    st.sidebar.header("Customer details")
    values = {}
    for column in bundle["feature_names"]:
        default = bundle["defaults"][column]
        label = column.replace("_", " ")
        if column in bundle["categorical_columns"]:
            choices = bundle["encoders"][column].classes_.tolist()
            index = choices.index(default) if default in choices else 0
            values[column] = st.sidebar.selectbox(label, choices, index=index)
        else:
            # Input bounds are deliberately permissive but prevent invalid negatives.
            min_value = 0.0 if column not in {"Customer_Value_Score"} else -10.0
            values[column] = st.sidebar.number_input(label, min_value=min_value, value=float(default))

    st.info("Enter customer details in the sidebar, then select **Predict churn risk**.")
    if not st.button("Predict churn risk", type="primary"):
        return

    encoded = pd.DataFrame([values], columns=bundle["feature_names"])
    for column, encoder in bundle["encoders"].items():
        encoded[column] = encoder.transform(encoded[column].astype(str))
    scaled = bundle["scaler"].transform(encoded)
    probability = float(bundle["model"].predict_proba(scaled)[0, 1])
    predicted_class = int(probability >= 0.50)
    confidence = max(probability, 1 - probability)

    result, confidence_col, category = st.columns(3)
    result.metric("Prediction", "Likely to churn" if predicted_class else "Likely to stay")
    confidence_col.metric("Churn probability", f"{probability:.1%}")
    category.metric("Risk category", risk_category(probability))
    st.caption(f"Model confidence: {confidence:.1%} (probability of the predicted class).")

    st.subheader("Why this prediction? (SHAP explanation)")
    # Exact linear SHAP values in log-odds space for this model and background.
    contributions = bundle["model"].coef_[0] * (scaled[0] - bundle["background_mean"])
    explanation = pd.DataFrame({
        "Feature": bundle["feature_names"],
        "Customer value": [values[name] for name in bundle["feature_names"]],
        "SHAP contribution (log-odds)": contributions,
    })
    explanation["Effect"] = np.where(
        explanation["SHAP contribution (log-odds)"] >= 0,
        "Increases churn risk",
        "Reduces churn risk",
    )
    explanation = explanation.reindex(
        explanation["SHAP contribution (log-odds)"].abs().sort_values(ascending=False).index
    )
    st.caption("Positive values push the prediction toward churn; negative values push it toward retention.")
    st.bar_chart(explanation.set_index("Feature")["SHAP contribution (log-odds)"].head(10))
    st.dataframe(explanation.head(10), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
