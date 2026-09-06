"""Create the reproducible model bundle used by the Task 6 application."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "wrangled_customer_data.csv"
BUNDLE_PATH = Path(__file__).resolve().parent / "churn_model_bundle.joblib"
TARGET = "Churn_Status"


def main():
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=TARGET).copy()
    y = data[TARGET]
    categorical_columns = X.select_dtypes(include="object").columns.tolist()
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

    encoders = {}
    for column in categorical_columns:
        # Task 5 used LabelEncoder for every categorical predictor.
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column].fillna("").astype(str))
        encoders[column] = encoder

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        random_state=42, max_iter=1000, solver="lbfgs", C=0.001
    ).fit(scaler.transform(X_train), y_train)

    defaults = {}
    for column in numerical_columns:
        defaults[column] = float(data[column].median())
    for column in categorical_columns:
        values = data[column].fillna("").astype(str)
        defaults[column] = values.mode().iloc[0]

    bundle = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": X.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "defaults": defaults,
        # With a linear model, coef * (x - background mean) are exact SHAP values
        # in log-odds space for the independent/interventional explanation.
        "background_mean": scaler.transform(X_train).mean(axis=0),
        "data_source": DATA_PATH.name,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Saved model bundle to {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
