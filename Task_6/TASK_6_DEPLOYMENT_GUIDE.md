# Task 6: Customer Churn Model Deployment

This deployment packages the Task 4–5 Logistic Regression churn model as an interactive Streamlit application. It accepts all 18 predictors used by the final model, returns the churn prediction, probability, confidence, and risk category, and provides an individual SHAP explanation.

## Run locally

From the project folder, install the application dependencies and build the trained-model artifact:

```powershell
python -m pip install -r deployment/requirements.txt
python deployment/build_model.py
streamlit run Task_6/app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`). Enter customer values in the sidebar and select **Predict churn risk**. The result area shows the predicted class, churn probability, confidence, and a four-level risk category. The explanation chart ranks the ten features with the largest SHAP contribution; positive values increase churn risk.

## Publish with Streamlit Community Cloud

1. Create a GitHub repository containing this project, including the `deployment` directory and `wrangled_customer_data.csv`.
2. Run `python deployment/build_model.py` and commit the generated `deployment/churn_model_bundle.joblib` file, or configure the host's build command to run it before starting the app.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), select **New app**, choose the repository and branch, and set the main file path to `deployment/app.py`.
4. Deployed app: (https://r2wfipdar6ppynqshdz7kc.streamlit.app/)

## Live application link

Streamlit App:(https://r2wfipdar6ppynqshdz7kc.streamlit.app/)

## Notes

- The bundle intentionally reproduces Task 5's preprocessing: label encoding, the same stratified split (`random_state=42`), standard scaling, and Logistic Regression settings.
- `ID` was included in the previous final model and remains available for faithful reproduction. It should be removed and the model retrained before a real production use because an identifier is not a meaningful business predictor.
- The Task 5 model had very low churn recall; treat its outputs as decision support and monitor/retrain it before operational use.
