📊 Customer Churn Detection and Deployment
📝 Problem Statement
Customer churn — when clients stop using a company’s services — reduces future revenue and increases acquisition costs. The goal of this project was to analyze bank customer data and build a machine learning model that can predict churn risk, helping businesses take proactive retention actions.

📂 Dataset Details
Size: 10,000 customer records

Variables: 20 features including demographics, account tenure, balance, product ownership, transaction activity, complaints, support calls, satisfaction, and churn status.

Target: churn (binary: 1 = customer left, 0 = customer stayed)

Churn rate: 27.32% (about 27 out of every 100 customers left)

Data cleaning:

No duplicates found

Missing values in Education (10.26%) filled with most common category (Secondary)

Plausible ranges for age, tenure, balance, and complaint resolution

⚙️ Approach
Data Preparation

Standardized numerical inputs

Label encoding for categorical variables

Feature engineering: customer value score, engagement level, risk profile

Train/test split: 80:20 stratified

Model Training

Algorithms tested: Logistic Regression, Gradient Boosting, Random Forest, Decision Tree, Support Vector Machine

Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

Model Selection

Logistic Regression chosen for best ROC-AUC (0.6245) and interpretability

Explainability tools: SHAP, DALEX, partial dependence plots

Deployment

Packaged preprocessing + model into a reusable bundle

Streamlit app created for real-time predictions, probabilities, risk categories, and SHAP explanations

📈 Results
Best Model: Logistic Regression

Performance:

Accuracy: 72.4%

Precision: 33.3% (1 in 3 flagged customers actually churned)

Recall: 1.1% (model missed most churners)

ROC-AUC: 0.6245 (moderate ability to rank churn risk)

Key Churn Signals (SHAP importance):

Account tenure

Customer value score

Email subscription

Support calls

Number of complaints

Insights:

Churned customers had lower balances, shorter tenure, fewer products, more complaints, and higher support calls.

Engagement level was a strong predictor: low-engagement customers had the highest churn rate (28.3%).

Demographics (age, gender) were weak predictors compared to behavioral and service-related features.

✅ Conclusion
The project successfully delivered a deployable churn prediction tool. While the Logistic Regression model provides moderate ranking ability, its low recall means it should be used as a decision-support tool rather than a standalone retention trigger. Future work should focus on improving recall through threshold tuning, class balancing, and richer behavioral features.
