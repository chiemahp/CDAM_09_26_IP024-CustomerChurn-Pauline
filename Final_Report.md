# Final Report: Customer Churn Detection and Deployment

## Executive Summary

This project analysed 10,000 bank-customer records to understand and predict customer churn. The data was cleaned, explored, transformed into business-focused features, modelled with five machine-learning algorithms, explained with SHAP and DALEX, and packaged in a Streamlit prediction application.

Logistic Regression was selected as the final model because it achieved the strongest ability to rank churners ahead of non-churners (ROC-AUC **0.6245**) among the models tested. In plain English, when the model compares one customer who churned with one who did not, it places the churner higher about 62 times out of 100—better than chance, but only moderately effective. Its test accuracy was **72.4%**, meaning it gave the correct stay/churn label for roughly 72 of every 100 customers. However, it detected only **1.1%** of real churners. Therefore, the model is suitable as a decision-support and exploratory tool, but it is not ready to be the sole trigger for retention action.

The strongest model signals were account tenure, customer value score, email subscription, support calls, and number of complaints. A Streamlit application was created to provide real-time predictions, probabilities, confidence, risk categories, and an explanation of each prediction.

## Introduction

Customer churn reduces future revenue and increases the cost of replacing customers. The aim of this work was to turn customer account, transaction, service, and engagement data into practical evidence that can support earlier retention action.

The source dataset contains 10,000 customer records and 20 variables, including demographics, account tenure, balance, product ownership, transaction activity, complaints, support calls, satisfaction, and churn status. Churn is a binary outcome: a value of 1 indicates that a customer left, and 0 indicates that the customer remained.

## Objectives

1. Clean and validate the customer-churn dataset.
2. Explore the customer base and identify patterns associated with churn.
3. Create useful features for modelling customer value, engagement, and risk.
4. Train and compare several machine-learning models for churn prediction.
5. Explain the final model's overall behaviour and individual predictions.
6. Deliver an interface that allows users to make and interpret live churn predictions.

## Methodology

### Data preparation

The dataset was checked for duplicates, missing values, invalid ranges, and inconsistent binary fields. No duplicate records were found. Education level had **1,026 missing values (10.26%)**; these were filled with the most common category, *Secondary*, so records were retained instead of discarded. All 10,000 records remained available for analysis.

Customer age ranged from 18 to 70 years, tenure from 1 to 180 months, average balance from $500 to $1,644,080, and complaint-resolution rate from 0.50 to 1.00. These ranges were judged plausible, so no observations were removed as invalid.

Feature engineering added a customer value score (a 0–1 combined measure of balance, transaction frequency, and tenure), engagement level, and risk profile. Categorical variables were label encoded for modelling. The data were split into training and test sets using an 80:20 stratified split, which kept the churn proportion similar in both sets. Numerical inputs were standardised so variables with large units, such as balance, did not dominate smaller-scale variables.

### Modelling and evaluation

Five classifiers were trained and compared: Logistic Regression, Gradient Boosting, Random Forest, Decision Tree, and Support Vector Machine. Evaluation used accuracy, precision, recall, F1-score, and ROC-AUC.

- **Accuracy** is the share of all predictions that were correct.
- **Precision** is the share of customers predicted to churn who actually churned.
- **Recall** is the share of actual churners that the model successfully found.
- **F1-score** balances precision and recall; it falls when either is poor.
- **ROC-AUC** measures how well the model ranks churn risk across all thresholds. A value of 0.50 is no better than random ordering, while 1.00 is perfect.

Logistic Regression was selected because it achieved the highest ROC-AUC and is straightforward to explain. SHAP global importance plots, SHAP summary plots, partial-dependence plots, and a DALEX break-down plot were used to make model behaviour understandable.

### Deployment

The final preprocessing steps, encoders, scaler, and Logistic Regression model were packaged into a reusable model bundle. A Streamlit application accepts the 18 model predictors and returns the predicted class, churn probability, confidence, four-level risk category, and the ten largest individual SHAP contributions. The source files and publishing instructions are available in [Task_6](Task_6/TASK_6_DEPLOYMENT_GUIDE.md).

## Results

### Data and exploratory findings

The observed churn rate was **27.32%**. Put simply, about 27 out of every 100 customers had churned, while about 73 remained active. This imbalance means a model can appear accurate by predicting that many customers will stay, which is why recall and ROC-AUC were assessed alongside accuracy.

The average account balance was **$150,108**, while the median was **$103,651**. Because the average is higher than the middle value, a relatively small number of very high-balance customers pull the average upward. Average tenure was **90.8 months**, or about **7.6 years**.

Churned customers had lower average balance (**$136,000** versus **$156,000**), shorter tenure (**77** versus **96 months**), fewer products (**1.8** versus **2.1**), and lower transaction frequency (**14.6** versus **15.2**). Their average support calls were higher (**1.5** versus **0.7**) and their satisfaction rate was lower (**0.71** versus **0.75**). In plain English, customers who leave tend to be less established, less engaged, less satisfied, and more likely to need help.

The low-engagement segment represented **37.7%** of customers and had the highest churn rate at **28.3%**. This means more than one in three customers are lightly engaged and nearly three in ten in this segment leave, making engagement improvement a worthwhile retention opportunity. Gender and age showed little practical value as churn indicators; the evidence suggests churn is driven more by customer experience and behaviour than demographics.

### Model comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72.4% | 33.3% | 1.1% | 2.1% | 0.6245 |
| Gradient Boosting | 72.7% | 50.0% | 0.2% | 0.4% | 0.6155 |
| Random Forest | 72.6% | 33.3% | 0.4% | 0.7% | 0.6111 |
| Decision Tree | 71.9% | 23.3% | 1.3% | 2.4% | 0.5964 |
| SVM | 72.7% | 0.0% | 0.0% | 0.0% | 0.4492 |

Although Gradient Boosting and SVM matched or slightly exceeded Logistic Regression's accuracy, their recall was almost zero. In practical terms, they mostly classified customers as likely to stay and rarely identified actual churners. Logistic Regression had the best ROC-AUC (**0.6245**), so it provided the most useful overall ranking of churn risk. Its precision of **33.3%** means roughly one in three customers it flagged actually churned. Its recall of **1.1%** means it missed nearly all customers who did churn, which is the primary limitation of the work.

The final model identified 18 customers as at risk in the test set, with 6 correct churn identifications and 12 false alarms. This may help target a very small outreach list, but it cannot be used as a comprehensive churn-detection system without improvement.

### Explainable AI findings

The five highest average SHAP importance values were:

| Feature | Mean absolute SHAP value | Plain-English interpretation |
|---|---:|---|
| Account tenure (months) | 0.1456 | This was the strongest driver of changes in model predictions across customers. |
| Customer value score | 0.1221 | The combined value measure materially changed predicted risk. |
| Email subscription | 0.0705 | Email engagement helped distinguish customers with different churn risk. |
| Support calls | 0.0464 | Frequent support contact was a meaningful warning signal. |
| Number of complaints | 0.0411 | More complaints contributed to higher predicted churn risk. |

SHAP values show how much each feature moves an individual prediction away from the model's baseline. A larger absolute value means that feature had more influence; it does not by itself prove that the feature caused churn. The individual example examined in Task 5 had a churn probability of **50.55%**, meaning the model regarded that customer as almost equally likely to churn as to stay. Account tenure, risk profile, customer value score, complaints, and support calls were the largest influences for that customer.

## Conclusion

The project successfully converted a raw churn dataset into a cleaned, analysed, modelled, explained, and deployable solution. The analysis consistently points to service friction, low engagement, complaints, and weaker customer relationships as important churn signals. The final Logistic Regression model offers modest risk-ranking ability and is transparent enough for stakeholders to interpret.

The 72.4% accuracy should not be interpreted as strong churn detection. Because only 1.1% of actual churners were found, the current model misses most customers who leave. Its best role is to support exploratory analysis, help prioritise a small high-risk list, and guide retention strategy while a stronger model and decision threshold are developed.

## Recommendations

1. **Improve first-contact resolution.** Proactively review customers with repeated support calls or complaints. These signals were consistently associated with elevated churn risk.
2. **Build engagement programmes.** Target low-engagement customers with relevant digital-banking, transaction, and product-adoption campaigns. This segment is large and has the highest observed churn rate.
3. **Use targeted retention offers.** Combine churn risk with customer value and service history, rather than treating every predicted risk the same. High-value customers with complaints should receive fast, personalised follow-up.
4. **Optimise the decision threshold.** The default 0.50 threshold produces very low recall. Test lower thresholds and quantify the trade-off between finding more churners and contacting more customers who would have stayed.
5. **Improve and retrain the model.** Add time-based behaviour, recent changes in support activity, offer history, and reason-for-leaving data. Evaluate class weighting, resampling, and tuned gradient-boosting models using recall and precision-recall metrics.
6. **Remove identifier leakage risk.** The current deployment preserves the previous model's `ID` feature for reproducibility. Customer IDs should be removed before the next production model is trained, because an identifier should not determine churn risk.
7. **Monitor intervention outcomes.** Run A/B tests that compare retention actions against a control group. Track whether contact reduces actual churn, not only whether the model predicts it.
8. **Publish and govern the application.** Deploy the Streamlit app using the included guide, restrict access to authorised staff, and monitor prediction quality and fairness before using it in customer-facing decisions.
