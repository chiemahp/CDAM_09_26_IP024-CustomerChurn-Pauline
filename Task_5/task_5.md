Summary of Explainable AI Analysis
What was accomplished:

1. Global Feature Importance (SHAP)
Top 5 Most Important Features:
Account_Tenure_Months (0.1456)
Customer_Value_Score (0.1221)
Email_Subscription (0.0705)
Support_Calls (0.0464)
Number_of_Complaints (0.0411)
2. Key Insights from XAI:
Account Tenure: Longer tenure paradoxically increases churn (service switching behavior)
Customer Value Score: Higher-value customers sometimes churn (may seek better offers)
Email Subscription: Subscribed customers are more engaged and less likely to churn
Support Calls: More support calls indicate problems and higher churn risk
Complaints: Direct indicator of dissatisfaction and churn risk
3. Partial Dependence Plots (PDP)
Shows how each top feature influences churn probability across its value range
Reveals non-linear relationships between features and predictions
4. Individual Prediction Explanation
Identified a churner (Test index 1090) with 50.55% churn probability
Explained which factors contributed most to this prediction
Top factors: Account Tenure, Risk Profile, Customer Value Score, Complaints, Support Calls
5. Visualizations Generated:
✅ shap_feature_importance.png - Global feature importance
✅ shap_summary_bar.png - SHAP bar plot
✅ shap_summary_beeswarm.png - SHAP beeswarm plot
✅ partial_dependence_plots.png - PDP for top 4 features
✅ breakown_plot_churner.png - DALEX Break Down plot
6. Business Recommendations:
Improve satisfaction and reduce complaints
Encourage email subscriptions for engagement
Promote multi-product adoption
Monitor high-risk predicted churners
Enhance support quality
Use insights for targeted retention campaigns
Deliverable: xai_analysis_summary.txt - Complete plain-English interpretation of model decisions and business applications.