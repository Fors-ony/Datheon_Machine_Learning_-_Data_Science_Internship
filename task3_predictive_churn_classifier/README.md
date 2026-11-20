# Task 3: Customer Churn Classifier (Random Forest & XGBoost)

Predicts telecom customer churn using the Telco Customer Churn dataset (7,043 customers, 21 columns).

## Approach
1. Cleaned data: converted `TotalCharges` to numeric (11 blank values from new customers with tenure 0 filled with 0), dropped `customerID`.
2. Stratified 80/20 train/test split.
3. OneHotEncoder for categorical features (ColumnTransformer).
4. SMOTE to balance classes (26.5% churn), applied inside an imblearn pipeline so it only touches training folds.
5. Trained Random Forest and XGBoost, evaluated with 5-fold stratified cross-validation.

## Results

Both models beat the 0.80 ROC-AUC target. Random Forest reached a cross-validated ROC-AUC of 0.818 and a test-set ROC-AUC of 0.825. XGBoost did better, reaching 0.844 in cross-validation and 0.844 on the test set.
The cross-validation and test scores for XGBoost are identical, which suggests it generalises well and is not overfitting.

XGBoost was chosen as the final model. At the default 0.5 threshold, it correctly identified 214 of the 374 customers who churned (recall 0.57), 
and 63% of the customers it flagged as churners really did churn (precision 0.63).
XGBoost on the test set: recall 0.57 and precision 0.63 for churners (default 0.5 threshold).

## Top drivers (XGBoost feature importance)
Month-to-month contract, no tech support, no online security, electronic check payment, no online backup.

## Files
- `churn_classifier.ipynb`: full notebook
- `roc_curve.png`, `confusion_matrix.png`, `feature_importance.png`: plots
- `Telco-Customer-Churn.csv`: dataset

## Tools
Python, pandas, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn
