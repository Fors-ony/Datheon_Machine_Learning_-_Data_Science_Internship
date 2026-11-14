# California Housing Price Prediction: OLS vs Ridge vs Lasso

## Objective
-Predict median house values in California districts and compare Ordinary Least
-Squares (OLS), Ridge (L2) and Lasso (L1) regression using RMSE, MAE and R².

## Dataset
California Housing dataset (20,640 districts, 10 columns). Target:`median_house_value`.

## Approach
1. Loaded the data and explored it (found 207 missing values in `total_bedrooms`).
2. Plotted a correlation heatmap and found strong multicollinearity among
   `total_rooms`, `total_bedrooms`, `population` and `households`.
3. Engineered features: `rooms_per_household`, `bedrooms_per_room`,
   `population_per_household` and an interaction feature
   `income_x_rooms_per_hh`.
4. One-hot encoded `ocean_proximity` (first category dropped).
5. Split 80/20, then imputed missing values (median) and standardized features,
   fitting both only on the training set to avoid data leakage.
6. Tuned Ridge and Lasso over 13 alpha values (0.01 to 10,000) using 5-fold
   GridSearchCV. Best alphas: Ridge 31.62, Lasso 100.

## Results (test set)
All three models performed almost the same on unseen data.
- OLS: RMSE of 68,917, MAE of 49,710 and R² of 0.638.
- Ridge: RMSE of 68,945, MAE of 49,670 and R² of 0.637.
- Lasso: RMSE of 68,934, MAE of 49,646 and R² of 0.637.
On average, each model's predictions were off by about $49,700 (MAE), and each explained roughly 64% of the variation in house prices (R²). OLS had the slightly lowest RMSE and the highest R², while Ridge and Lasso had the slightly lowest MAE. 
These gaps are tiny, so no model is clearly better than the others.

## Key findings
- The three models perform almost identically. Train R² (0.659) and test R²
  (0.637) are close, so the models are not overfitting and regularization has
  little to correct with 16 features and over 16,000 training rows.
- `median_income` is the strongest predictor (correlation 0.69).
- Lasso did not remove any feature at its best alpha.

## Limitations
- `median_house_value` is capped at 500,001, so the model underpredicts those
  districts and produces a visible stripe in the plots.
- A few predictions are negative, since linear models are not bounded.
- Residual spread grows with predicted price, so predictions are less reliable
  for expensive districts.

## How to run
Install: pandas, numpy, matplotlib, seaborn, scikit-learn.
Open `housing_regression.ipynb` and run all cells (`housing.csv` must be in the
same folder).
