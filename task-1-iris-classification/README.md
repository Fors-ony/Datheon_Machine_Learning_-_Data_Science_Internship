# Multi-Class Benchmark Classification Pipeline (Iris)

A classification pipeline on the Iris dataset that compares Logistic Regression and a Decision Tree.

## What it does
- Loads and inspects the data (150 rows, 3 balanced species, no missing values)
- Explores it with a Seaborn pairplot
- Splits the data 80/20 (stratified), then scales features with `StandardScaler` (fitted on the training set only, to avoid data leakage)
- Trains Logistic Regression and a Decision Tree
- Evaluates both with accuracy, precision, recall, F1-score, classification reports and confusion matrix heatmaps
- Plots decision boundaries using two features (petal length and width)

## Results (test set, 30 flowers)
Both models got 28 out of 30 test flowers right, so the accuracy was 93.3% for each. Precision, recall and F1-score were also 93.3% for both (averaged across the three species).

Setosa was always predicted correctly. The only two mistakes were one versicolor predicted as virginica and one virginica predicted as versicolor. These are the same two species that overlapped in the pairplot.

The Decision Tree scored 100% on the training data but 93.3% on the test data, so it overfits a little. The test set is only 30 flowers, so I can't say one model is better than the other. Next time I would try cross-validation to compare them more fairly.
(Precision, recall and F1 are macro-averaged.)

## Files
- `iris_classification_pipeline.ipynb` – the full notebook
- `iris.csv` – the dataset

## How to run
    pip install pandas numpy seaborn matplotlib scikit-learn
Then open the notebook and click Run All.
    pip install pandas numpy seaborn matplotlib scikit-learn
Then open the notebook and click Run All.
