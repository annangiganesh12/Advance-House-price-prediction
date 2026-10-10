# Advanced House Price Prediction

A machine learning project that predicts house sale prices from 79 property features using the Ames Housing dataset (Kaggle "House Prices: Advanced Regression Techniques" format). The project covers the full workflow: exploratory data analysis, feature engineering, encoding, scaling, a tuned Lasso regression model, feature selection, validation, and generation of a prediction file for the test set.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Learnings](#key-learnings)
- [Scope for Improvement and Future Plan](#scope-for-improvement-and-future-plan)

---

## Overview

**Goal:** predict the final `SalePrice` of a house from attributes such as lot size, overall quality, living area, basement, garage, neighbourhood, and the year it was built or remodelled.

**Approach in one line:** clean and engineer the features, log-transform skewed values, encode categories, scale everything to the same range, and let **Lasso regression** (L1 regularization) pick the most useful features while predicting log-price.

---

## Results

Validation set performance (20% hold-out, predictions converted back to the original price scale):

| Metric | Score |
|--------|-------|
| **R² score** | **0.8571** |
| MAE (Mean Absolute Error) | 19,726.33 |
| RMSE (Root Mean Squared Error) | 33,110.24 |

The model explains about 86% of the variance in house prices, with an average error of roughly 19.7k per house.

> Metrics are computed with the best Lasso model found by grid search, before the feature-selection step. The final test predictions use the model refit on the features selected by `SelectFromModel`.

---

## Dataset

The data follows the Ames Housing format used in the Kaggle competition.

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 1,460 | Features plus the target `SalePrice` |
| `test.csv` | 1,459 | Features only, used for final predictions |
| `submission.csv` | 1,459 | Sample submission format (`Id`, `SalePrice`) |

- **Features:** 79 columns covering numeric measurements (areas, counts, years) and categorical descriptors (zoning, neighbourhood, quality ratings, and more)
- **Target:** `SalePrice`

---

## Pipeline

### 1. Exploratory data analysis
- Missing-value counts per feature, and how missingness relates to the average sale price
- Discrete features vs. average price, histograms of continuous features, and scatter plots against `SalePrice`
- Correlation of numeric features with the target
- Box plots for outlier inspection, and category breakdowns for categorical features

The plotting calls are commented out (`plt.show()`) so the script runs end to end without stopping. Uncomment them to view the charts.

### 2. Missing values
- **Categorical features:** missing values are replaced with a new `MISSING` label, since absence often carries meaning (for example, no basement or no garage).
- **Numeric features:** filled with the **median**, which is robust to outliers.

### 3. Temporal features
`YearBuilt`, `YearRemodAdd` and `GarageYrBlt` are converted into **ages** (`YrSold` minus the year), which is more informative than raw years.

### 4. Skewness correction
Numeric features with skewness above 0.5 are transformed with `log1p`. The target `SalePrice` is transformed too, so the model predicts log-price and predictions are converted back with `expm1`.

### 5. Rare categories
Categories that make up less than 1% of the training rows are grouped into a single `Rare_var` label to reduce noise and overfitting.

### 6. Encoding
- **Ordinal encoding** for ranked quality features (`ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`) using the scale Po=1, Fa=2, TA=3, Gd=4, Ex=5, and 0 for missing.
- **One-hot encoding** (with `drop_first=True`) for all remaining nominal features. Train and test are encoded together so both end up with identical columns.

### 7. Scaling and splitting
- 80/20 train/validation split with `random_state=42`
- `MinMaxScaler` fit on the training split and applied to validation and test

### 8. Modelling
- **Lasso regression** with `GridSearchCV` (5-fold cross-validation) over `alpha` in `[0.01, 0.001, 0.1, 1, 10, 100]`
- **`SelectFromModel`** keeps only the features Lasso assigns non-zero weight, and the model is refit on them
- Final predictions are converted back to dollars with `expm1` and written to `PREDICTED_SALEPRICES.csv`

---

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Statistics:** SciPy (skewness)
- **Machine learning:** scikit-learn (Lasso, GridSearchCV, SelectFromModel, MinMaxScaler, metrics)
- **Visualization:** matplotlib

---

## Project Structure

```
.
├── P1.py                      # full pipeline: EDA, preprocessing, training, evaluation, predictions
├── DATASETS/
│   └── archive/
│       ├── train.csv          # training data
│       ├── test.csv           # test data
│       └── submission.csv     # sample submission format
├── PREDICTED_SALEPRICES.csv   # final predicted sale prices for the test set
├── final_train.csv            # intermediate processed numeric training data
├── final_test.csv             # intermediate processed numeric test data
├── R-SQUARE SCORE.png         # screenshot of validation metrics
└── README.md
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/Advance-House-price-prediction.git
cd Advance-House-price-prediction
pip install pandas numpy scipy scikit-learn matplotlib
```

### 2. Update the data paths

`P1.py` currently reads the datasets from an absolute Windows path. Change the two `read_csv` lines near the top to relative paths:

```python
train_set = pd.read_csv('DATASETS/archive/train.csv')
test_set = pd.read_csv('DATASETS/archive/test.csv')
```

### 3. Run

```bash
python P1.py
```

The script prints dataset shapes, the validation metrics (MAE, RMSE, R²) and a preview of the predictions, and writes `PREDICTED_SALEPRICES.csv`.

---

## Key Learnings

- Log-transforming skewed features and the target noticeably helps linear models on house price data.
- Lasso's L1 penalty acts as built-in feature selection, which suits a dataset with many one-hot encoded columns.
- Treating "missing" as its own category preserves useful information (for example, houses without a basement or garage).
- Grouping rare categories keeps the feature space smaller and reduces overfitting.
- Encoding train and test together prevents column mismatches between the two sets.

---

## Scope for Improvement and Future Plan

### Modelling
- Compare with tree-based and boosting models (Random Forest, XGBoost, LightGBM, CatBoost) and combine them with stacking or blending
- Add Ridge and ElasticNet as baselines, and widen the hyperparameter search
- Report cross-validated scores instead of a single split, and analyse residuals to find where the model struggles

### Feature engineering and data quality
- Create combined features such as total square footage, total bathrooms and house age at sale
- Handle known outliers (for example very large living areas with low prices)
- Derive missing values group-wise (for example `LotFrontage` by neighbourhood) instead of one global median

### Code quality
- Refactor into an `sklearn` `Pipeline` with `ColumnTransformer` so preprocessing is fit only on training data and reused safely at prediction time
- Replace absolute paths with relative paths or a config file, and split the script into modules (preprocessing, training, evaluation)
- Save the trained model and scaler with `joblib`, and add unit tests

### Deployment and DevOps
- Serve the model through a REST API (FastAPI) or a Streamlit app where users enter house details and get a price estimate
- Containerize with Docker and add a CI workflow (GitHub Actions) for linting, tests and retraining checks
- Track experiments and model versions with MLflow, and monitor prediction drift once deployed

