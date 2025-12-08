
# 🏡 House Price Regression — Kaggle Competition 


## 🔑 Summary
This project is a Kaggle-style regression workflow to predict house sale prices. It follows a notebook-first approach (EDA → Feature Engineering → Modeling → Evaluation → Submission) and combines feature engineering, robust preprocessing, and stacked models. The final model is a **StackingRegressor** (stacked ensemble) with a cross-validated **RMSE = 19,292.2378** and a Kaggle public leaderboard score of **0.12730 (RMSLE)**.

---

## 📌 Objective
Predict the final sale price of houses using structured tabular features. The evaluation metric used is **Root Mean Squared Error (RMSE)** for local validation and **RMSLE** for Kaggle submissions (score reported above).

---

## 📊 Data & Where to find it
- **Train / Test source:** Kaggle dataset (House Prices - Advanced Regression Techniques)
- Typical files: `train.csv`, `test.csv`, `sample_submission.csv`
- Rows & features: dataset contains 37 numerical and 43 categorical features related to house structure, quality and sale details.

---

## 🧭 Project Structure (notebook-style)
```
project_root/
├── house_price_prediction.ipynb  
├── test.csv 
├── train.csv 
├── submission.csv
├── requirements.txt
└── README.md
```

---

## 🧩 Feature Engineering 
The following columns were dropped due to many missing values:

```python
['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2']
```

New features created (and applied to both train and test):

```python
# Age and remodel-derived features
HouseAge = YrSold - YearBuilt
HouseRemodelAge = YrSold - YearRemodAdd

# Areas and totals
TotalSF = 1stFlrSF + 2ndFlrSF + BsmtFinSF1 + BsmtFinSF2
TotalArea = GrLivArea + TotalBsmtSF

# Bathrooms aggregated
TotalBaths = BsmtFullBath + FullBath + 0.5 * (BsmtHalfBath + HalfBath)

# Porch / outdoor areas aggregated
TotalPorchSF = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF
```

After creating aggregates, original columns used to compute them were removed to avoid redundancy:

```python
# Dropped columns (examples)
['YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2',
 'GrLivArea', 'TotalBsmtSF', 'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath',
 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
```

These transformations stabilize variance, reduce dimensionality and encode aggregated signals (size, age, amenities).

---

## ⚙️ Preprocessing & Pipelines

The dataset was split into `X` and `y` and processed using a **ColumnTransformer** with two pipelines:

**Numeric pipeline**:
- `SimpleImputer(strategy='median')`
- `StandardScaler()`

**Categorical pipeline**:
- `SimpleImputer(strategy='most_frequent')`
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

Example snippet:

```python
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
```

**Target transformation**:
All models were trained using `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` to stabilize skewness in `SalePrice` and improve RMSE.

---

## 🧪 Models & Final Stack
Several regressors were trained and compared (examples):

- Linear models (Ridge, Lasso)  
- Tree-based (RandomForest, GradientBoosting)  
- Boosting libraries (XGBoost / LightGBM)  
- Support Vector Regressor / KNN (baseline)  

**Final model:** `StackingRegressor`   
**Local validation metric:** `RMSE = 19,292.2378` 
**Kaggle public leaderboard score:** `RMSLE = 0.12730` 

---


## 🔎 Model Performance & Metrics

Local cross-validated performance reported as RMSE:
- **RMSE (mean cv)**: **19,292.2378**

Kaggle submission:
- **RMSLE (public)**: **0.12730**

---

## 💡 Key Takeaways (EDA & Engineering insights)

- **OverallQual and GrLivArea/TotalArea** are among the strongest predictors of SalePrice.  
- **Feature engineering** (TotalSF, TotalArea, TotalBaths, TotalPorchSF) improved model stability and lowered RMSE versus single raw features.  
- **Target transformation (log1p)** was essential to reduce skewness and stabilize model errors.  
- **Missing-value-heavy columns** (e.g., PoolQC, Alley) were removed because their imputation would add noise.  
- **Outliers in living area** significantly affect simple linear models; ensemble methods handle them better.  
- **One-hot encoding** increased dimensionality but allowed tree/boosting models to capture categorical effects.  

---

## 🧾 Lessons Learned

- Aggregating related features into meaningful totals often yields stronger predictive signals than keeping raw components.  
- Using a target transformation (log) with `TransformedTargetRegressor` reliably improves RMSE for highly skewed targets.  
- Stacking diverse learners (trees + linear) reduces variance and produces a better generalization on leaderboard metrics.  
- Careful handling of missing values and categorical encoding is crucial for real-world tabular data.  
- Visual inspections (boxplots, scatterplots) are still invaluable — automated pipelines should complement, not replace, manual EDA.

---

## 📦 How to run (quick)

1. Clone the repo: `git clone <repo-url>`  
2. Create environment: `conda create -n house python=3.10 && conda activate house`  
3. Install dependencies: `pip install -r requirements.txt`  
4. Open notebook `house_price_prediction.ipynb` and run all cells.  

---

## 🔭 Next steps / Improvements

- Create a robust pipeline for feature selection and model versioning (MLflow)  
- Implement target encoding / frequency encoding for high-cardinality categoricals  
- Hyperparameter optimization with Optuna or Bayesian search  
- Build a small inference API for model serving (FastAPI + Docker)  

---

## 👨‍💻 Author

Mateus Vieira Vasconcelos  
Data Analysis & Machine Learning Enthusiast

---
