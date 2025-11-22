
# 🏠 House Prices Prediction — Regression Project

A complete end-to-end machine learning workflow applied to the **Kaggle House Prices** dataset.  
This project includes EDA, preprocessing, baseline models, stacking, and a final Kaggle-ready submission.

---

## 📊 Exploratory Data Analysis (EDA)

A comprehensive exploratory analysis was performed to understand the structure of the dataset and identify patterns relevant to predicting house prices. The following steps summarize the EDA insights without relying on images:

1. Distribution Analysis

Key numerical variables such as SalePrice, GrLivArea, TotalSF, and LotArea were inspected for distribution shape.

SalePrice showed right-skewness, which motivated a log transformation.

Several predictors also exhibited skewed distributions, influencing the choice of scaling and transformation strategies.

2. Outlier Detection

Scatter plots and numerical summaries revealed several extreme outliers, particularly in:

Above-ground living area (GrLivArea)

Basement area (TotalBsmtSF)

Lot size metrics

Outliers that clearly deviated from the natural cluster of values were removed to stabilize model variance.

3. Categorical Feature Insights

Categorical variables such as:

OverallQual

Neighborhood

KitchenQual

GarageCond

were analyzed using group statistics and value counts. Many of these showed clear relationships with SalePrice, confirming their importance.

4. Correlation Investigation

A correlation matrix was analyzed to understand linear relationships between numerical variables and the target.
Key findings:

OverallQual had one of the strongest positive correlations with SalePrice.

Total surface-related features (TotalSF, GrLivArea, TotalBath) also correlated strongly.

Some features such as GarageArea and GarageCars were redundant, leading to selective dropping.

5. Missing Value Assessment

A missing-value audit revealed:

Some features had minimal missingness and were suitable for imputation.

Others had extremely high missingness (e.g., PoolQC, MiscFeature), providing little predictive value and were removed.

6. Feature Behavior Observations

Certain ordinal variables required special handling due to non-linear patterns (e.g., condition and quality ratings).

Some categorical features showed strong price separation across categories, confirming their predictive relevance.

---

## 🧼 Data Cleaning & Preprocessing

- Removal of extreme outliers
- Cleaning of missing values
- Feature dropping based on correlation and redundancy
- Feature engineering (`TotalSF`, `TotalBath`, `HouseAge`, etc.)

Preprocessing is handled using **ColumnTransformer** with:
- StandardScaler for numerical features
- OneHotEncoder for categorical features

---

## 🤖 Models Implemented

- **Ridge Regression**
- **RandomForestRegressor**
- **XGBoostRegressor**
- **LightGBM**
- **CatBoost**

All models wrapped in:
```
TransformedTargetRegressor(func=log1p, inverse_func=expm1)
```

---

## 🏆 Stacked Ensemble

A stacking regressor combining:

- Ridge  
- RandomForest  
- XGBoost
- LightGBM
- CatBoost

with **Ridge** as the meta-learner.

---

## 🧪 Model Evaluation

Evaluation performed using **5-fold cross-validation** with **RMSE**.

---

## 📤 Final Submission

Final Kaggle-ready predictions saved as:

```
submission.csv
```

---

## 📁 Project Structure

```
📦 House-Price-Prediction/
├── house_price_prediction.ipynb
├── train.csv
├── test.csv
├── submission.csv
└── README.md
```

---

## 📚 What I Learned

- How to conduct structured EDA  
- Detect and handle outliers  
- Build preprocessing pipelines  
- Compare regression models  
- Use stacking to improve predictions  
- Build a complete ML pipeline from data to submission  

---

## 🚀 Future Improvements

- Hyperparameter tuning (Optuna)
- Advanced stacking/blending
- SHAP interpretability
- Feature engineering expansion

---

## ▶️ How to Run This Project

### **1. Install dependencies**
```bash
pip install -r requirements.txt
```

### **2. Open the notebook**
```bash
jupyter notebook house_price_prediction.ipynb
```

### **3. Run all cells**
Make sure `train.csv` and `test.csv` are in the same directory as the notebook.

### **4. Generate final predictions**
The notebook will automatically create:
```
submission.csv
```
You can upload this file to Kaggle for scoring.

---

## 📬 Contact
If you'd like to discuss the project, collaborate, or see more of my work:

**Author:** Mateus Vieira Vasconcelos  
**GitHub:** https://github.com/ludoteca12
