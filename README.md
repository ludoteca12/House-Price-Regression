# House Prices — Advanced Regression Techniques

## 📌 Overview
This project predicts housing prices using the Ames Housing Dataset from the Kaggle competition **House Prices: Advanced Regression Techniques**.  
It showcases a complete end-to-end machine learning workflow, including:
- Exploratory Data Analysis (EDA)
- Feature preprocessing using Scikit-Learn Pipelines
- Model training with log-transformed targets
- Cross‑validation for evaluation
- Kaggle‑ready submission generation

This project is part of my data science portfolio and demonstrates my skills in data cleaning, visualization, modeling, and best practices for reproducible ML pipelines.

---

## 📂 Repository Structure

```
House-Prices--regression-
│
├── House_prices_Regression.ipynb     # Original notebook 
│
├── heatMap/                          # Saved heatmap images
├── histPlot/                         # Saved histogram images
├── scatterPlots/                     # Saved scatterplot images
├── boxPlots/                         # Saved boxplot images
│
├── train.csv                         # Training data
├── test.csv                          # Test data
│
├── requirements.txt                  # Dependencies for full reproducibility
└── README.md                         # This file
```

---

## 📊 Exploratory Data Analysis (EDA)

Since this project used manually exported images, all visualizations are stored in folders:

### 🔥 Heatmaps  
Correlation heatmaps highlight the strongest predictors of **SalePrice**, including:
- OverallQual  
- GrLivArea  
- GarageCars  
- TotalBsmtSF  

### 📈 Histogram  
Key distribution showcase skewed variable such as:
- SalePrice  

### 📉 Scatterplots  
Showing relationships such as:
- SalePrice vs. GrLivArea  
- SalePrice vs. TotalBsmtSF  

### 📦 Boxplots  
Used to inspect outliers across:
- Neighborhood  
- HouseStyle  
- OverallQual  

---

## 🧹 Data Preprocessing

All preprocessing is performed using **Scikit‑Learn Pipelines** with:
- **Median** imputation for numerical columns  
- **Most frequent** imputation for categorical columns  
- **StandardScaler** for numerical features  
- **OneHotEncoder** for categorical features 
- **ColumnTransformer** to combine both pipelines  
- **TransformedTargetRegressor** with `log1p` to reduce SalePrice skew  

This ensures:
- Full reproducibility  
- No data leakage  
- Clean CV performance estimates  

---

## 🤖 Modeling

The primary model used is **XGBoost**, chosen for:
- Strong performance on tabular data  
- Robustness with mixed-type features  
- Good handling of non-linearities  

### ✔️ Baseline Model
A simple but powerful baseline using:
```text
XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
```

### ✔️ Target Transformation
`SalePrice` is log-transformed via:
```
func = np.log1p
inverse_func = np.expm1
```

### ✔️ Evaluation
Cross-validation uses RMSE:
```
cv = 5 folds
scoring = neg_root_mean_squared_error
```

---

## 📈 Results

### Cross-validated RMSE (baseline)
```
~ 0.11 (log-space RMSE)
```

This score is competitive for a baseline model and can be improved with:
- Feature engineering  
- Hyperparameter tuning (Optuna or RandomizedSearchCV)  
- Model stacking or ensembling  

---

## 📤 Kaggle Submission

The Kaggle-ready notebook automatically produces:
```
HousePriceSubmission.csv
```

Format:
```
Id, SalePrice
```

---

## 🧠 Explainability (SHAP)
The portfolio notebook includes optional SHAP code for:
- Summary plot  
- Feature importance  
- Individual prediction explanations  

To enable SHAP visualization:
```
pip install shap
```

---

## ▶️ How to Run This Project

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Place the data**
Ensure `train.csv` and `test.csv` are in the project root or in a `data/` directory depending on the notebook you run.

### **3. Open the notebook**
Open the main notebook:
- `House_prices_Regression.ipynb`

### **4. Run all cells**
The notebook will:
- Load and preprocess data  
- Train models  
- Evaluate results  
- Generate predictions  
- Export Kaggle submission  

---

## 🚀 Future Improvements

To enhance performance and robustness:
- Hyperparameter tuning with Optuna  
- Feature engineering  
- Stacking models (CatBoost + XGBoost + LightGBM)  
- Outlier detection  
- Automated figure saving and logging  
- GitHub Actions for notebook execution & HTML export  

---

## 📬 Contact
If you'd like to discuss the project, collaborate, or see more of my work:

**Author:** Mateus Vieira Vasconcelos  
**GitHub:** https://github.com/ludoteca12

