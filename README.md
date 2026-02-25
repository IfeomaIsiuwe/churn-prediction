# Telecom Customer Churn Prediction

A machine learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The project covers the full data science pipeline — from exploratory analysis through to model tuning and business interpretation.

---

## Business Context

Customer churn is one of the most costly problems in subscription-based businesses. Identifying at-risk customers before they leave allows businesses to intervene with targeted retention strategies. This project mirrors real-world churn analysis work, building and comparing classification models to detect churners as accurately as possible.

---

## Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,032 customers × 20 features (after cleaning)
- **Target variable:** Churn (1 = churned, 0 = stayed)
- **Class distribution:** 73% No Churn / 27% Churn — imbalanced dataset

---

## Project Structure

```
churn-prediction/
│
├── churn_analysis.ipynb        # Full analysis notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── README.md
```

---

## Methodology

### 1. Data Cleaning
- Identified and fixed `TotalCharges` column stored as text instead of numeric
- Removed 11 rows with missing values introduced during conversion
- Dropped `customerID` (non-predictive identifier)
- Encoded `Churn` target as binary (Yes=1, No=0)

### 2. Exploratory Data Analysis (EDA)
Key findings from visual analysis:
- **Tenure:** Churned customers had significantly shorter tenure — most left within the first 10 months
- **Monthly Charges:** Churned customers paid higher monthly charges on average
- **Contract Type:** Month-to-month contract customers showed the highest churn risk

### 3. Feature Engineering
- Applied one-hot encoding to all categorical variables
- Expanded feature set from 20 to 31 columns

### 4. Modelling — Class Imbalance Handling
Given the 73/27 class split, both models were trained with `class_weight='balanced'` (cost-sensitive learning) to prevent the model from ignoring the minority churn class.

Two models were built and compared:
- **Logistic Regression** (with StandardScaler pipeline)
- **Random Forest Classifier**

### 5. Hyperparameter Tuning
Both models were tuned using `GridSearchCV` with 5-fold cross-validation, optimising for **recall** — prioritising the detection of actual churners over overall accuracy.

---

## Results

| Model | AUC Score | Churn Recall |
|---|---|---|
| Logistic Regression (baseline) | 0.836 | 80% |
| Random Forest (baseline) | 0.818 | 49% |
| Logistic Regression (tuned) | 0.833 | 79% |
| **Random Forest (tuned)** | **0.836** | **80%** |

### Key Insight
The baseline Random Forest missed 49% of churners — a significant business risk. After hyperparameter tuning with GridSearchCV optimising for recall, churn detection improved to 80%, matching the best AUC score of 0.836.

**The tuned Random Forest was selected as the final model** based on its combination of highest AUC and strongest churn recall.

---

## Feature Importance

The top drivers of churn identified by the Random Forest model:

1. **TotalCharges** — highest financial exposure correlates with churn
2. **Tenure** — newer customers churn at significantly higher rates
3. **MonthlyCharges** — higher monthly bills increase churn risk
4. **Contract type** — month-to-month contracts are highest risk
5. **Payment method** — electronic check users show elevated churn

---

## Business Recommendation

> Customers on month-to-month contracts, paying via electronic check, with high monthly charges and short tenure represent the highest churn risk segment. Targeted retention interventions — such as loyalty discounts or contract upgrade incentives — should prioritise this group.

---

## Technical Skills Demonstrated

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Exploratory Data Analysis (EDA)
- Feature engineering and one-hot encoding
- Classification modelling (Logistic Regression, Random Forest)
- Cost-sensitive learning (`class_weight='balanced'`)
- Hyperparameter tuning (GridSearchCV)
- Model evaluation (AUC, Recall, Confusion Matrix, Classification Report)
- Business-driven model selection

---

## Author

**Ifeoma Isiuwe**  
MSc Data Science (Distinction) — Northumbria University  
[LinkedIn](https://www.linkedin.com/in/ifeoma-isiuwe) | ifeomaisiuwe@gmail.com
