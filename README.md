# Telecom Customer Churn Prediction

Businesses lose significant revenue every time a customer quietly cancels and walks away. The challenge is that by the time it's obvious, it's usually too late. This project explores whether we can predict who's likely to leave before they do, using machine learning on real telecom customer data.

---

## Why This Project?

Customer churn is one of the most costly problems in businesses, and it is a challenge I encountered firsthand during my time at United Capital, where I worked on identifying, reactivating dormant and at-risk accounts to support retention. This project is my way of formalising that experience with a rigorous, end-to-end machine learning pipeline, building and comparing classification models to detect at-risk customers before they leave.

---

## Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,032 customers × 20 features (after cleaning)
- **Target variable:** Churn (1 = churned, 0 = stayed)
- **Challenge:** Only 27% of customers churned therefore the data is imbalanced, and a naive model could look accurate while completely missing the point


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

## What I Did (Methodology)

### 1. Data Cleaning
The dataset had a subtle issue, TotalCharges was stored as text instead of numbers, hiding 11 rows with blank values. I caught this during initial exploration, fixed the data type, and removed the affected rows before any analysis began. I also dropped customerID since it's just an identifier with no predictive value, and encoded the Churn column as binary (Yes=1, No=0) to prepare it for modelling.

### 2. Exploratory Data Analysis (EDA)
Before building anything, I wanted to understand what the data was actually saying. Three patterns stood out immediately:
- **New customers churn far more than loyal ones** — most churners left within their first 10 months
**Higher monthly charges correlate with churn** — expensive customers are more likely to leave
- **Contract type matters a lot** — month-to-month customers are at significantly higher risk than those on annual contracts
These insights shaped the entire modelling approach.

### 3. Feature Engineering
Many of the dataset's columns contained text categories like "Yes/No", "Month-to-month", or "Fiber optic". Machine learning models can only work with numbers. I applied one-hot encoding to convert every categorical column into numeric format, expanding the feature set from 20 to 31 columns. This step ensured the models could learn from every variable in the dataset without losing any information.

### 4. Handling Class Imbalance and Modelling
With only 27% of customers churning, a model could score 73% accuracy by just predicting "No Churn" every time, which is useless for a business trying to intervene. To fix this, both models were trained with class_weight='balanced', a cost-sensitive learning approach that forces the model to take the minority churn class seriously rather than optimising for the easy majority.

Models:
I built two models and evaluated them not just on accuracy but on recall: how many actual churners did we catch? The first was a Logistic Regression model built inside a StandardScaler pipeline to ensure the data was properly scaled before fitting. The second was a Random Forest Classifier, an ensemble model that builds multiple decision trees and combines their outputs for a more robust prediction

### 5. Hyperparameter Tuning
Both models were tuned using GridSearchCV with 5-fold cross-validation, optimising for recall and prioritising the detection of actual churners over overall accuracy. The baseline Random Forest was the weaker starting point, missing nearly half of all churners, so the improvement after tuning was the most significant.

---

## Results

| Model | AUC Score | Churn Recall |
|---|---|---|
| Logistic Regression (baseline) | 0.836 | 80% |
| Random Forest (baseline) | 0.818 | 49% |
| Logistic Regression (tuned) | 0.833 | 79% |
| **Random Forest (tuned)** | **0.836** | **80%** |

### Key Insight
The baseline Random Forest was catching less than half of churners, a serious problem if this were deployed in a real business. After tuning, it matched Logistic Regression at 80% recall and an AUC of 0.836. The tuned Random Forest was selected as the final model based on its combination of highest AUC and strongest churn recall.

---

## What's Actually Driving Churn?

Feature importance from the Random Forest revealed the top churn signals:

1. **TotalCharges** — customers who have spent more overall are more likely to leave
2. **Tenure** — the shorter someone has been a customer, the higher their risk
3. **MonthlyCharges** — higher monthly bills increase churn likelihood
4. **Contract type** — month-to-month contracts are highest risk
5. **Payment method** — electronic check users show elevated churn rates

---

## Business Recommendation

The highest-risk customers share a clear profile: short tenure, high monthly charges, month-to-month contract, paying by electronic check. These customers should be the primary target for proactive retention, whether that is a loyalty discount, a contract upgrade incentive, or a personalised outreach before the decision to leave is made.

---

## Technical Skills Demonstrated

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Exploratory Data Analysis (EDA)
- Feature engineering and one-hot encoding
- Classification modelling (Logistic Regression, Random Forest)
- Cost-sensitive learning (`class_weight='balanced'`)
- Hyperparameter tuning (GridSearchCV)
- Model evaluation (AUC, Recall, Confusion Matrix, Classification Report)
- Business-driven model selection and interpretation

---

## Author

**Ifeoma Isiuwe**  
[LinkedIn](https://www.linkedin.com/in/ifeoma-isiuwe) | ifeomaisiuwe@gmail.com
