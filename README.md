Fraud Detection Project
Task 1: Data Analysis & Preprocessing
📌 Project Overview

This project focuses on building a fraud detection system using machine learning.
Task 1 covers data cleaning, exploratory data analysis (EDA), feature engineering, geolocation integration, and data preprocessing to prepare the dataset for modeling.

We worked with two datasets:

Fraud transaction dataset (Fraud_Data.csv)
Credit card transaction dataset (creditcard.csv)
IP-to-country mapping dataset (IpAddress_to_Country.csv)
📁 Project Structure
fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│
├── README.md
⚙️ Task 1 Objectives
Clean raw datasets
Perform exploratory data analysis (EDA)
Handle missing values and duplicates
Perform geolocation enrichment using IP addresses
Create meaningful features for modeling
Encode categorical variables
Scale numerical features
Handle class imbalance using SMOTE
🧹 1. Data Cleaning
Actions performed:
Removed duplicate records
Checked missing values (none or minimal)
Converted date columns to datetime format
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])
📊 2. Exploratory Data Analysis (EDA)
Univariate Analysis:
Fraud distribution (class)
Age distribution
Purchase value distribution
sns.countplot(x='class', data=df)
sns.histplot(df['age'])
sns.histplot(df['purchase_value'])
Bivariate Analysis:
Fraud vs Age
Fraud vs Purchase Value
Browser vs Fraud
Source vs Fraud
sns.boxplot(x='class', y='age', data=df)
sns.boxplot(x='class', y='purchase_value', data=df)
Key Observations:
The dataset is highly imbalanced (fraud is rare)
Certain browsers and sources show higher fraud rates
Purchase behavior differs between fraud and non-fraud cases
🌍 3. Geolocation Integration
Objective:

Map IP addresses to countries using range-based lookup.

Steps:
Converted IP to integer
Merged datasets using merge_asof
Filtered valid IP ranges
df['ip_address'] = df['ip_address'].astype(int)
merged_df = pd.merge_asof(
    df.sort_values('ip_address'),
    ip_df.sort_values('lower_bound_ip_address'),
    left_on='ip_address',
    right_on='lower_bound_ip_address',
    direction='backward'
)
Result:
Each transaction is linked to a country
Fraud analysis performed by country
🛠️ 4. Feature Engineering
Created Features:
⏱ Time Features
df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
df['hour_of_day'] = df['purchase_time'].dt.hour
df['day_of_week'] = df['purchase_time'].dt.dayofweek
🔁 Frequency Features
df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
df['device_frequency'] = df.groupby('device_id')['device_id'].transform('count')
🔄 5. Data Transformation
One-Hot Encoding:
df = pd.get_dummies(df, columns=['source','browser','sex','country'], drop_first=True)
Feature Scaling:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['age','purchase_value','time_since_signup','transaction_count']] = scaler.fit_transform(...)
⚖️ 6. Handling Class Imbalance
Method Used: SMOTE

Applied only on training data.

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
Reason:

SMOTE generates synthetic fraud samples instead of removing data, helping the model learn better fraud patterns.

📈 Final Outcome

After Task 1 completion:

✔ Clean datasets
✔ Feature-rich dataset
✔ Encoded categorical variables
✔ Scaled numerical features
✔ Balanced training data (SMOTE applied)
✔ Country-level fraud insights

🚀 Next Step

➡ Proceed to Task 2: Modeling

Logistic Regression
Random Forest
XGBoost
Model evaluation (Accuracy, Precision, Recall, ROC-AUC)

# Task 2: Model Building and Training

## Objective

The objective of Task 2 was to build, train, and evaluate machine learning models for fraud detection on two highly imbalanced datasets:

1. E-commerce Fraud Detection Dataset (`Fraud_Data.csv`)
2. Credit Card Fraud Detection Dataset (`creditcard.csv`)

The task focused on developing baseline and ensemble classification models, handling class imbalance, and comparing model performance using appropriate evaluation metrics.

---

## Data Preparation

### E-commerce Fraud Dataset

* Loaded the processed dataset generated in Task 1.
* Separated features and target variable (`class`).
* Removed non-numeric and non-predictive columns.
* Applied stratified train-test split to preserve class distribution.
* Applied SMOTE on the training set only to address class imbalance.

### Credit Card Fraud Dataset

* Loaded the credit card transaction dataset.
* Separated features and target variable (`Class`).
* Applied stratified train-test split.
* Applied SMOTE on the training set only.

---

## Baseline Model: Logistic Regression

Logistic Regression was selected as the baseline model due to its simplicity and interpretability.

### Training

* Trained on the SMOTE-resampled training data.
* Used `max_iter=1000` to improve convergence.

### Evaluation Metrics

* F1 Score
* AUC-PR (Average Precision Score)
* Confusion Matrix
* Classification Report

---

## Ensemble Model: Random Forest

Random Forest was selected as the ensemble model because of its strong performance on classification tasks and its ability to capture non-linear relationships.

### Hyperparameters

* `n_estimators=100`
* `max_depth=10`
* `random_state=42`

### Evaluation Metrics

* F1 Score
* AUC-PR
* Confusion Matrix
* Classification Report

---

## Cross Validation

Stratified K-Fold Cross Validation was performed using:

* Number of folds: 5
* Shuffle: True
* Random State: 42

The mean and standard deviation of F1 scores were reported to estimate model stability and generalization performance.

---

## Model Comparison

The models were compared using:

* F1 Score
* AUC-PR

These metrics were chosen because fraud detection datasets are highly imbalanced and overall accuracy can be misleading.

Comparison tables were generated for both datasets to identify the best-performing model.

---

## Model Selection

### E-commerce Fraud Dataset

Random Forest achieved better performance than Logistic Regression based on F1 Score and AUC-PR and was selected as the final model.

### Credit Card Fraud Dataset

Random Forest outperformed Logistic Regression and was selected as the final model.

---

## Saved Models

The following trained models were saved:

* `models/random_forest_fraud.pkl`
* `models/random_forest_creditcard.pkl`

These models will be used in Task 3 for explainability and interpretation using SHAP.

---

## Key Takeaways

* SMOTE successfully addressed severe class imbalance.
* Random Forest consistently outperformed Logistic Regression.
* F1 Score and AUC-PR provided more reliable evaluation than accuracy.
* The selected models demonstrated strong capability for fraud detection across both datasets.



Model Building and Training
Objective

This notebook develops and evaluates fraud detection models for both the e-commerce transaction dataset and the credit card transaction dataset. Due to the highly imbalanced nature of fraud data, evaluation focuses on F1-Score and AUC-PR rather than accuracy.

E-commerce Fraud Modeling
Data Preparation
Feature and Target Separation

The target variable is class, where:

1 = Fraudulent transaction
0 = Legitimate transaction

Features were separated from the target before model training.

Train-Test Split

A stratified train-test split was used to preserve the original fraud distribution in both training and testing datasets.

Handling Class Imbalance

The fraud dataset is highly imbalanced. SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training set to generate synthetic fraud examples and improve model learning while avoiding data leakage.

Logistic Regression (Baseline Model)

Logistic Regression was selected as the baseline model because it is simple, interpretable, and commonly used for binary classification problems.

The model was evaluated using:

F1-Score
AUC-PR
Confusion Matrix
Classification Report
Random Forest Model

Random Forest was selected as the ensemble model because it can capture complex relationships and interactions among features.

Hyperparameters used:

n_estimators = 100
max_depth = 10

The model was evaluated using the same metrics as Logistic Regression for fair comparison.

Cross Validation

Stratified 5-Fold Cross Validation was performed to obtain a more reliable estimate of model performance.

Reported metrics:

Mean F1-Score
Standard Deviation of F1-Score
Model Comparison

The performance of Logistic Regression and Random Forest was compared using F1-Score and AUC-PR.

The comparison table below summarizes the results.

Model Selection

Random Forest achieved higher F1-Score and AUC-PR than Logistic Regression.

Therefore, Random Forest was selected as the final model for e-commerce fraud detection.

Credit Card Fraud Modeling
Data Preparation

The target variable is Class, where:

1 = Fraudulent transaction
0 = Legitimate transaction

A stratified train-test split was applied to preserve class proportions.

SMOTE was used on the training set to address class imbalance.

Logistic Regression (Baseline Model)

Logistic Regression served as the baseline model for credit card fraud detection.

Performance was evaluated using:

F1-Score
AUC-PR
Confusion Matrix
Classification Report
Random Forest Model

Random Forest was trained as the ensemble model using:

n_estimators = 100
max_depth = 10

The model was evaluated using the same metrics as the baseline model.

Cross Validation

Stratified 5-Fold Cross Validation was performed to assess model stability and generalization performance.

Model Comparison

Both models were compared using F1-Score and AUC-PR.

The comparison table below summarizes the results.

Model Selection

Random Forest outperformed Logistic Regression in detecting fraudulent transactions.

Therefore, Random Forest was selected as the final model for credit card fraud detection.

Final Conclusion

Both datasets were treated as separate fraud detection problems because they contain different feature structures.

Key findings:

SMOTE successfully balanced the training datasets.
Logistic Regression provided an interpretable baseline.
Random Forest consistently achieved better fraud detection performance.
Cross-validation confirmed the robustness of the selected model.
Random Forest was chosen as the final model for both datasets and saved for explainability analysis in Task 3.

## Task 3: Model Explainability with SHAP 
🧠 Overview

This project focuses on model interpretability using SHAP (SHapley Additive exPlanations) for two fraud detection systems:

E-commerce Fraud Detection Model
Credit Card Fraud Detection Model

The objective is to explain model predictions, understand key drivers of fraud, and translate insights into actionable business recommendations.

🎯 Objectives
Extract built-in feature importance from ensemble models
Apply SHAP for global and local interpretability
Analyze:
True Positives
False Positives
False Negatives
Compare SHAP vs traditional feature importance
Identify top drivers of fraud
Provide business recommendations based on explainability
📦 Models Used
1. E-commerce Fraud Model
Algorithm: Random Forest Classifier
Target: Fraud vs Non-fraud transactions
Key engineered features include:
device_frequency
time_since_signup
traffic source features
2. Credit Card Fraud Model
Algorithm: Random Forest Classifier
Dataset: PCA-transformed credit card transactions
Features: V1–V28 + transaction amount
📊 1. Feature Importance (Baseline)
E-commerce Model

Top features from Random Forest:

device_frequency
time_since_signup
source_Direct
source_SEO
browser type features
Credit Card Model

Top features:

V14
V10
V4
V12
V17

📌 These were extracted using:

model.feature_importances_
📈 2. SHAP Global Explainability

SHAP summary plots were generated for both models to show overall feature impact.

Key Insights:
E-commerce Model
device_frequency and time_since_signup are dominant fraud indicators
Behavioral features are stronger than demographic features
Credit Card Model
V14, V10, V4, V12, V17 are the most influential variables
Model relies heavily on PCA-derived transaction patterns
🔍 3. SHAP Local Explainability

SHAP waterfall plots were used to explain individual predictions:

Cases analyzed:
✅ True Positive (correct fraud detection)
❌ False Positive (legitimate flagged as fraud)
❌ False Negative (missed fraud)
📌 Example Insights
True Positive
Strong fraud drivers:
device_frequency
time_since_signup
Model correctly identifies suspicious behavioral patterns
False Positive
Legitimate transaction flagged due to:
high device usage pattern
suspicious source behavior signals
False Negative
Fraud missed because:
legitimate-looking account age
normal behavioral signals outweighed fraud indicators
⚖️ 4. SHAP vs Feature Importance Comparison
Key Findings
Strong agreement between SHAP and Random Forest importance
Both models consistently identify:
device_frequency / V14
time_since_signup / V10
Minor ranking differences due to:
SHAP = local + global explanations
RF importance = global only
🧠 5. Top 5 Drivers of Fraud
E-commerce Fraud Model
device_frequency
time_since_signup
sex_M
source_Direct
source_SEO
Some features such as gender and traffic source appear influential in fraud predictions.
However, these variables do not directly cause fraud.

Their importance reflects patterns in historical data and may introduce bias if used for automated decision-making.

Therefore:
- These features should NOT be used as sole decision rules
- Human oversight is required for fairness validation
Credit Card Fraud Model
V14
V10
V4
V12
V17
⚠️ 6. Surprising Findings
Demographic features (e.g., sex) influence predictions
Traffic source features impact fraud detection
These may reflect dataset bias rather than true causality

📌 Important: Correlation ≠ causation

💡 7. Business Recommendations
1. Monitor High-Frequency Devices

Devices with unusually high transaction counts should be flagged.

➡ Based on: device_frequency

2. Strengthen New Account Verification

Transactions shortly after signup require additional checks.

➡ Based on: time_since_signup

3. Risk-Based Traffic Source Scoring

Adjust fraud risk depending on user acquisition channel.

➡ Based on: source_Direct, source_SEO

4. Real-Time Fraud Scoring System

Deploy model into production for live monitoring.

➡ Based on SHAP behavioral pattern detection

5. Analyst Support with SHAP

Use SHAP explanations to support fraud investigation teams.

📌 8. Conclusion

The Random Forest models demonstrate strong fraud detection performance across both datasets.

SHAP analysis provided:

Transparent model interpretability
Identification of key fraud drivers
Insight into false positive and false negative behavior
Strong alignment with business decision-making needs

Overall, SHAP significantly improves trust, transparency, and usability of machine learning models in fraud detection systems.

🛠️ Tools & Libraries
Python
Pandas
Scikit-learn
Matplotlib
SHAP
Joblib
