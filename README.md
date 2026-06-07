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
