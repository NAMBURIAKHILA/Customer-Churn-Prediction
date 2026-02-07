Project Overview

Customer churn is a major challenge in the telecommunications industry, where retaining existing customers is far more cost-effective than acquiring new ones.
This project focuses on building a machine learning pipeline to predict whether a customer is likely to churn based on their demographic details, services subscribed, and billing information.

The solution includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment-ready pipeline saving, along with a simple frontend for interaction.

📂 Dataset Information

Source: Telecom Customer Churn Dataset

Rows: 7,043

Columns: 21

🔹 Features Description
Column Name	Description
CustomerID	Unique customer identifier
Gender	Male / Female
SeniorCitizen	0 = No, 1 = Yes
Partner	Yes / No
Dependents	Yes / No
Tenure	Number of months with the company
PhoneService	Yes / No
MultipleLines	Yes / No / No phone service
InternetService	DSL / Fiber optic / No
OnlineSecurity	Yes / No / No internet service
OnlineBackup	Yes / No / No internet service
DeviceProtection	Yes / No / No internet service
TechSupport	Yes / No / No internet service
StreamingTV	Yes / No / No internet service
StreamingMovies	Yes / No / No internet service
Contract	Month-to-month / One year / Two year
PaperlessBilling	Yes / No
PaymentMethod	Electronic check / Mailed check / Bank transfer / Credit card
MonthlyCharges	Monthly billing amount
TotalCharges	Total lifetime charges
Churn	Target variable (Yes / No)
🧪 Exploratory Data Analysis (EDA)

Checked missing values and data types

Analyzed churn distribution

Visualized correlations between features and churn

Identified key factors influencing customer churn

⚙️ Data Preprocessing

Converted TotalCharges to numeric

Handled missing values using median imputation

Label encoded categorical features

Standardized numerical features

Applied SMOTE to handle class imbalance

⚙️ Models Used

The following machine learning models were trained and evaluated across multiple train-test splits:

Random Forest

Gradient Boosting

AdaBoost

Support Vector Machine (SVM)

XGBoost

The best-performing model was selected based on accuracy.

📈 Model Evaluation

Accuracy comparison across different train-test splits

Confusion matrix visualization



💾 Model Deployment

The final trained model, scaler, encoders, and feature names were saved as a pipeline

This allows direct reuse of the model for predictions without retraining

🌐 Frontend Integration

A simple frontend interface was developed to interact with the churn prediction model

Users can input customer details and receive churn prediction results

🛠️ Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

HTML

Google Colab

Git & GitHub

🚀 How to Run the Project
pip install -r requirements.txt


Run the notebook:

jupyter notebook notebooks/churn_prediction.ipynb

📌 Future Improvements

Add ROC–AUC and Precision–Recall analysis

Deploy model using Flask or FastAPI

Improve frontend UI and validation

Add real-time prediction API
