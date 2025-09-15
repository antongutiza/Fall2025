# Data source: Telco Customer Churn dataset from Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

try:
    # Load the dataset from the CSV file.
    # Make sure the 'Telco-Customer-Churn.csv' file is in the same location as this script.
    df = pd.read_csv("Telco-Customer-Churn.csv")
except FileNotFoundError:
    print("Error: The 'Telco-Customer-Churn.csv' file was not found.")
    print("Please make sure you have downloaded the file and it is in the same directory as this script.")
    exit()

# Drop the CustomerID as it's not a useful feature for the model.
df.drop('customerID', axis=1, inplace=True)

# Replace ' ' with NaN and drop rows with missing values.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Define features (X) and target (y)
# The target is the 'Churn' column, and all other columns are features.
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical and numerical columns for preprocessing
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines for both categorical and numerical features
numerical_transformer = StandardScaler() # Scales numerical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Encodes categorical data

# Combine the transformers into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and a logistic regression model.
# Logistic Regression is a standard choice for binary classification tasks like churn prediction.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make a prediction for a new customer
# Example customer data: A female senior citizen with no partner, and a 1-month tenure.
new_customer_data = {
    'gender': ['Female'],
    'SeniorCitizen': [1],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [1],
    'PhoneService': ['No'],
    'MultipleLines': ['No phone service'],
    'InternetService': ['DSL'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['Yes'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [29.85],
    'TotalCharges': [29.85]
}
new_customer = pd.DataFrame(new_customer_data)
predicted_churn = model.predict(new_customer)
predicted_proba = model.predict_proba(new_customer)[0][1]

print("\n--- Model Prediction ---")
print(f"Predicted churn status for the new customer: {predicted_churn[0]}")
print(f"Predicted probability of churn: {predicted_proba:.2f}")

# Display model coefficients to show the importance of each feature.
# This part is a little more complex due to the pipeline, but it gives valuable insights.
feature_names = (model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(categorical_features)).tolist()
feature_names += numerical_features.tolist()

coefficients = model.named_steps['classifier'].coef_[0]

print("\n--- Top 10 Model Coefficients ---")
# Create a DataFrame to sort and display the top coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
top_coefficients = coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(10)

for index, row in top_coefficients.iterrows():
    print(f"{row['Feature']:<30}: {row['Coefficient']:,.2f}")
