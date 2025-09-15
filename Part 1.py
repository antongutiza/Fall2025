
# Data source: Ames Housing Dataset from Kaggle (https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


df = pd.read_csv("AmesHousing.csv")
df.columns = df.columns.str.replace(' ', '_')

# Features and target
# 'Gr_Liv_Area' is the square footage of the house.
# 'Neighborhood' is the location.
# 'SalePrice' is the target variable.
X = df[['Gr_Liv_Area', 'Neighborhood']]
y = df['SalePrice']

# Preprocessing: One-hot encode the categorical 'Neighborhood' column.
preprocessor = ColumnTransformer(
    transformers=[
        ('location', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Neighborhood'])
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and a linear regression model.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model.
model.fit(X_train, y_train)

# Make a prediction for a new house: 2000 sq ft in OldTown.
new_house = pd.DataFrame({'Gr_Liv_Area': [2000], 'Neighborhood': ['OldTown']})
predicted_price = model.predict(new_house)

print("\n--- Model Prediction ---")
print(f"Predicted price for a 2000 sq ft house in OldTown: ${predicted_price[0]:,.2f}")

# Display model coefficients to show the impact of each feature.
feature_names = (model.named_steps['preprocessor']
                 .named_transformers_['location']
                 .get_feature_names_out(['Neighborhood'])).tolist()
feature_names.append('Gr_Liv_Area')

coefficients = model.named_steps['regressor'].coef_

print("\n--- Model Coefficients ---")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:,.2f}")

```
