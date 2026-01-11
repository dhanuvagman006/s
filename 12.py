#12. Program to Implement multiple linear regression using iris
#dataset, visualize and analyze the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
#Loading the Iris Dataset
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
data
#Defining Features and Target
X = data.drop('target', axis=1) # all features
y = data['target'] # target variable
X
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize (scale) the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Create and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# Predict target values for the test set
y_pred = model.predict(X_test_scaled)
# Evaluate model performance
mse = mean_squared_error(y_test, y_pred) # Mean Squared Error
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Value:", r2)
# Visualization: Actual vs Predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Perfect line
plt.title("Multiple Linear Regression on Iris Dataset")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
# Analysis of the Results
print("\nAnalysis of Results")
print("---------------------")
print("Coefficients:", model.coef_) # Model weights
print("Intercept:", model.intercept_) # Bias value
print("Feature Importances: Not available for LinearRegression model")
print("Mean Squared Error:", mse)
print("R-squared Value:", r2)
print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)