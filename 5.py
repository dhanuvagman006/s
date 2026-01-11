#5. Program to perform cross validation for a given dataset to
#measure Root Mean Squared Error (RMSE), Mean Absolute Error
#(MAE) and R2 Error using Validation Set, Leave One Out
#Cross-Validation(LOOCV) and K-fold Cross-Validation approaches


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Create sample dataset directly in the code
data = pd.DataFrame({
 'feature1': [1,2,3,4,5,6,7,8,9,10],
 'feature2': [2,1,3,5,7,4,6,8,9,11],
 'feature3': [5,4,6,7,9,10,12,14,13,15],
 'target': [3,4,6,7,9,11,13,15,16,18]
})
# -----------------------------------------------------------
# Split data into features (X) and target (y)
# -----------------------------------------------------------
X = data.drop('target', axis=1)
y = data['target']
# -----------------------------------------------------------
# Validation Set Approach
# -----------------------------------------------------------
print("Validation Set Approach")
print("----------------------------")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Error:", r2)
# -----------------------------------------------------------
# Leave-One-Out Cross-Validation (LOOCV)
# -----------------------------------------------------------
print("\nLeave One Out Cross-Validation (LOOCV) Approach")
print("------------------------------------------------")
loo = LeaveOneOut()
rmse_loo, mae_loo, r2_loo = [], [], []
for train_index, val_index in loo.split(X):
 X_train_loo, X_val_loo = X.iloc[train_index], X.iloc[val_index]
 y_train_loo, y_val_loo = y.iloc[train_index], y.iloc[val_index]
 model.fit(X_train_loo, y_train_loo)
 y_pred_loo = model.predict(X_val_loo)
 rmse_loo.append(np.sqrt(mean_squared_error(y_val_loo, y_pred_loo)))
 mae_loo.append(mean_absolute_error(y_val_loo, y_pred_loo))
 r2_loo.append(r2_score(y_val_loo, y_pred_loo))
print("RMSE (LOOCV):", np.mean(rmse_loo))
print("MAE (LOOCV):", np.mean(mae_loo))
print("R2 Error (LOOCV):", np.mean(r2_loo))
# -----------------------------------------------------------
# K-Fold Cross-Validation
# -----------------------------------------------------------
print("\nK-fold Cross-Validation Approach")
print("-------------------------------")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_kf, mae_kf, r2_kf = [], [], []
for train_index, val_index in kf.split(X):
 X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
 y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
 model.fit(X_train_kf, y_train_kf)
 y_pred_kf = model.predict(X_val_kf)
 rmse_kf.append(np.sqrt(mean_squared_error(y_val_kf, y_pred_kf)))
 mae_kf.append(mean_absolute_error(y_val_kf, y_pred_kf))
 r2_kf.append(r2_score(y_val_kf, y_pred_kf))
print("RMSE (K-fold):", np.mean(rmse_kf))
print("MAE (K-fold):", np.mean(mae_kf))
print("R2 Error (K-fold):", np.mean(r2_kf))