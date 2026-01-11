import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
data = pd.DataFrame({
 'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'y': [2, 4, 5, 4, 5, 7, 8, 9, 10, 12],
 'A': [10, 13, 12, 15, 14, 18, 20, 22, 21, 25],
 'B': [5, 3, 4, 6, 8, 7, 6, 5, 9, 10]
})
print("Correlation")
print("-----------")
corr_matrix = data.corr()
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()
print("\nRank Correlation")
print("----------------")
rank_corr_matrix = data.rank().corr()
print(rank_corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(rank_corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title("Rank Correlation Matrix")
plt.show()
print("\nRegression")
print("-----------")
X = data['X']
y = data['y']
model = LinearRegression()
model.fit(X.values.reshape(-1, 1), y)
print("Coefficient:", model.coef_)
plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.plot(X, model.predict(X.values.reshape(-1, 1)), color='red')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
plt.figure(figsize=(10, 8))
plt.scatter(y, model.predict(X.values.reshape(-1, 1)) - y)
plt.title("Residual Plot")
plt.xlabel("y")
plt.ylabel("Residuals")
plt.show()
print("\nPearson Correlation")
print("------------------")
pearson_corr, _ = pearsonr(X, y)
print("Pearson Correlation Coefficient:", pearson_corr)
print("\nSpearman Rank Correlation")
print("------------------------")
spearman_corr, _ = spearmanr(X, y)
print("Spearman Rank Correlation Coefficient:", spearman_corr)