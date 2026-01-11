#11. Program to implement the working of linear discriminant
#analysis using iris dataset and visualize the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
# -----------------------------------------------------
# Loading the Iris Dataset
# -----------------------------------------------------
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
# -----------------------------------------------------
# Standardizing the Features
# -----------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('target', axis=1))
# -----------------------------------------------------
# PCA for Comparison (2 Components)
# -----------------------------------------------------
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
# -----------------------------------------------------
# Applying LDA (2 Components)
# -----------------------------------------------------
lda = LinearDiscriminantAnalysis(n_components=2)
data_lda = lda.fit_transform(data_scaled, data['target'])
# -----------------------------------------------------
# Plotting LDA Results
# -----------------------------------------------------
plt.figure(figsize=(10, 8))
plt.scatter(data_lda[:, 0], data_lda[:, 1], c=data['target'])
plt.title("LDA of Iris Dataset")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.show()
# -----------------------------------------------------
# Plotting PCA Results
# -----------------------------------------------------
plt.figure(figsize=(10, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['target'])
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
# -----------------------------------------------------
# Analysis of Results
# -----------------------------------------------------
print("\nAnalysis of Results")
print("---------------------")
print("Shape of Original Data:", data.shape)
print("Shape of Data after LDA:", data_lda.shape)
print("Number of Components Retained (LDA):", lda.n_components)
print("Explained Variance Ratio (LDA):", lda.explained_variance_ratio_)
print("LDA Coefficients:\n", lda.coef_)