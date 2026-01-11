#10. Program to implement PCA for Wisconsin dataset, visualize and
#analyze the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# -----------------------------------------------------
# Creating a Sample Dataset (Instead of Loading CSV)
# -----------------------------------------------------
# Simulating a dataset similar to the Wisconsin dataset
np.random.seed(42)
datas = pd.DataFrame({
 'diagnosis': np.random.choice(['M', 'B'], 50),
 'mean_radius': np.random.uniform(5, 20, 50),
 'mean_texture': np.random.uniform(10, 30, 50),
 'mean_perimeter': np.random.uniform(40, 150, 50),
 'mean_area': np.random.uniform(200, 2500, 50),
 'mean_smoothness': np.random.uniform(0.05, 0.15, 50),
 'Unnamed: 32': np.random.randint(0, 2, 50) # Dummy unwanted column
})
# -----------------------------------------------------
# Removing Unwanted Columns
# -----------------------------------------------------
data = datas.drop('diagnosis', axis=1) # Removing label column
data1 = data.drop('Unnamed: 32', axis=1) # Removing unnecessary column
# -----------------------------------------------------
# Standardizing the Data
# -----------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data1)
# -----------------------------------------------------
# Performing PCA
# -----------------------------------------------------
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
# -----------------------------------------------------
# Printing Important PCA Information
# -----------------------------------------------------
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# -----------------------------------------------------
# Plotting PCA Scatter Plot
# -----------------------------------------------------
plt.figure(figsize=(10, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.title("PCA Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
# -----------------------------------------------------
# Plotting Cumulative Explained Variance
# -----------------------------------------------------
plt.figure(figsize=(10, 8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()
# -----------------------------------------------------
# Complete Analysis
# -----------------------------------------------------
print("\nAnalysis of Results")
print("---------------------")
print("Shape of Original Data:", data1.shape)
print("Shape of Data after PCA:", data_pca.shape)
print("Number of Components Retained:", pca.n_components_)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Singular Values:", pca.singular_values_)
print("Components (Principal Axes):\n", pca.components_)