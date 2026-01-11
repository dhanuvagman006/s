import pandas as pd
import numpy as np
data = pd.DataFrame({
 'Values': [10, 20, 20, 30, 40, 50, 50, 50, 60, 70]
})
data
print("Central Tendency Measures")
print("----------------------------")
mean = np.mean(data['Values'])
print("Mean:", mean)
median = np.median(data['Values'])
print("Median:", median)
mode = data['Values'].mode().iloc[0]
print("Mode:", mode)
print("\nMeasures of Dispersion")
print("----------------------------")
std_dev = np.std(data['Values'])
print("Standard Deviation:", std_dev)
variance = np.var(data['Values'])
print("Variance:", variance)
mean_dev = np.mean(np.abs(data['Values'] - mean))
print("Mean Deviation:", mean_dev)
q1 = np.percentile(data['Values'], 25)
q3 = np.percentile(data['Values'], 75)
quartile_dev = (q3 - q1) / 2
print("Quartile Deviation:", quartile_dev)
print("\nSummary Statistics:")
print(data['Values'].describe())