# 4. Program to measure central tendency and measures of dispersion:
# Mean, Median, Mode, Standard Deviation, Variance, Mean deviation
# and Quartile deviation for a frequency distribution/data.


import pandas as pd
import numpy as np
#load the dataset
# Direct data (no CSV file)
data = pd.DataFrame({
 'Values': [10, 20, 20, 30, 40, 50, 50, 50, 60, 70]
})
data
print("Central Tendency Measures")
print("----------------------------")
#mean calculation
mean = np.mean(data['Values'])
print("Mean:", mean)
#Median calucation
median = np.median(data['Values'])
print("Median:", median)
#Mode Calculation
mode = data['Values'].mode().iloc[0]
print("Mode:", mode)
#Standard deviation
print("\nMeasures of Dispersion")
print("----------------------------")
#standard deviation
std_dev = np.std(data['Values'])
print("Standard Deviation:", std_dev)
#Variance
variance = np.var(data['Values'])
print("Variance:", variance)
#Mean Deviation
mean_dev = np.mean(np.abs(data['Values'] - mean))
print("Mean Deviation:", mean_dev)
#Quartile Deviation
q1 = np.percentile(data['Values'], 25)
q3 = np.percentile(data['Values'], 75)
quartile_dev = (q3 - q1) / 2
print("Quartile Deviation:", quartile_dev)
print("\nSummary Statistics:")
print(data['Values'].describe())