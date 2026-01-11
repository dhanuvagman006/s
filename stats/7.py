#7. Program to implement one sample, two sample and paired sample
#t-tests for a sample data and analyse the results.

import pandas as pd
import numpy as np
from scipy import stats

# Load sample data
data = pd.read_csv('Documents/Data.csv')
print(data)

# One Sample T-Test
print("One Sample T-Test")
print("------------------")
# Define the null hypothesis mean
null_mean = 0
# Perform one sample t-test
t_stat, p_val = stats.ttest_1samp(data['Values'], null_mean)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)

# Interpret the results
if p_val < 0.05:
    print("Reject the null hypothesis. The sample mean is significantly different from the null mean.")
else:
    print("Fail to reject the null hypothesis. The sample mean is not significantly different from the null mean.")

# Sample data
data1 = [10, 12, 14, 11]
# Perform two-sample t-test
t_stat, p_val = stats.ttest_ind(data['Values'], data1)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)

# Interpretation
if p_val < 0.05:
    print("Reject the null hypothesis. The two sample means are significantly different.")
else:
    print("Fail to reject the null hypothesis. The two sample means are not significantly different.")

# Before and after data
after = [70, 68, 60, 66]
# Perform Paired Sample T-Test
t_stat, p_val = stats.ttest_rel(data['Values'], after)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)

# Interpret the results
if p_val < 0.05:
    print("Reject the null hypothesis. There is a significant difference between before and after.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between before and after.")

# Analyze the results
print("\nAnalysis of Results")
print("---------------------")
# Calculate the mean and standard deviation of each sample
mean1 = np.mean(data['Values'])
std_dev1 = np.std(data['Values'])
print("Sample 1: Mean =", mean1, ", Standard Deviation =", std_dev1)

# Analyze the results
print("\nAnalysis of Results")
print("---------------------")
# Calculate the mean and standard deviation of each sample
mean1 = np.mean(data1)
std_dev1 = np.std(data1)
print("Sample 1: Mean =", mean1, ", Standard Deviation =", std_dev1)

# Analyze the results
print("\nAnalysis of Results")
print("---------------------")
# Calculate the mean and standard deviation of each sample
mean1 = np.mean(after)
std_dev1 = np.std(after)
print("Sample 1: Mean =", mean1, ", Standard Deviation =", std_dev1)