import pandas as pd
import numpy as np
from scipy import stats
data = pd.read_csv('Documents/Data.csv')
print(data)
print("One Sample T-Test")
print("------------------")
null_mean = 0
t_stat, p_val = stats.ttest_1samp(data['Values'], null_mean)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)
if p_val < 0.05:
    print("Reject the null hypothesis. The sample mean is significantly different from the null mean.")
else:
    print("Fail to reject the null hypothesis. The sample mean is not significantly different from the null mean.")
data1 = [10, 12, 14, 11]
t_stat, p_val = stats.ttest_ind(data['Values'], data1)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)
if p_val < 0.05:
    print("Reject the null hypothesis. The two sample means are significantly different.")
else:
    print("Fail to reject the null hypothesis. The two sample means are not significantly different.")
after = [70, 68, 60, 66]
t_stat, p_val = stats.ttest_rel(data['Values'], after)
print("T-Statistic:", t_stat)
print("P-Value:", p_val)
if p_val < 0.05:
    print("Reject the null hypothesis. There is a significant difference between before and after.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between before and after.")
print("\nAnalysis of Results")
print("---------------------")
mean1 = np.mean(data['Values'])
std_dev1 = np.std(data['Values'])
print("Sample 1: Mean =", mean1, ", Standard Deviation =", std_dev1)
print("\nAnalysis of Results")
print("---------------------")
mean1 = np.mean(data1)
std_dev1 = np.std(data1)
print("Sample 1: Mean =", mean1, ", Standard Deviation =", std_dev1)
print("\nAnalysis of Results")
print("---------------------")
mean1 = np.mean(after)
std_dev1 = np.std(after)
print("Sample 1: Mean =", mean1, ", Standard Deviation =", std_dev1)