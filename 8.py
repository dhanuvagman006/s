import pandas as pd
import numpy as np
from scipy import stats
data = pd.DataFrame({
 "Group": ["A", "B", "C", "A"],
 "Values": [10, 18, 20, 15]
})
print(data)
print("One-way ANOVA")
print("-------------")
Groups= data['Group']
Groups
groupA = data[data['Group'] == 'A']['Values']
groupB = data[data['Group'] == 'B']['Values']
groupC = data[data['Group'] == 'C']['Values']
f_stat, p_val = stats.f_oneway(groupA, groupB, groupC)
print("F-Statistic:", f_stat)
print("P-Value:", p_val)
if p_val < 0.05:
 print("Reject the null hypothesis. The means of the groups are significantly different.")
else:
 print("Fail to reject the null hypothesis. The means of the groups are not significantly different.")
data = pd.DataFrame({
 "Group1": ["A", "B", "C", "A"],
 "Group2":["F","M","M","F"],
 "Values": [10, 18, 20, 15]
})
print(data)
Groups1= data['Group1']
Groups1
Groups2= data['Group2']
Groups2
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Values ~ C(Group1) + C(Group2) + C(Group1):C(Group2)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
print("Group1:")
print(" F-Statistic:", anova_table.loc["C(Group1)", "F"])
print(" P-Value :", anova_table.loc["C(Group1)", "PR(>F)"])
print("\nGroup2:")
print(" F-Statistic:", anova_table.loc["C(Group2)", "F"])
print(" P-Value :", anova_table.loc["C(Group2)", "PR(>F)"])
p_val1 = anova_table.loc["C(Group1)", "PR(>F)"]
p_val2 = anova_table.loc["C(Group2)", "PR(>F)"]
if p_val1 < 0.05 and p_val2 < 0.05:
 print("Reject the null hypothesis. The means of the groups are significantly different.")
else:
 print("Fail to reject the null hypothesis. The means of the groups are not significantly different.")
print("Tukey's HSD Test")
print("-----------------")
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(
 endog=data["Values"],
 groups=data["Group1"],
 alpha=0.05
)
print(tukey)
print()
