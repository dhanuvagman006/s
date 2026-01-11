# 1. Program on data wrangling: Combining and merging datasets,
# Reshaping and Pivoting
# ---------------------------------------------------
import pandas as pd
# ----------------------------------------------------
# PART 1: COMBINING AND MERGING DATASETS
# ----------------------------------------------------
print("Combining and Merging Datasets")
print("-------------------------------")
# Create first sample dataset
dataset1 = pd.DataFrame({
 'Name': ['John', 'Mary', 'David'],
 'Age': [25, 31, 42]
})
# Create second sample dataset
dataset2 = pd.DataFrame({
 'Name': ['Emily', 'Michael', 'Sarah'],
 'Age': [28, 35, 38]
})
# Concatenating the two datasets (stacking them vertically)
combined_dataset = pd.concat([dataset1, dataset2])
print("Concatenated Dataset:")
print(combined_dataset)
# ----------------------------------------------------
# PART 2: MERGING DATASETS USING A COMMON COLUMN
# ----------------------------------------------------
# Create first dataset with ID and Name
dataset1 = pd.DataFrame({
 'ID': [1, 2, 3],
 'Name': ['John', 'Mary', 'David']
})
# Create second dataset with ID and Age
dataset2 = pd.DataFrame({
 'ID': [1, 2, 3],
 'Age': [25, 31, 42]
})
# Merge the two datasets using the ID column
merged_dataset = pd.merge(dataset1, dataset2, on='ID')
print("\nMerged Dataset:")
print(merged_dataset)
# ----------------------------------------------------
# PART 3: RESHAPING AND PIVOTING
# ----------------------------------------------------
print("\nReshaping and Pivoting")
print("-------------------------------")
# Create a sample dataset with repeated IDs and Years
dataset = pd.DataFrame({
 'ID': [1, 1, 2, 2],
 'Year': [2018, 2019, 2018, 2019],
 'Sales': [100, 120, 80, 90]
})
# Reshape using pivot_table: makes Years into columns
reshaped_dataset = pd.pivot_table(dataset, values='Sales', index='ID', columns='Year')
print("Reshaped Dataset:")
print(reshaped_dataset)
# ----------------------------------------------------
# PART 4: PIVOTING USING MELT (Long Format Conversion)
# ----------------------------------------------------
dataset = pd.DataFrame({
 'ID': [1, 2],
 '2018': [100, 80],
 '2019': [120, 90]
})
# Melt converts wide data to long data format
pivoted_dataset = pd.melt(
 dataset,
 id_vars='ID', # Keep ID as it is
 value_vars=['2018', '2019'],# Columns to unpivot
 var_name='Year', # New column name for years
 value_name='Sales' # New column name for sales values
)
print("\nPivoted Dataset:")
print(pivoted_dataset)