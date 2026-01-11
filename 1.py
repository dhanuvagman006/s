import pandas as pd
print("Combining and Merging Datasets")
print("-------------------------------")
dataset1 = pd.DataFrame({
 'Name': ['John', 'Mary', 'David'],
 'Age': [25, 31, 42]
})
dataset2 = pd.DataFrame({
 'Name': ['Emily', 'Michael', 'Sarah'],
 'Age': [28, 35, 38]
})
combined_dataset = pd.concat([dataset1, dataset2])
print("Concatenated Dataset:")
print(combined_dataset)
dataset1 = pd.DataFrame({
 'ID': [1, 2, 3],
 'Name': ['John', 'Mary', 'David']
})
dataset2 = pd.DataFrame({
 'ID': [1, 2, 3],
 'Age': [25, 31, 42]
})
merged_dataset = pd.merge(dataset1, dataset2, on='ID')
print("\nMerged Dataset:")
print(merged_dataset)
print("\nReshaping and Pivoting")
print("-------------------------------")
dataset = pd.DataFrame({
 'ID': [1, 1, 2, 2],
 'Year': [2018, 2019, 2018, 2019],
 'Sales': [100, 120, 80, 90]
})
reshaped_dataset = pd.pivot_table(dataset, values='Sales', index='ID', columns='Year')
print("Reshaped Dataset:")
print(reshaped_dataset)
dataset = pd.DataFrame({
 'ID': [1, 2],
 '2018': [100, 80],
 '2019': [120, 90]
})
pivoted_dataset = pd.melt(
 dataset,
 id_vars='ID',
 value_vars=['2018', '2019'],
 var_name='Year',
 value_name='Sales'
)
print("\nPivoted Dataset:")
print(pivoted_dataset)