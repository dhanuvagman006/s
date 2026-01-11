# 2. Program on Data Transformation: String Manipulation,
# Regular Expressions


import pandas as pd
import re
#Creating dataset
dataset = pd.DataFrame({
'Name': ['John Smith', 'Mary Johnson', 'David Lee'],
'Address': ['123 Main St', '456 Elm St', '789 Oak St']
})
dataset
#Convert to lowercase
dataset['Name'] = dataset['Name'].str.lower()
dataset['Address'] = dataset['Address'].str.lower()
dataset
#Convert to uppercase
dataset['Name'] = dataset['Name'].str.upper()
dataset['Address'] = dataset['Address'].str.upper()
dataset
#Strip whitespace(removes extra spaces at the beginning and end)
dataset['Name'] = dataset['Name'].str.strip()
dataset['Address'] = dataset['Address'].str.strip()
dataset
#Replace Strings(finds a word and replaces it)
dataset['Name'] = dataset['Name'].str.replace('JOHN', 'JONATHAN')
dataset['Address'] = dataset['Address'].str.replace('ST', 'STREET')
dataset
#Creating another dataset(to apply Regular expression)
dataset = pd.DataFrame({
'Email': ['john@example.com', 'mary@example.org', 'david@example.net'],
'Phone': ['123-456-7890', '098-765-4321', '555-123-4567']
})
dataset
dataset['Domain'] = dataset['Email'].str.extract(r'@(.*)', expand=False)
dataset
dataset['Valid Phone'] = dataset['Phone'].str.contains(
 r'^\d{3}-\d{3}-\d{4}$', regex=True)
dataset
#Extract phone number components
dataset[['AreaCode','Prefix','LineNumber']]=dataset['Phone'].str.extract(r'(\d{3})-(\d{3})-(\d{4})')
Dataset