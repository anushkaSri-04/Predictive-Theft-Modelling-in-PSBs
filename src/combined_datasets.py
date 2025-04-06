#THIS FILE CONTAINS COMBINING DATASETS AND REMOVING DUPLICATES OR
# NULL VALUES AND PERFORMING LABEL ENCODER I.E. TRANSFORMING STRING VAKUE INTO NUMERICAL VALUE

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

ds1 = pd.read_csv("Datasets/application_data.csv")


# dataset=pd.concat([ds1,ds2], ignore_index=True)
ds1.columns = ds1.columns.str.strip()
dataset=ds1.drop_duplicates()
dataset=dataset.fillna(dataset.mean(numeric_only=True))
# dataset.to_csv("Datasets/merged_loan_dataset.csv", index=False)
# mergedData=pd.read_csv("Datasets/merged_loan_dataset.csv")
#LABE ENCODER
# cols_to_encode = [
#     'NAME_CONTRACT_TYPE', 'GENDER', 'OWN_CAR', 'OWN_REALTY', 'NAME_TYPE_SUITE',
#     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
#     'NAME_HOUSING_TYPE', 'OCCUPATION', 'ORGANIZATION_TYPE'
# ]
label_encoder = LabelEncoder()
for col in ['NAME_CONTRACT_TYPE']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['GENDER']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['OWN_CAR']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['OWN_REALTY']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['NAME_TYPE_SUITE']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['NAME_INCOME_TYPE']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['NAME_EDUCATION_TYPE']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['NAME_FAMILY_STATUS']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['NAME_HOUSING_TYPE']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['OCCUPATION']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

label_encoder = LabelEncoder()
for col in ['ORGANIZATION_TYPE']:
    dataset[col] = label_encoder.fit_transform(dataset[col])

most_common = dataset['OCCUPATION'].mode()[0]
dataset['OCCUPATION'].fillna(most_common, inplace=True)
most_common1 = dataset['NAME_TYPE_SUITE'].mode()[0]
dataset['NAME_TYPE_SUITE'].fillna(most_common1, inplace=True)

#PRINTING COLUMNS WITH NULL / MISSING VALUES 
print("missing value after: ")
print([col for col in dataset.columns if dataset[col].isnull().sum()>0])

print(dataset.isnull().sum())
dataset = dataset.drop_duplicates()
dataset.columns = dataset.columns.str.strip()
mergedData = dataset.fillna(dataset.mean(numeric_only=True))

mergedData.to_csv("Datasets/FinalDataset.csv", index=False)
FinalDataset = pd.read_csv("Datasets/FinalDataset.csv")
# print("final datset columns: ",FinalDataset.columns)