import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

ds1 = pd.read_csv("Datasets/application_data.csv")
ds1.columns = ds1.columns.str.strip()
dataset = ds1.drop_duplicates()
dataset = dataset.fillna(dataset.mean(numeric_only=True))

label_encoder = LabelEncoder()
categorical_cols = dataset.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if dataset[col].isnull().any():
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)
    try:
        dataset[col] = label_encoder.fit_transform(dataset[col].astype(str))
    except Exception as e:
        print(f"Could not encode column {col}: {e}")

if 'OCCUPATION' in dataset.columns:
    dataset['OCCUPATION'].fillna(dataset['OCCUPATION'].mode()[0], inplace=True)

if 'NAME_TYPE_SUITE' in dataset.columns:
    dataset['NAME_TYPE_SUITE'].fillna(dataset['NAME_TYPE_SUITE'].mode()[0], inplace=True)

columns_to_drop = [
    'FLAG_WORK_PHONE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'FLAG_EMAIL',
    'NAME_CONTRACT_TYPE', 'REG_REGION_NOT_WORK_REGION', 'FLAG_EMP_PHONE', 'LIVE_REGION_NOT_WORK_REGION',
    'REG_REGION_NOT_LIVE_REGION', 'FLAG_CONT_MOBILE', 'FLAG_MOBIL', 'REGION_RATING_CLIENT_W_CITY',
    'REGION_RATING_CLIENT', 'NAME_TYPE_SUITE', 'ORGANIZATION_TYPE', 'OWN_REALTY'
]
dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns])

if 'DAYS_BIRTH' in dataset.columns:
    dataset['AGE'] = (-dataset['DAYS_BIRTH']) / 365.25
    dataset['AGE'] = dataset['AGE'].astype(int)

if 'DAYS_EMPLOYED' in dataset.columns:
    dataset['YEARS_EMPLOYED'] = (-dataset['DAYS_EMPLOYED']) / 365.25

if 'DAYS_REGISTRATION' in dataset.columns:
    dataset['YEARS_REGISTRATION'] = (-dataset['DAYS_REGISTRATION']) / 365.25

if 'DAYS_ID_PUBLISH' in dataset.columns:
    dataset['YEARS_ID_PUBLISH'] = (-dataset['DAYS_ID_PUBLISH']) / 365.25

dataset = dataset.drop(columns=[col for col in ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'] if col in dataset.columns])
dataset.to_csv("Datasets/FinalDataset.csv", index=False)
# #Dataset 2
# ds2 = pd.read_csv("Datasets/loan_data_1.csv")
# ds2.columns = ds2.columns.str.strip()
# ds2 = ds2.drop_duplicates()
# ds2 = ds2.fillna(ds2.mean(numeric_only=True))

# categorical_cols_ds2 = ds2.select_dtypes(include=['object']).columns
# for col in categorical_cols_ds2:
#     if ds2[col].isnull().any():
#         ds2[col].fillna(ds2[col].mode()[0], inplace=True)
#     try:
#         ds2[col] = label_encoder.fit_transform(ds2[col].astype(str))
#     except Exception as e:
#         print(f"Could not encode column {col} in ds2: {e}")


# FinalDataset = pd.concat([dataset, ds2], ignore_index=True)
# FinalDataset.drop_duplicates(inplace=True)
# FinalDataset.fillna(FinalDataset.mean(numeric_only=True), inplace=True)


# FinalDataset.to_csv("Datasets/MergedDataset.csv", index=False)


# print("Merged dataset created and saved as 'Datasets/MergedDataset.csv'")
# print("Final columns with missing values (if any):")
# print([col for col in FinalDataset.columns if FinalDataset[col].isnull().sum() > 0])
