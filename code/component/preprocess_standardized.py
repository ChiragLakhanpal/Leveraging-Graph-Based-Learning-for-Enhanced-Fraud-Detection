import pandas as pd
import argparse
import os
import polars as pl
import numpy as np
import category_encoders as ce
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re

warnings.filterwarnings("ignore")

data = pd.read_csv('/home/ec2-user/Capstone/card_transaction.v1.csv', nrows = 1000000)

data["Errors?"] = data["Errors?"].fillna("No errors")
data["Zip"] = data["Zip"].astype(str)
data["MCC"] = data["MCC"].astype(str)

print(data["Amount"].head())
print(data.dtypes)
print(data.isnull().sum())

def rename_columns(data):

    data.columns = map(str.lower, data.columns)
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('[^\w\s]', '', regex=True)

    return data

def target_binary(data, target):

    label_encoder = preprocessing.LabelEncoder()
    data[target] = label_encoder.fit_transform(data[target])

    return data

def identify_time_variables(data):

    time_columns = []
    time_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?\b'

    for column in data.columns:
        if data[column].dtype == "object":
            if data[column].str.contains(time_pattern).any():
                time_columns.append(column)

    for column in time_columns:
        data[['hour', 'minute']] = data[column].str.split(':', expand=True).astype(float)
        data.drop(columns = column)

    return data

def clean_columns(data):

    pattern = r'[^\w\s.,]'

    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].str.replace('$', '')

    return data

def convert_to_numeric(data):

    pattern_numeric = r'\d+'
    pattern_numeric_period_numeric = r'\d+\.\d+'

    for column in data.columns:
        if data[column].dtype == "object":
            if data[column].str.contains(pattern_numeric_period_numeric).any():
                data[column].astype(float)

    return data


# def fill_missing_values(data):
#
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     for column in data.columns:
#         if data[column].dtype in numerics:
#             if data[column].isnull().sum() > 0 and data.columns.isnull().sum() * 100 / len(df) > 10 and data.columns.isnull().sum() * 100 / len(df) < 50:
#                 mean = data.columns.mean()
#                 data = data.columns.fillna(value=mean, inplace=True)
#             elif data.columns.isnull().sum() > 0 and data.columns.isnull().sum() * 100 / len(df) < 10:
#                 data = data.columns.dropna()
#             else:
#                 data = data.columns.drop()
#         elif data[column].dtype == "object":
#             mode = data.columns.mode()
#             data = data.columns.fillna(value=mode)
#
#     return data

def cat_encoding(data,target):

    ohe_columns = []
    te_columns = []

    for column in data.columns:
        if data[column].dtype == "object" and data[column].nunique() <= 25:
            ohe_columns.append(column)
            encoded_data = pd.get_dummies(data, columns=ohe_columns)
        elif data[column].dtype == "object" and data[column].nunique() > 25:
            te_columns.append(column)
            #te_columns = list(te_columns)
            target_encoder = ce.TargetEncoder(cols = te_columns)
            target_encoded = target_encoder.fit_transform(data, data[target])
            # # encoder = ce.TargetEncoder(return_df=True)
            # target_encoded = encoder.fit_transform(data,y=data['target'],cols = te_columns)

    return encoded_data,target_encoded


def preprocess_data(data):
    data = rename_columns(data)
    target = "is_fraud"
    data = target_binary(data, target)
    data = identify_time_variables(data)
    data = clean_columns(data)
    data['amount'] = data['amount'].astype(float)
    #data = convert_to_numeric(data)
    # data = fill_missing_values(data)
    data = cat_encoding(data,target)

    return data

data_processed = preprocess_data(data)

print(data_processed['amount'].head())
print(data_processed.head())
print(data_processed.dtypes)
print(data_processed.isnull().sum())

