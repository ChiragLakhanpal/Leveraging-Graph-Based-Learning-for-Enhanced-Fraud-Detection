import argparse 
import os
import polars as pl
import pandas as pd
import numpy as np
import category_encoders as ce
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# System path
parent_dir = os.getcwd()

def read_data(data_path, file_type="csv"):
    """Read data from the specified path and file type."""
    try:
        if file_type == "csv":
            return pl.read_csv(data_path)
        elif file_type == "parquet":
            return pl.read_parquet(data_path)
        elif file_type == "xls" or file_type == "xlsx":
            pd_df = pd.read_excel(data_path)
            return pl.from_pandas(pd_df)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at specified path: {data_path}")
    except Exception as e:
        raise Exception(f"Error reading {data_path}: {e}")


def save_data(data, output_path, file_type="csv"):
    """Save preprocessed data to the specified path and file type."""
    # Extract directory part from output_path
    output_dir = os.path.dirname(output_path)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if file_type == "csv":
            data.write_csv(output_path)
        elif file_type == "parquet":
            data.write_parquet(output_path)
        elif file_type == "xls" or file_type == "xlsx":
            pd_df = data.to_pandas()
            pd_df.to_excel(output_path, index=False)
        else:
            data.write_csv(output_path)
    except Exception as e:
        raise Exception(f"Error saving data to {output_path}: {e}")


def preprocess_data(data):
    
    # Rename columns
    data = data.rename({'User':'user',
                         'Card':'card', 
                         'Year':'year',
                         'Month':'month',
                         'Day':'day',
                         'Time':'time',
                         'Amount':'amount',
                         'Use Chip':'use_chip', 
                         'Merchant Name':'merchant_id', 
                         'Merchant City':'merchant_city', 
                         'Merchant State':'merchant_state',
                         'Zip':'zip', 
                         'Errors?':'errors',
                         'Is Fraud?':'is_fraud'})
    
    # Create card_id (Node)
    data = data.with_columns((pl.col('user').cast(str) + pl.col('card').cast(str)).alias('card_id'))
    data = data.drop(['user', 'card'])
       
    # Dealing with Date and time variables
    data = data.with_columns([
            pl.col('time').str.strptime(pl.Datetime, format='%H:%M').dt.hour().alias('hour'),
            pl.col('time').str.strptime(pl.Datetime, format='%H:%M').dt.minute().alias('minute')
        ]).drop('time')   
    
    # Clean Amount column
    data = data.with_columns(pl.col('amount').str.replace(r'\$', '').cast(pl.Float64).alias('amount'))
    
    # One-hot encode categorical columns
    data = data.to_dummies("use_chip", drop_first=True)
    data = data.to_dummies("errors", drop_first=True)
        
    # Convert is_fraud to binary
    data = data.with_columns(pl.when(pl.col('is_fraud') == 'Yes').then(1).otherwise(0).alias('is_fraud'))

    # Define categorical columns
    for column in ['zip', 'MCC', 'merchant_city', 'merchant_state', 'merchant_id', 'card_id']:
        data = data.with_columns(pl.col(column).cast(pl.Utf8).cast(pl.Categorical)) 

    # Impute missing values of zip, merchant_state and MCC with mode
    
    data_pd = data.select(['zip', 'merchant_state', 'MCC']).to_pandas()

    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = imputer.fit_transform(data_pd)
    
    data_imputed_df = pd.DataFrame(data_imputed, columns=['zip', 'merchant_state', 'MCC'])
    
    data_imputed_pl = pl.from_pandas(data_imputed_df)
    
    data = data.drop(['zip', 'merchant_state', 'MCC']).hstack(data_imputed_pl)
            
        
    # target encoding high cardinality columns
    high_cardinality = ['zip', 'merchant_city', 'merchant_state','MCC']
    
    for column in high_cardinality:
        # Calculate the mean of 'is_fraud' 
        mean_encoded = data.groupby(column).agg(pl.col('is_fraud').mean().alias('target_encoded'))
        
        data = data.join(mean_encoded, on=column, how='left')
        
        data = data.drop(column).rename({'target_encoded': column})
        
    # Apply the renaming
    data = data.rename({col: col.rstrip(",") for col in data.columns if ',' in col})

    return data
    
def split_data(X, y, data=None, test_size=0.2):
    """Split the data into train and test sets."""
    try:
        if data is not None and 'is_fraud' in data:
            return train_test_split(X, y, test_size=test_size, random_state=42, stratify=data['is_fraud'])
        else:
            return train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

        
def main():
    parser = argparse.ArgumentParser(description="Preprocess data")

    # Default paths
    default_data_path = "Data/Raw Data/data.csv"
    default_output_path = "Data/Processed Data/data.csv"

    parser.add_argument("--data_path", type=str, default=default_data_path, 
                        help=f"Path to the data file. Default: {default_data_path}")
    parser.add_argument("--output_path", type=str, default=default_output_path,
                        help=f"Path to the output file. Default: {default_output_path}")
    parser.add_argument("--file_type", type=str, default="csv",
                        help="Type of the file to be read. Options: csv, parquet, xls, etc. Default: csv")
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set. Default: 0.2')

    args = parser.parse_args()

    data_path = os.path.join(parent_dir, args.data_path) if args.data_path == default_data_path else args.data_path
    output_path = os.path.join(parent_dir, args.output_path) if args.output_path == default_output_path else args.output_path
    try:
        # Read data and display the first few rows
        data = read_data(data_path=data_path, file_type=args.file_type)
        print(data.head())

        # Preprocess the data
        data_processed = preprocess_data(data)
        
        # Print the shape of the data before and after processing
        print(f"Data shape before processing: {data.shape}")
        print(f"Data shape after processing: {data_processed.shape}")
        
        # Save the processed data
        save_data(data=data_processed, output_path=output_path, file_type=args.file_type)
        print(f"Data successfully processed and saved to {output_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()