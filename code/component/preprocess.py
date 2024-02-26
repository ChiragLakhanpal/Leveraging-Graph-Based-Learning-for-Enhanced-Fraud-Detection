import argparse 
import os
import pandas as pd
import numpy as np
import category_encoders as ce
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# System path
parent_dir = os.getcwd()

def read_data(data_path, file_type="csv"):
    """Read data from the specified path and file type."""
    try:
        if file_type == "csv":
            return pd.read_csv(data_path)
        elif file_type == "parquet":
            return pd.read_parquet(data_path)
        elif file_type == "xls" or file_type == "xlsx":
            return pd.read_excel(data_path)
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
            data.to_csv(output_path, index=False)
        elif file_type == "parquet":
            data.to_parquet(output_path, index=False)
        elif file_type == "xls" or file_type == "xlsx":
            data.to_excel(output_path, index=False)
        else:
            data.to_csv(output_path, index=False) 
    except Exception as e:
        raise Exception(f"Error saving data to {output_path}: {e}")

def preprocess_data(data):
    
    # Rename columns
    data.rename(columns={'User':'user',
                         'Card':'card', 
                         'Year':'year',
                         'Month':'month',
                         'Day':'day',
                         'Time':'time',
                         'Amount':'amount',
                         'Use Chip':'use_chip', 
                         'Merchant Name':'merchant_name', 
                         'Merchant City':'merchant_city', 
                         'Merchant State':'merchant_state',
                         'Zip':'zip', 
                         'Errors?':'errors',
                         'Is Fraud?':'is_fraud'}, inplace=True)
    
    # Create card_id (Node)
    data['card_id'] = data['user'].astype(str) + data['card'].astype(str)
    data = data.drop(['user', 'card'], axis=1)
    
    # Create merchant_id (Node)
    data.rename(columns={'merchant_name':'merchant_id'}, inplace=True)
    
       
    # Dealing with Date and time variables
    data['time'] = pd.to_datetime(data['time'], format='%H:%M')
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    data.drop('time', axis=1, inplace=True)    

    # Clean Amount column
    data['amount'] = data['amount'].str.replace('$', '')
    data['amount'] = data['amount'].astype('float')
            
    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['errors'], dtype=int)
    data = pd.get_dummies(data, columns=['use_chip'], dtype=int)
    
        
    # Convert is_fraud to binary
    data['is_fraud'] = data['is_fraud'].map({'Yes': 1, 'No': 0})
    
    # drop null values
    data = data.dropna(subset=['zip', 'MCC'])

    # Define categorical columns
    categorical_columns = ['zip', 'MCC', 'merchant_city', 'merchant_state', 'merchant_id', 'card_id']
    for column in categorical_columns:
        data[column] = data[column].astype('category')
    # target encoding high cardinality columns
    high_cardinality = ['zip', 'merchant_id', 'merchant_city', 'merchant_state','MCC']
    for column in high_cardinality:
        target_encoder = ce.TargetEncoder()
        data[column] = target_encoder.fit_transform(data[column], data['is_fraud'])
    
    # # Clearning columns names 
    rename_cols = ["errors_Bad CVV,","errors_Bad Card Number,","errors_Bad Expiration,","errors_Bad PIN,","errors_Bad Zipcode,","errors_Insufficient Balance,","errors_Technical Glitch,"]
    rename_mapping = {name: name.rstrip(",") for name in rename_cols}
    data.rename(columns=rename_mapping, inplace=True)
    
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