from preprocess import read_data, preprocess_data, split_data
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# System path
parent_dir = os.getcwd()

def preprocess_xgboost(data):
    # Drop id columns
    data = data.drop(columns=['merchant_id', 'card_id'])
    return data

def best_hyperparams(X_train, X_test, y_train, y_test):
    pass

def train_xgboost(X_train, X_test, y_train, y_test):
    # Train the XGBoost model
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy
    

def main():

    # Default paths
    default_data_path = "Data/Processed Data/data.csv"
    default_output_path = "Data/Predictions/XGBoost/predictions.csv'"

    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='Data/Processed Data/data.csv', 
                      help='Path to the processed data')
    args.add_argument('--file_type', type=str, default='csv', 
                      help='Type of the file to be read. Options: csv, parquet, xls, etc. Default: csv')
    args.add_argument('--output_path', type=str, default='Data/Predictions/XGBoost/predictions.csv', 
                      help='Path to output predictions. Default: Data/Predictions/XGBoost/predictions.csv')
    
    data_path = os.path.join(parent_dir, args.data_path) if args.data_path == default_data_path else args.data_path
    output_path = os.path.join(parent_dir, args.output_path) if args.output_path == default_output_path else args.output_pat
        
    data_path = args.data_path
    # Read data and display 
    data = read_data(data_path=data_path, file_type=args.file_type)

    # Preprocess the data
    data_processed = preprocess_data(data)
    
    # Preprocess for XGBoost
    data_processed = preprocess_xgboost(data_processed)
    
    X = data_processed.drop(columns=['is_fraud'])
    y = data_processed['is_fraud']
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X = X, y = y, test_size=0.2)
    
    # Train the XGBoost model
    model, accuracy = train_xgboost(X_train, X_test, y_train, y_test)
    
    print(f"Accuracy: {accuracy}")
    

if __name__ == '__main__':
    main()
    
    