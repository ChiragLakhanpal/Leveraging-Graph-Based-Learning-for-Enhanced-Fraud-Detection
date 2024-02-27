from preprocess import read_data, preprocess_data, split_data
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve
import argparse
import os
import pandas as pd
import polars as pl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# System path
parent_dir = os.getcwd()

def preprocess_xgboost(data):
    # Drop id columns
    data = data.drop(['merchant_id', 'card_id'])
    return data

def best_hyperparams(X_train, X_test, y_train, y_test):
    pass

def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic',
                              device='cuda')
    
    # Fit the model with training data
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    aucpr = auc(recall_curve, precision_curve)
    
    return model, accuracy, precision, recall, f1, cm, roc_auc, aucpr
    
def display_metrics(accuracy, precision, recall, f1, cm, roc_auc, aucpr):
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: {cm}")
    print(f"ROC AUC: {roc_auc}")
    print(f"AUCPR: {aucpr}")


def main():

    # Default paths
    default_data_path = "Data/Raw Data/data.csv"
    default_output_path = "Data/Predictions/XGBoost/predictions.csv'"

    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument('--data_path', type=str, default='Data/Raw Data/data.csv', 
                      help='Path to the Raw data')
    parser.add_argument('--file_type', type=str, default='csv', 
                      help='Type of the file to be read. Options: csv, parquet, xls, etc. Default: csv')
    parser.add_argument('--output_path', type=str, default='Data/Predictions/XGBoost/predictions.csv', 
                      help='Path to output predictions. Default: Data/Predictions/XGBoost/predictions.csv')
    
    args = parser.parse_args()
    
    data_path = os.path.join(parent_dir, args.data_path) if args.data_path == default_data_path else args.data_path
    output_path = os.path.join(parent_dir, args.output_path) if args.output_path == default_output_path else args.output_path
        
    # Read data and display 
    data = read_data(data_path=data_path, file_type=args.file_type)

    # Preprocess the data
    data_processed = preprocess_data(data)
    
    # Preprocess for XGBoost
    data_processed = preprocess_xgboost(data_processed)

    data_processed = data_processed.to_pandas()
    
    X = data_processed.drop(columns=['is_fraud'])
    y = data_processed['is_fraud']

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X = X, y = y,data= data_processed, test_size=0.2)
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Train target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")
    print(f"Number of frauds in train data: {y_train.sum()}")
    print(f"Number of frauds in test data: {y_test.sum()}")
    
    # Train the XGBoost model
    model, accuracy, precision, recall, f1, cm, roc_auc, aucpr = train_xgboost(X_train, X_test, y_train, y_test)
        
    # Display metrics
    display_metrics(accuracy, precision, recall, f1, cm, roc_auc, aucpr)
    

if __name__ == '__main__':
    main()
    
    