# Essential Libraries
from preprocess import read_data, preprocess_data, split_data
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, roc_auc_score, auc, precision_recall_curve
import argparse
import os
import pandas as pd
import polars as pl
import prettytable
import numpy as np

# Hyperparameter tuning
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Visualizations and plotting
import matplotlib.pyplot as plt
import seaborn as sns
import dtreeviz


# System path
parent_dir = os.getcwd()

# Set random seed
seed = 42
np.random.seed(seed)

def preprocess_xgboost(data):
    # Drop id columns
    data = data.drop(['merchant_id', 'card_id'])
    return data

def best_hyperparams(X_train, X_test, y_train, y_test):
    pass

def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(use_label_encoder=False, 
                              eval_metric='logloss', 
                              objective='binary:logistic',
                              device='cuda',
                              tree_method='gpu_hist',
                              sampling_method = 'gradient_based')
    
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
    
def display_metrics(accuracy, precision, recall, f1, cm, roc_auc, aucpr, model, X_train, y_train, y_scores):
    # Print metrics
    results = prettytable.PrettyTable()
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    results.add_row(["ROC AUC", roc_auc])
    results.add_row(["AUCPR", aucpr])
    print(results)
    
    # Visualizing confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Assuming y_scores is the output from a decision function or probability estimates for the positive class
    precision_curve, recall_curve, _ = precision_recall_curve(y_train, y_scores)
    fpr, tpr, _ = roc_curve(y_train, y_scores)
    roc_auc = auc(fpr, tpr)

    # Visualize the AUCPR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()

    # Visualize the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, marker='.', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    feature_names = X_train.columns
    class_names = ['Not Fraud', 'Fraud']
    
    # viz = dtreeviz(model,
    #             X_train,
    #             y_train,
    #             target_name="target",
    #             feature_names=feature_names,
    #             class_names=class_names)

    # # For displaying inline in Jupyter Notebook or similar environments, you can directly use
    # viz.view()
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
    
    # Print train and test data shapes and number of frauds through prettytable
    table = prettytable.PrettyTable()
    table.field_names = ["Data", "Rows", "Columns", "Frauds", "Non-Frauds"]
    table.add_row(["Train", X_train.shape[0], X_train.shape[1], y_train.sum(), y_train.shape[0] - y_train.sum()])
    table.add_row(["Test", X_test.shape[0], X_test.shape[1], y_test.sum(), y_test.shape[0] - y_test.sum()])
    print(table)
    
    # Train 
    model, accuracy, precision, recall, f1, cm, roc_auc, aucpr = train_xgboost(X_train, X_test, y_train, y_test)
        
    # Display metrics
    display_metrics(accuracy= accuracy, 
                    precision= precision, 
                    recall= recall, 
                    f1= f1, 
                    cm= cm, 
                    roc_auc= roc_auc, 
                    aucpr= aucpr, 
                    model= model, 
                    X_train= X_train, 
                    y_train= y_train, 
                    y_scores= model.predict_proba(X_train)[:, 1])
    

if __name__ == '__main__':
    main()
    
    