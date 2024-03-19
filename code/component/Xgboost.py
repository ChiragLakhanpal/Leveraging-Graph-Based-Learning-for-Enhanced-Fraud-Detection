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
# Class Imbalance
from imblearn.over_sampling import SMOTE
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


def Hyperparameter_tuning(X_train, X_test, y_train, y_test):
    space = {
        'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'n_estimators': hp.choice('n_estimators', np.arange(100, 1000, 100, dtype=int)),
    }
    
    def objective(space):
        model = xgb.XGBClassifier(
            max_depth=int(space['max_depth']),
            min_child_weight=space['min_child_weight'],
            subsample=space['subsample'],
            colsample_bytree=space['colsample_bytree'],
            learning_rate=space['learning_rate'],
            n_estimators=int(space['n_estimators']),
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic',
            tree_method='gpu_hist',
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = f1_score(y_test, y_pred)
        
        
        return {'loss': -roc_auc, 'status': STATUS_OK}
    
    trials = Trials()
    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)
    
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)
    
    # print trials
    for trial in trials.trials[:2]:
        print(trial)
        
    return best_hyperparams

def train_xgboost(X_train, X_test, y_train, y_test, hyperparameters=None):
    if hyperparameters:
        model = xgb.XGBClassifier(use_label_encoder=False, 
                                  eval_metric='logloss', 
                                  objective='binary:logistic',
                                  n_jobs=-1,
                                  device='cuda',
                                  tree_method='gpu_hist',
                                  **hyperparameters) 
    else:
        model = xgb.XGBClassifier(use_label_encoder=False, 
                                  eval_metric='logloss', 
                                  objective='binary:logistic',
                                  n_jobs=-1,
                                  device='cuda',
                                  tree_method='gpu_hist'
                                  ) 
    # Fit and evaluate the model as before
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
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    
    
    return model, accuracy, precision, recall, f1, cm, roc_auc, aucpr, y_prob, fpr, tpr, roc_thresholds, precision_curve, recall_curve

def dataset_stats(data, X_train, X_test, y_train, y_test):
    table = prettytable.PrettyTable()
    # Display dataset stats
    print("Dataset Stats")
    
    table.field_names = ["Data", "Rows", "Columns", "Frauds", "Non-Frauds", "Fraud Percentage"]
    table.add_row(["Complete Dataset", data.shape[0], data.shape[1], data['is_fraud'].sum(), data.shape[0] - data['is_fraud'].sum(), 
                   f"{round(data['is_fraud'].sum() / data.shape[0] * 100, 2)}%"])
    table.add_row(["Train", X_train.shape[0], X_train.shape[1], y_train.sum(), y_train.shape[0] - y_train.sum(), 
                   f"{round(y_train.sum() / y_train.shape[0] * 100, 2)}%"])
    table.add_row(["Test", X_test.shape[0], X_test.shape[1], y_test.sum(), y_test.shape[0] - y_test.sum(),
                   f"{round(y_test.sum() / y_test.shape[0] * 100, 2)}%"])
    print(table)

    
    
def display_metrics(title, model, accuracy, precision, recall, f1, cm, roc_auc, aucpr, y_prob, fpr, tpr, roc_thresholds, precision_curve, recall_curve):
    # Print metrics
    
    results = prettytable.PrettyTable(title=title)
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
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


    # Visualize the AUCPR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show(block =True)

    # Visualize the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, marker='.', label=f'ROC curve (area = {aucpr:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    

    
    # viz = dtreeviz(model,
    #             X_train,
    #             y_train,
    #             target_name="target",
    #             feature_names=feature_names,
    #             class_names=class_names)

    # viz.view()


def class_imbalance(data, X_train, y_train, seed=42):
    # Class Imbalance in the original data
    class_imbalance = data['is_fraud'].value_counts(normalize=True)
    
    # Applying SMOTE
    smote = SMOTE(random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Before SMOTE
    plt.figure(figsize=(10, 8))
    sns.countplot(x=data['is_fraud'])
    plt.title('Class Imbalance Before SMOTE')
    plt.show()
    
    # After SMOTE
    plt.figure(figsize=(10, 8))
    sns.countplot(x=y_train_res)
    plt.title('Class Imbalance After SMOTE')
    plt.show()
    
    return X_train_res, y_train_res

def predict(model, data, output_path):
    # Predictions
    y_pred = model.predict(data)
    y_prob = model.predict_proba(data)[:, 1]
    
    data['is_fraud'] = y_pred
    data['probability'] = y_prob
    
    # Ensure the directory exists
    directory_path = os.path.dirname(output_path) 
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    
    # Save predictions 
    predictions = pd.DataFrame({'is_fraud': y_pred, 'probability': y_prob})
    
    predictions.to_csv(output_path, index=False)  

    
def main():

    # Default paths
    default_data_path = "Data/Sampled Dataset.csv"
    default_output_path = "Data/Predictions/XGBoost"

    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument('--data_path', type=str, default='Data/Sampled Dataset.csv',
                      help='Path to the Raw data')
    parser.add_argument('--file_type', type=str, default='csv', 
                      help='Type of the file to be read. Options: csv, parquet, xls, etc. Default: csv')
    parser.add_argument('--output_path', type=str, default='Data/Predictions/XGBoost/predictions.csv', 
                      help='Path to output predictions. Default: Data/Predictions/XGBoost/predictions.csv')
    
    args = parser.parse_args()
    
    data_path = os.path.join(parent_dir, args.data_path) if args.data_path == default_data_path else args.data_path
    output_path = os.path.join(parent_dir, args.output_path) if args.output_path == default_output_path else args.output_path
        
    # Read data and display 
    # data = read_data(data_path=data_path, file_type=args.file_type)

           
    data = pl.read_csv('/home/ec2-user/Capstone/Data/Sampled Dataset.csv')

    # Preprocess the data
    data_processed = preprocess_data(data)

    # Preprocess for XGBoost
    data_processed = preprocess_xgboost(data_processed)

    data_processed = data_processed.to_pandas()
    
    #data_processed = data_processed.sample(frac=0.70).reset_index(drop=True)
            
    X = data_processed.drop(columns=['is_fraud'])
    y = data_processed['is_fraud']


    # Split the data
    X_train, X_test, y_train, y_test = split_data(X = X, y = y,data= data_processed, test_size=0.2)

    # Display dataset stats
    dataset_stats(data_processed, X_train, X_test, y_train, y_test)
    
    # Class Imbalance
    X_train_res, y_train_res = class_imbalance(data=data_processed, X_train=X_train, y_train= y_train)
    
    # Hyperparameter Tuning
    best_hyperparams = Hyperparameter_tuning(X_train_res, X_test, y_train_res, y_test)
    
    # Train with Base Hyperparameters
    base_model, base_accuracy, base_precision, base_recall, base_f1, base_cm, base_roc_auc, base_aucpr, base_y_prob, base_fpr, base_tpr, base_roc_thresholds, base_precision_curve, base_recall_curve = train_xgboost(X_train, X_test, y_train, y_test)
    display_metrics("Base Model without SMOTE", base_model, base_accuracy, base_precision, base_recall, base_f1, base_cm, base_roc_auc, base_aucpr, base_y_prob, base_fpr, base_tpr, base_roc_thresholds, base_precision_curve, base_recall_curve)

    # Train with Base Hyperparameters
    base_model, base_accuracy, base_precision, base_recall, base_f1, base_cm, base_roc_auc, base_aucpr, base_y_prob, base_fpr, base_tpr, base_roc_thresholds, base_precision_curve, base_recall_curve = train_xgboost(X_train_res, X_test, y_train_res, y_test)
    display_metrics("Base Model with SMOTE", base_model, base_accuracy, base_precision, base_recall, base_f1, base_cm, base_roc_auc, base_aucpr, base_y_prob, base_fpr, base_tpr, base_roc_thresholds, base_precision_curve, base_recall_curve)
                                   
    # Train with Tuned Hyperparameters
    tuned_model, tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_cm, tuned_roc_auc, tuned_aucpr, tuned_y_prob, tuned_fpr, tuned_tpr, tuned_roc_thresholds, tuned_precision_curve, tuned_recall_curve = train_xgboost(X_train, X_test, y_train, y_test, hyperparameters=best_hyperparams)
    display_metrics("Tuned Model without SMOTE", tuned_model, tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_cm, tuned_roc_auc, tuned_aucpr, tuned_y_prob, tuned_fpr, tuned_tpr, tuned_roc_thresholds, tuned_precision_curve, tuned_recall_curve)

    # Train with Tuned Hyperparameters
    tuned_model, tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_cm, tuned_roc_auc, tuned_aucpr, tuned_y_prob, tuned_fpr, tuned_tpr, tuned_roc_thresholds, tuned_precision_curve, tuned_recall_curve = train_xgboost(X_train_res, X_test, y_train_res, y_test, hyperparameters=best_hyperparams)
    display_metrics("Tuned Model with SMOTE", tuned_model, tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_cm, tuned_roc_auc, tuned_aucpr, tuned_y_prob, tuned_fpr, tuned_tpr, tuned_roc_thresholds, tuned_precision_curve, tuned_recall_curve)

    # Predictions
    predict(tuned_model, X_test, output_path)

if __name__ == '__main__':
    main()