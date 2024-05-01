# inference.py
import sys
sys.path.append('../../component')

import os
import argparse
import pickle
import prettytable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from preprocess import read_data,clean_col_name
import sys

seed = 42

np.random.seed(seed)
parent_dir = os.getcwd()

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except ModuleNotFoundError as e:
        print(f"Skipping model {model_path} due to missing module: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, metrics):
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results = {}
        for metric in metrics:
            if metric == 'acc':
                results['Accuracy'] = accuracy_score(y_test, y_pred)
            elif metric == 'precision':
                results['Precision'] = precision_score(y_test, y_pred)
            elif metric == 'recall':
                results['Recall'] = recall_score(y_test, y_pred)
            elif metric == 'f1':
                results['F1 Score'] = f1_score(y_test, y_pred)
            elif metric == 'roc_auc':
                results['ROC AUC'] = roc_auc_score(y_test, y_prob)
            elif metric == 'pr_auc':
                results['PR AUC'] = average_precision_score(y_test, y_prob)
        return results

    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

    
def display_results(results_dict):
    table = prettytable.PrettyTable(['Model'] + list(next(iter(results_dict.values())).keys()))
    for model, results in results_dict.items():
        row = [model] + list(results.values())
        table.add_row(row)
    print(table)

def generate_aucpr_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AUC={average_precision_score(y_test, y_prob):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def generate_aucroc_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_test, y_prob):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inference script for multiple models')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to the trained models. Only pickle (pkl) and pytorch (pt) files are supported.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data')
    parser.add_argument('--metrics', nargs='+', default=['acc', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'],
                        help='Metrics to evaluate (default: acc, precision, recall, f1, roc_auc, pr_auc)')
    parser.add_argument('--generate_aucpr', default= True,action='store_true', help='Generate AUCPR plot')
    parser.add_argument('--generate_aucroc',default= True, action='store_true', help='Generate AUCROC plot')
    parser.add_argument('--target', type=str, required=True, help='Name of the target column')
    args = parser.parse_args()


    target = clean_col_name(args.target)

    test_data = pd.read_csv(args.test_data)
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    results_dict = {}
    y_prob_dict = {}
    for model_path in args.models:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model = load_model(model_path)
        if model is None:
            continue
        print(f"Evaluating model: {model_name}")
        results = evaluate_model(model, X_test, y_test, args.metrics)
        if results is None:
            continue
        results_dict[model_name] = results

        y_prob = model.predict_proba(X_test)[:, 1]
        y_prob_dict[model_name] = y_prob

    display_results(results_dict)

    if args.generate_aucpr:
        generate_aucpr_plot(y_test, y_prob_dict)
    if args.generate_aucroc:
        generate_aucroc_plot(y_test, y_prob_dict)

if __name__ == '__main__':
    main()