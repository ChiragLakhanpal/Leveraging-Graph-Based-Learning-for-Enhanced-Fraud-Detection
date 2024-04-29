# train_models.py

import utils
from preprocess import preprocess_data, split_data, read_data,clean_col_name
import os
import numpy as np
import argparse
import prettytable
from sklearn.svm import SVC
import argparse
import pandas as pd
from utils import evaluate_model, display_results, dataset_stats, generate_aucpr_plot
import pickle

seed = 42

np.random.seed(seed)  
parent_dir = os.getcwd()

def train_model(X_train, y_train, params={}):
    '''
    Train an SVM Classifier model
    :param X_train: Training features
    :param y_train: Training target
    :param params: SVM Classifier parameters
    :returns Trained model
    '''
    model = SVC(**params)
    model.fit(X_train, y_train)
    return model

def save_predictions(model, X_test, y_test, output_path):
    '''
    Generate predictions using the trained model and save them to a file along with the original dataframe
    
    :param model: Trained model
    :param X_test: Test features
    :param y_test: Test target
    :param output_path: Path to save the predictions
    
    :returns None
    '''
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_prob > 0.5).astype(int)
        predictions_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        predictions_df['Prediction'] = y_pred
        predictions_df['Probability'] = y_prob
        
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        predictions_df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")
    
    except Exception as e:
        print(f"Error generating predictions: {e}")
                
def save_model(model, model_path):
    '''
    Save the trained model to a file
    
    :param model: Trained model
    :param model_path: Path to save the model
    
    :returns None
    '''
    try:
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
            
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        print(f"Model saved to {model_path}")
    
    except Exception as e:
        print(f"Error saving the model: {e}")
        
def main():
    
    default_data_path = "Data/Raw Data/data.csv"
    default_output_path = "Data/Predictions/SVM/predictions_SVM.csv"
    default_model_path = "Models/SVM/SVM_model.pkl"
        
    parser = argparse.ArgumentParser(description='Train an k-Nearest Neighbors model for binary classification')
    parser.add_argument('--data_path', type=str, default=default_data_path, 
                      help='Path to the Raw data. Default: Data/Raw Data/data.csv')
    parser.add_argument('--file_type', type=str, default='csv', 
                      help='Type of the file to be read. Options: csv, parquet, xls, etc. Default: csv')
    parser.add_argument('--output_path', type=str, default=default_output_path,
                        help='Path to save the predictions. Default: Data/Predictions/SVM/predictions_SVM.csv')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help='Path to save the trained model. Default: Models/SVM/SVM_model.pkl')
                
    parser.add_argument('--target', type=str, required=True, help='Name of the target column')
    parser.add_argument('--preprocess', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Preprocess the data. Default: True')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing. Default: 0.2')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of training data for validation. Default: 0.2')
    parser.add_argument('--train_only', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Train the model only. Default: False')
    parser.add_argument('--sample', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Sample the data. Default: False')
    parser.add_argument('--sample_size', type=float, default=1.0, help='Proportion of data to sample. Default: 0.6')

    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter. Default: 1.0')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type. Options: linear, poly, rbf, sigmoid. Default: rbf')
    parser.add_argument('--degree', type=int, default=3, help='Degree of the polynomial kernel function. Default: 3')
    parser.add_argument('--gamma', type=str, default='scale', help='Kernel coefficient. Options: scale, auto. Default: scale')
    parser.add_argument('--probability', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Whether to enable probability estimates. Default: True')    
    
    args = parser.parse_args()

    data_path = os.path.join(parent_dir, args.data_path) if args.data_path == default_data_path else args.data_path
    output_path = os.path.join(parent_dir, args.output_path) if args.output_path == default_output_path else args.output_path
    model_path = os.path.join(parent_dir, args.model_path) if args.model_path == default_model_path else args.model_path
    
    target = clean_col_name(args.target)
    sample_size = args.sample_size
            
    data = read_data(os.path.join(parent_dir, data_path), args.file_type)

    if args.preprocess:
        data = preprocess_data (dataframe = data, 
                detect_binary=True, 
                numeric_dtype=True, 
                one_hot=True, 
                na_cleaner_mode="mode", 
                normalize=False,
                balance=False, 
                sample= True,
                sample_size = sample_size,
                stratify_column ='Is Fraud?',
                datetime_columns=['Time'],
                clean_columns=['Amount'], 
                remove_columns=['Card', 'User', 'Merchant Name'], 
                consider_as_categorical=['Use Chip','Merchant City','Merchant State','Zip','MCC','Errors?'],
                target='Is Fraud?',
                verbose=True)

    params = {
        'C': args.C,
        'kernel': args.kernel,
        'degree': args.degree,
        'gamma': args.gamma,
        'probability': args.probability
    }
                
    X = data.drop(columns=[target])
    y = data[target]

    if not args.train_only:
        X_train, _ , X_test, y_train, _ ,y_test = split_data(X = X, y = y, data= data, test_size=args.test_size, val_data=True)
        dataset_stats(data=data, target=target, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    else:
        X_train, y_train = X, y
        dataset_stats(data=data, target=target, X_train=X_train, y_train=y_train)

    model = train_model(X_train, y_train, params)    
    save_model(model, model_path) 
    
    if not args.train_only:
        evaluate_model(model, X_test, y_test)
        display_results(model, X_test, y_test, title='SVM Results')
        generate_aucpr_plot(y_test=y_test, y_prob=model.predict_proba(X_test)[:, 1])
        save_predictions(model, X_test, y_test, output_path)

if __name__ == '__main__':
    main()
    
