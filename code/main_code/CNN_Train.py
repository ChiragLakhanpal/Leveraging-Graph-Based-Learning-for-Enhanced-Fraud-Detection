import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append('../component')

from preprocess import *
from deep_learning.dataloader import *
from deep_learning.models import *
from deep_learning.utils import *


import warnings
warnings.filterwarnings('ignore')

# Set random seed

seed = 42
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# System path
parent_dir = os.getcwd()
data_path = os.path.join(parent_dir, '../../Data/card_transaction.v1.csv')
save_data_path = os.path.join(parent_dir, '../../Data')

# Setting the device to GPU if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m_name = "CNN"

def main():
    """
        Main function of the program. Reads data and runs the preprocess function, prepares the data loader,
        initializes a CNN model, and trains and validates the model.

        :params None

        Example:
            To run the program:
            $ python your_script.py
        """
    
    args = arg_parser()

    # Read the data and run the preprocess function

    data = read_data(data_path=data_path)

    data = preprocess_data(dataframe=data,
                           detect_binary=True,
                           numeric_dtype=False,
                           one_hot=True,
                           na_cleaner_mode="mode",
                           normalize=False,
                           balance=False,
                           sample=True,
                           sample_size=0.6,
                           stratify_column='Is Fraud?',
                           datetime_columns=['Time'],
                           clean_columns=['Amount'],
                           remove_columns=[],
                           consider_as_categorical=['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC',
                                                    'Errors?'],
                           target='Is Fraud?',
                           verbose=True)

    X = data.drop(args.target, axis=1)
    y = data[args.target]

    # Creating a train test split

    X_train, X_val, X_test, y_train,y_val, y_test = split_data(X, y, data, val_size=0.2,test_size=0.2, val_data=True)

    # Save test data to csv
    
    save_test_data(X_test, y_test,save_data_path)

    # Printing dataset stats

    dataset_stats(data, X_train, X_val,X_test, y_train,y_val,y_test)

    # Preparing train and validation data using dataloader

    data_loader = CustomDataLoader(batch_size=args.batch_size, device=device)
    train_loader, val_loader, _ = data_loader.prepare_train_val_loader(X_train, y_train, X_val, y_val)


    model = CNN(args.cnn_input_size,args.cnn_hidden_size,args.hidden_size,args.cnn_out_size,args.output_dim,args.kernel_size,args.kernel_size_pool).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Calling the trainer object to train and validate data

    trainer = DL_Trainer(model)
    trainer.train_val(train_loader, val_loader,args.epochs,optimizer,criterion,args.best_val_f1,m_name)

if __name__ == "__main__":
    main()
