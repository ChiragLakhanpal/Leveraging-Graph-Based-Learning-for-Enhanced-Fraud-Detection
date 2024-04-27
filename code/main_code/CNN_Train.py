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

from preprocess_auto import *
from deep_learning.dataloader import *
from deep_learning.models import *
from deep_learning.utils import *
from utils import dataset_stats

import warnings
warnings.filterwarnings('ignore')

# Set random seed

seed = 42
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Setting the device to GPU if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m_name = "CNN"

def main():
    
    args = arg_parser()

    data = read_data('../../Data/card_transaction.v1.csv')

    data = preprocess_data(dataframe = data,
    detect_binary = True,
    numeric_dtype = False,
    one_hot = True,
    na_cleaner_mode = "mode",
    normalize = False,
    balance = False,
    datetime_columns = ['Time'],
    clean_cols = ['Amount'],
    remove_columns = [],
    consider_as_categorical = ['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?'],
                               target = 'Is Fraud?',
    verbose = True)

    X = data.drop(args.target, axis=1)
    y = data[args.target]

    X_train, X_val, X_test, y_train,y_val, y_test = split_data(X, y, data, val_size=0.2,test_size=0.2, val_data=True)
    save_test_data(X_test, y_test)

    dataset_stats(data, X_train, X_val,X_test, y_train,y_val,y_test)

    data_loader = CustomDataLoader(batch_size= args.batch_size,device=device)
    train_loader, val_loader = data_loader.prepare_train_val_loader(X_train, y_train, X_val, y_val)

    model = CNN(args.cnn_output_dim).to(device)
    criterion, optimizer = model_parameters()

    trainer = DL_Trainer(model)
    trainer.train_val(train_loader, val_loader,args.epochs,optimizer,criterion,args.best_val_f1,m_name)

if __name__ == "__main__":
    main()
