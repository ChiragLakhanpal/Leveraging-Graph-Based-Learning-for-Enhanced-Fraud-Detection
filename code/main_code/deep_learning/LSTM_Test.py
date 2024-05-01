import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import prettytable

import sys
sys.path.append('../../component')

from preprocess import read_data
from deep_learning.dataloader import *
from deep_learning.utils import *
from deep_learning.models import *

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
data_path = os.path.join(parent_dir, 'Data/test_data.csv')

# Setting the device to GPU if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the model path

m_name = "LSTM"
model_path = "models/"+f'model_{m_name}.pt'

def lstm_main():
    """
    Main function of the program. Reads test data from a CSV file, prepares the data loader,
    initializes a LSTM model, and makes predictions using the provided test data.

    :params None

    Example:
        To run the program:
        $ python your_script.py
    """
    
    args = arg_parser()

    # Read the data and get the test split

    data = read_data(data_path=data_path)

    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    # Preparing test data using dataloader

    data_loader = CustomDataLoader(batch_size=args.batch_size, device=device)
    test_loader,_ = data_loader.prepare_test_loader(X_test, y_test)
    _, shape = data_loader.prepare_test_loader(X_test, y_test)

    model = LSTM(args.output_dim, args.lstm_input_size, args.hidden_size, args.lstm_n_layers, shape).to(device)

    # Calling the trainer object to test data

    trainer = DL_Trainer(model)
    accuracy, precision, recall, f1, roc_auc, auc_pr, targets, outputs, probs = trainer.predict(test_loader, model_path,m_name)

    return targets, outputs, probs

if __name__ == "__main__":
    targets, outputs, probs = lstm_main()
