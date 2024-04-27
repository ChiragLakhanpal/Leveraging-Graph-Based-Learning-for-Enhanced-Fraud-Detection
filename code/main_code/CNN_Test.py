import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import prettytable

import sys
sys.path.append('../component')

from deep_learning.dataloader import *
from deep_learning.utils import *
from deep_learning.models import CNN
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

# Defining the model path

m_name = "CNN"
model_path = f'model_{m_name}.pt'

def main():
    """
        Main function of the program. Reads test data from a CSV file, prepares the data loader,
        initializes a Convolutional Neural Network (CNN) model, and makes predictions using the provided test data.

        :params None

        Example:
            To run the program:
            $ python your_script.py
        """
    args = arg_parser()

    data = pd.read_csv('../component/test_data.csv')

    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    data_loader = CustomDataLoader(batch_size=args.batch_size, device=device)
    test_loader = data_loader.prepare_test_loader(X_test, y_test)

    model = CNN(args.cnn_output_dim).to(device)

    trainer = DL_Trainer(model)
    trainer.predict(test_loader, model_path,m_name)

if __name__ == "__main__":
    main()
