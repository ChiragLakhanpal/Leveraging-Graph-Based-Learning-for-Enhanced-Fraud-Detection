## Directory overview

This directory contains the code for the Capstone project, which includes various machine learning and deep learning models for training and inference.

The directory is structured as follows:
```
.
├── code
│   ├── component
│   │   ├── classical_machine_learning
│   │   │   └── utils.py
│   │   ├── deep_learning
│   │   │   ├── dataloader.py
│   │   │   ├── models.py
│   │   │   └── utils.py
│   │   ├── gnn
│   │   │   ├── model.py
│   │   │   └── utils.py
│   │   └── preprocess.py
│   └── main_code
│       ├── classical_machine_learning
│       │   ├── models
│       │   ├── CatBoost_train.py
│       │   ├── Inference.py
│       │   ├── LightGBM_train.py
│       │   ├── Logistic_train.py
│       │   ├── Random_Forest_train.py
│       │   ├── SVM_train.py
│       │   └── Xgboost_train.py
│       ├── deep_learning
│       │   ├── models
│       │   ├── plots
│       │   ├── CNN_LSTM_Test.py
│       │   ├── CNN_LSTM_Train.py
│       │   ├── CNN_Test.py
│       │   ├── CNN_Train.py
│       │   ├── Inference.py
│       │   ├── LSTM_Test.py
│       │   └── LSTM_Train.py
│       └── gnn
│           ├── models
│           ├── plots
│           ├── GNN_Train.py
│           └── Inference.py
```

The code directory is organized into the following subdirectories:

- `component`: Contains utility files and helper functions for different model types.
  - `classical_machine_learning`: Utility functions for classical machine learning models.
    - `utils.py`: Utility functions for classical machine learning.
  - `deep_learning`: Utility files for deep learning models.
    - `dataloader.py`: Data loading and preprocessing functions for deep learning models.
    - `models.py`: Deep learning model architectures.
    - `utils.py`: Utility functions for deep learning.
  - `gnn`: Utility files for Graph Neural Network (GNN) models.
    - `model.py`: GNN model architecture.
    - `utils.py`: Utility functions for GNN.
  - `preprocess.py`: Data preprocessing functions.

- `main_code`: Contains the main training and inference scripts for different models.
  - `classical_machine_learning`: Classical machine learning models.
    - `models`: Trained classical machine learning models.
    - `CatBoost_train.py`: Training script for CatBoost model.
    - `Inference.py`: Inference script for classical machine learning models.
    - `LightGBM_train.py`: Training script for LightGBM model.
    - `Logistic_train.py`: Training script for Logistic Regression model.
    - `Random_Forest_train.py`: Training script for Random Forest model.
    - `SVM_train.py`: Training script for Support Vector Machine (SVM) model.
    - `Xgboost_train.py`: Training script for XGBoost model.
  - `deep_learning`: Deep learning models.
    - `models`: Trained deep learning models.
    - `plots`: Plots generated during training and evaluation.
    - `CNN_LSTM_Test.py`: Testing script for CNN-LSTM model.
    - `CNN_LSTM_Train.py`: Training script for CNN-LSTM model.
    - `CNN_Test.py`: Testing script for CNN model.
    - `CNN_Train.py`: Training script for CNN model.
    - `Inference.py`: Inference script for deep learning models.
    - `LSTM_Test.py`: Testing script for LSTM model.
    - `LSTM_Train.py`: Training script for LSTM model.
  - `gnn`: Graph Neural Network (GNN) models.
    - `models`: Trained GNN models.
    - `plots`: Plots generated during training and evaluation.
    - `GNN_Train.py`: Training script for GNN model.
    - `Inference.py`: Inference script for GNN model.


## Usage

For detailed usage information and instructions for each script, please refer to the individual files in the respective directories.

The `main_code` directory contains subdirectories for different model types, including classical machine learning, deep learning, and graph neural networks (GNN). Each subdirectory contains scripts for training and inference.

The `component` directory provides utility functions and helper classes that can be used across different models for tasks such as data preprocessing, loading, and other common operations.

When running the scripts, make sure to provide the required input data and configure the appropriate model parameters as specified in each script's documentation.

## Requirements

The code in this directory requires Python 3.10 and the following libraries to name a few:
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- PyTorch
- dgl (for GNN)
- Matplotlib (for plotting)

Please refer to the individual script files for specific dependencies and requirements.
