# Leveraging Graph-Based Learning for Credit Card Fraud Detection:  A comparative study of traditional, deep learning and graph-based approaches.

## Abstract

Anomaly detection in financial transactions is crucial for preventing fraud, money laundering, and other illicit activities. Credit card fraud presents a substantial challenge, resulting in significant financial losses for businesses and individuals annually. Our research aims to explore the potential of Graph Neural Networks (GNNs) in enhancing performance in fraud detection by leveraging their capacity to analyze transaction patterns. We conduct a comparative study of Graph Neural Networks (GNNs), traditional machine learning models, and deep learning models for detecting fraud in credit card transactions. 

- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Files Overview](#files%20overview)


## Dataset

The dataset used in this project is a synthetic credit card fraud detection dataset provided by IBM. The dataset contains around 24 million transaction records with 12 fields. Among these transactions, 0.1% of the transactions account to fraudulent transactions.

Link to the GitHub folder of the dataset: https://github.com/IBM/TabFormer/tree/main

The dataset is contained in the  ./data/credit_card folder of the GitHub. To extract the .csv file from transaction.tgz, run the following command:

```
  tar -xvf <path/to/compressed dataset>
```

## Models

The following models are implemented and evaluated in this project:

1. Graph Neural Networks - Relational Graph Convolution Network
2. Classical Models - Logistic Regression, Random Forest, LightGBM, CatBoost, XGBoost
3. Deep Learning Models - Convolutional Neural Networks (CNNs), Long Short Term Memory (LSTM), Hybrid architecture using CNN and LSTM

Each model is trained and evaluated using appropriate techniques and metrics specific to fraud detection tasks.


## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/ChiragLakhanpal/Capstone.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The project provides scripts for training and evaluating different models. Here are some examples of how to use the scripts:

1. Run preprocess.py:
  ```
  python preprocess.py --data_path /path/to/dataset --test_size 0.2 --no-val_data --val_size 0.2 --detect_binary --no-numeric_dtype --one_hot --na_cleaner_mode "remove row" --no- 
  normalize --no-balance --sample --sample_size 0.2 --stratify_column '<target variable>' --datetime_columns Time --clean_columns Amount --consider_as_categorical '<columns separated by columns>'
  --target 'target variable' --verbose
  ```
2. Train XGBoost model:
   ```
   python train_xgboost.py --path-to-dir /path/to/dataset --verbose --learning-rate 0.1 --n-estimators 100
   ```

3. Train GNN model:
   ```
   python GNN_Train.py --path-to-dir /path/to/dataset --verbose --epochs 100 --hidden-dim 64 --n-layers 3 
   ```

For thorough documentation, please visit [individual directories](#https://github.com/ChiragLakhanpal/Capstone/tree/main/code/main_code) to train and infer the models.




Replace `/path/to/dataset` with the actual path to your dataset directory. You can also adjust the hyperparameters and options according to your requirements.


## Files Overview

The code folder contains two subfolders component and main_code:

The component code contains 3 folders for utils named classical_machine_learning, deep_learning, gnn and a precprocess file.

- classical_machine_learning:
  - utils.py : This file contains all the utils necessary for classical models.
- deep_learning:
  - dataloader.py : Contains the code for creating a custom dataset and dataloader.
  - models.py : Contains the model classes for deep learning.
  - utils.py : Contains utils such as argparse function, metrics, training and predict class etc.
- gnn:
  - model.py : Contains the model class for RGCN
  - utils.py : Contains utils such as argparse function, metrics, training and predict class etc.
- preprocess.py : Contains a standardized preprocessing code to preprocess any dataset.

The main_code contains 3 folders named classical_machine_learning, deep_learning, GNN.

- classical_machine_learning:
  - models : Contains all the model pickle files saved after training.
  - CatBoost_train.py : Contains the code to train and test the CatBoost Model.
  - LightGBM_train.py : Contains the code to train and test the LightGBM Model.
  - Logistic_train.py : Contains the code to train and test the Logistic Regression Model.
  - Random_Forest_train.py : Contains the code to train and test the Random Forest Model.
  - Xgboost_train.py : Contains the code to train and test the XGBoost Model.
  - Inference.py : Run all models to get the evaluation metrics and plots.
 
- deep_learning:
  - models : Contains the .pt file of models saved after training.
  - plots : Contains the saved plots.
  - CNN_Train.py : Contains the code to train the CNN model.
  - CNN_Test.py : Contains the code to test the CNN model.
  - LSTM_Train.py : Contains the code to train the LSTM model.
  - LSTM_Test.py : Contains the code to test the LSTM model.
  - CNN-LSTM_Train.py : Contains the code to train the CNN-LSTM model.
  - CNN-LSTM_Test.py : Contains the code to test the CNN-LSTM model.
  - Inference.py : Run all models to get the evaluation metrics and plots.

- GNN:
  - models : Contains the .pt file of models saved after training.
  - plots : Contains the saved plots.
  - GNN_Train.py : Contains the code to train the GNN model.
  - GNN_Test.py : Contains the code to test the GNN model.
  - Inference.py : Run all models to get the evaluation metrics and plots.
