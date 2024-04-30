# Leveraging Graph-Based Learning for Credit Card Fraud Detection:  A comparative study of traditional, deep learning and graph-based approaches.

## Abstract

Anomaly detection in financial transactions is crucial for preventing fraud, money laundering, and other illicit activities. Credit card fraud presents a substantial challenge, resulting in significant financial losses for businesses and individuals annually. Our research aims to explore the potential of Graph Neural Networks (GNNs) in enhancing performance in fraud detection by leveraging their capacity to analyze transaction patterns. We conduct a comparative study of Graph Neural Networks (GNNs), traditional machine learning models, and deep learning models for detecting fraud in credit card transactions. 

- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Files Overview](#filesoverview)


## Dataset

The dataset used in this project is a synthetic credit card fraud detection dataset provided by IBM. The dataset contains around 24 million transaction records with 12 fields. Among these transactions, 0.1% of the transactions account to fradualent transactions.

Link to the github folder of the dataset: https://github.com/IBM/TabFormer/tree/main

The dataset is contained in the  ./data/credit_card folder of the github. To extract the .csv file from transaction.tgz, run the following command:

```
   python train_xgboost.py --path-to-dir /path/to/dataset --verbose --learning-rate 0.1 --n-estimators 100
```

## Models

The following models are implemented and evaluated in this project:

1. Graph Neural Netorks - Relational Graph Convolution Network
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

1. Train XGBoost model:
   ```
   python train_xgboost.py --path-to-dir /path/to/dataset --verbose --learning-rate 0.1 --n-estimators 100
   ```

2. Train GNN model:
   ```
   python GNN_Train.py --path-to-dir /path/to/dataset --verbose --epochs 100 --hidden-dim 64 --n-layers 3 
   ```

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
