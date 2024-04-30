# Capstone

# Leveraging Graph-Based Learning for Credit Card Fraud Detection:  A comparative study of traditional, deep learning and graph-based approaches.

Anomaly detection in financial transactions is crucial for preventing fraud, money laundering, and other illicit activities. Credit card fraud presents a substantial challenge, resulting in significant financial losses for businesses and individuals annually. Our research aims to explore the potential of Graph Neural Networks (GNNs) in enhancing performance in fraud detection by leveraging their capacity to analyze transaction patterns. We conduct a comparative study of Graph Neural Networks (GNNs), traditional machine learning models, and deep learning models for detecting fraud in credit card transactions. 

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Anomaly detection in financial transactions is crucial for preventing fraud, money laundering, and other illicit activities. This project explores various approaches to tackle this problem, including Graph Neural Networks (GNNs), traditional machine learning models such as XGBoost and Random Forest, and deep learning models like Convolutional Neural Networks (CNNs) and Long Short Term Memory (LSTM).

## Dataset

The dataset used in this project is a synthetic credit card fraud detection dataset provided by IBM. The dataset contains around 24 million transaction records with 12 fields. Among these transactions, 0.1% of the transactions account to fradualent transactions.

Link to the github folder of the dataset: https://github.com/IBM/TabFormer/tree/main

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

2. Train Random Forest model:
   ```
   python train_random_forest.py --path-to-dir /path/to/dataset --verbose --n-estimators 100 --max-depth 5
   ```

3. Train GNN model:
   ```
   python train_gnn.py --path-to-dir /path/to/dataset --verbose --hidden-dim 64 --n-layers 3 --dropout 0.5
   ```

4. Train CNN model:
   ```
   python train_cnn.py --path-to-dir /path/to/dataset --verbose --n-filters 64 --kernel-size 3 --pool-size 2
   ```

5. Train RNN model:
   ```
   python train_rnn.py --path-to-dir /path/to/dataset --verbose --hidden-dim 128 --n-layers 2 --dropout 0.3
   ```

Replace `/path/to/dataset` with the actual path to your dataset directory. You can also adjust the hyperparameters and options according to your requirements.

## Results

The evaluation results and comparisons of different models will be provided in the `results` directory. The results will include metrics such as accuracy, precision, recall, F1-score, and ROC-AUC score. Additionally, visualizations and analysis of the results will be presented to gain insights into the performance of each model.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the existing code style and guidelines.

## License

This project is licensed under the [MIT License](LICENSE).
