# Capstone

# Anomaly Detection in Financial Transactions: GNN vs Traditional Models vs Deep Learning Models

This project aims to conduct a comparative study of Graph Neural Networks (GNNs), traditional machine learning models, and deep learning models for detecting anomalies in financial transactions. The goal is to evaluate the performance of these approaches and provide insights into their effectiveness in identifying fraudulent or suspicious activities.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Anomaly detection in financial transactions is crucial for preventing fraud, money laundering, and other illicit activities. This project explores various approaches to tackle this problem, including Graph Neural Networks (GNNs), traditional machine learning models such as XGBoost and Random Forest, and deep learning models like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

## Dataset

The dataset used in this project consists of labeled financial transaction records, where each transaction is characterized by a set of features and a corresponding label indicating whether it is anomalous or not. The dataset is not included in this repository due to privacy and confidentiality reasons. However, you can replace the dataset with your own or use a publicly available dataset with similar characteristics.

## Models

The following models are implemented and evaluated in this project:

1. Graph Neural Networks (GNNs)
2. XGBoost
3. Random Forest
4. Convolutional Neural Networks (CNNs)
5. Recurrent Neural Networks (RNNs)

Each model is trained and evaluated using appropriate techniques and metrics specific to anomaly detection tasks.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/anomaly-detection-financial-transactions.git
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
