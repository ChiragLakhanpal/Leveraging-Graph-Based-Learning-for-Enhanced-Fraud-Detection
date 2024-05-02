
# Relational Graph Convolutional Network (RGCN) Models

The `gnn` folder contains the implementation of Relational Graph Convolutional Network (RGCN) models for the comparative study. RGCN is a variant of Graph Neural Networks (GNNs) that can handle heterogeneous graph-structured data with multiple edge types.

## Models

The `models` folder contains the implementation of the RGCN models used in the study. RGCN models are designed to operate on heterogeneous graphs and learn node representations by considering the different types of relationships between nodes.

## Plots

The `plots` folder stores all the generated plots related to the RGCN models, including evaluation metrics and visualizations. These plots are used to analyze and compare the performance of the RGCN models.

## Scripts

### `GNN_Train.py`

This script is used to train the RGCN model. It loads the training data, defines the RGCN architecture, and trains the model using the specified hyperparameters. The trained model is saved for future evaluation and inference.

Usage: `python GNN_Train.py`

### `Inference.py`

This script is used to perform inference using the trained RGCN model. It loads the trained model, applies it to the test data, and generates predictions. It also calculates evaluation metrics to assess the performance of the RGCN model.

Usage: `python Inference.py`

## Usage

To train the RGCN model, navigate to the `gnn` folder and run the following command:

```
python GNN_Train.py
```

This will load the training data, train the RGCN model, and save the trained model for future use.

To perform inference using the trained RGCN model, run the following command:

```
python Inference.py
```

This script will load the trained RGCN model, apply it to the test data, and generate predictions. It will also calculate evaluation metrics to assess the performance of the model.

Make sure to have the required dependencies installed and the dataset properly prepared before running the scripts.

Note: The RGCN model training and inference scripts are independent and can be run separately. There is no specific order requirement between the scripts in the `gnn` folder.

# Classical Machine Learning Models

The `classical_machine_learning` folder contains the implementation of various classical machine learning models for the comparative study.

## Models

The `models` folder contains the trained models for each classical machine learning algorithm used in the study. These models are saved in pickle (pkl) format and can be loaded for inference or further analysis.

## Scripts

### `CatBoost_train.py`

This script is used to train a CatBoost model for binary classification. It loads the training data, preprocesses it if specified, and trains the CatBoost model using the provided hyperparameters. The trained model is saved for future evaluation and inference.

Usage:
```
python CatBoost_train.py --data_path DATA_PATH --file_type FILE_TYPE --model_path MODEL_PATH --output_path OUTPUT_PATH --target TARGET [--preprocess] [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--train_only] [--sample] [--sample_size SAMPLE_SIZE] --iterations ITERATIONS --learning_rate LEARNING_RATE --depth DEPTH --l2_leaf_reg L2_LEAF_REG
```

### `LightGBM_train.py`

This script is used to train a LightGBM model for binary classification. It loads the training data, preprocesses it if specified, and trains the LightGBM model using the provided hyperparameters. The trained model is saved for future evaluation and inference.

Usage:
```
python LightGBM_train.py --data_path DATA_PATH --file_type FILE_TYPE --output_path OUTPUT_PATH --model_path MODEL_PATH --target TARGET [--preprocess] [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--train_only] [--sample] [--sample_size SAMPLE_SIZE] --n_estimators N_ESTIMATORS --learning_rate LEARNING_RATE --max_depth MAX_DEPTH --num_leaves NUM_LEAVES --subsample SUBSAMPLE --colsample_bytree COLSAMPLE_BYTREE --objective OBJECTIVE --n_jobs N_JOBS
```

### `Logistic_train.py`

This script is used to train a Logistic Regression model for binary classification. It loads the training data, preprocesses it if specified, and trains the Logistic Regression model using the provided hyperparameters. The trained model is saved for future evaluation and inference.

Usage:
```
python Logistic_train.py --data_path DATA_PATH --file_type FILE_TYPE --output_path OUTPUT_PATH --model_path MODEL_PATH --target TARGET [--preprocess] [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--train_only] [--sample] [--sample_size SAMPLE_SIZE] --C C --penalty PENALTY --solver SOLVER --max_iter MAX_ITER
```

### `Random_Forest_train.py`

This script is used to train a Random Forest model for binary classification. It loads the training data, preprocesses it if specified, and trains the Random Forest model using the provided hyperparameters. The trained model is saved for future evaluation and inference.

Usage:
```
python Random_Forest_train.py --data_path DATA_PATH --file_type FILE_TYPE --output_path OUTPUT_PATH --model_path MODEL_PATH --target TARGET [--preprocess] [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--train_only] [--sample] [--sample_size SAMPLE_SIZE] --n_estimators N_ESTIMATORS --max_depth MAX_DEPTH --min_samples_split MIN_SAMPLES_SPLIT --min_samples_leaf MIN_SAMPLES_LEAF --n_jobs N_JOBS
```

### `SVM_train.py`

This script is used to train a Support Vector Machine (SVM) model for binary classification. It loads the training data, preprocesses it if specified, and trains the SVM model using the provided hyperparameters. The trained model is saved for future evaluation and inference.

Usage:
```
python SVM_train.py --data_path DATA_PATH --file_type FILE_TYPE --output_path OUTPUT_PATH --model_path MODEL_PATH --target TARGET [--preprocess] [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--train_only] [--sample] [--sample_size SAMPLE_SIZE] --C C --kernel KERNEL --degree DEGREE --gamma GAMMA [--probability]
```

### `Xgboost_train.py`

This script is used to train an XGBoost model for binary classification. It loads the training data, preprocesses it if specified, and trains the XGBoost model using the provided hyperparameters. The trained model is saved for future evaluation and inference.

Usage:
```
python Xgboost_train.py --data_path DATA_PATH --file_type FILE_TYPE --model_path MODEL_PATH --output_path OUTPUT_PATH --target TARGET [--preprocess] [--test_size TEST_SIZE] [--val_size VAL_SIZE] [--train_only] [--sample] [--sample_size SAMPLE_SIZE] --max_depth MAX_DEPTH --learning_rate LEARNING_RATE --n_estimators N_ESTIMATORS --subsample SUBSAMPLE --colsample_bytree COLSAMPLE_BYTREE [--use_gpu]
```

### `Inference.py`

This script is used to perform inference using the trained classical machine learning models. It loads the trained models, applies them to the test data, and generates predictions. It also calculates evaluation metrics to assess the performance of the models and generates AUCPR and AUCROC plots if specified.

Usage:
```
python Inference.py --models MODELS [MODELS ...] --test_data TEST_DATA [--metrics METRICS [METRICS ...]] [--generate_aucpr] [--generate_aucroc] --target TARGET
```

Make sure to provide the necessary arguments and paths when running the scripts. The scripts accept various hyperparameters and options to customize the training and inference process.


# Deep Learning Models

Note: Please ensure that you run the `CNN_Train.py` script first before running any test files (`CNN_Test.py`, `LSTM_Test.py`, `CNN_LSTM_Test.py`). The `CNN_Train.py` script generates the necessary test data that is required by the other test files. Running the test files without executing `CNN_Train.py` first may result in errors or incorrect results.

The `deep_learning` folder contains the implementation of various deep learning models for a comparative study, including CNN (Convolutional Neural Network), LSTM (Long Short-Term Memory), and a hybrid model combining CNN and LSTM.

## Models

The `models` folder contains the implementation of the following deep learning models:

- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- CNN+LSTM (Hybrid model combining CNN and LSTM)

These models are used for the deep learning portion of the comparative study.

## Plots

The `plots` folder stores all the generated plots, including AUCPR (Area Under the Precision-Recall Curve) and AUCROC (Area Under the Receiver Operating Characteristic Curve) plots. These plots are used to evaluate and compare the performance of the deep learning models.

## Scripts

### `CNN_LSTM_Train.py`

This script is used to train the hybrid model that combines CNN and LSTM. It loads the training data, defines the model architecture, and trains the model using the specified hyperparameters.

Usage: `python CNN_LSTM_Train.py`

### `CNN_LSTM_Test.py`

This script is used to evaluate the trained CNN+LSTM hybrid model on the test data. It loads the trained model, makes predictions on the test set, and generates evaluation metrics.

Usage: `python CNN_LSTM_Test.py`

### `CNN_Train.py`

This script is used to train the CNN model. It loads the training data, defines the CNN architecture, and trains the model using the specified hyperparameters.

Usage: `python CNN_Train.py`

### `CNN_Test.py`

This script is used to evaluate the trained CNN model on the test data. It loads the trained model, makes predictions on the test set, and generates evaluation metrics.

Usage: `python CNN_Test.py`

### `LSTM_Train.py`

This script is used to train the LSTM model. It loads the training data, defines the LSTM architecture, and trains the model using the specified hyperparameters.

Usage: `python LSTM_Train.py`

### `LSTM_Test.py`

This script is used to evaluate the trained LSTM model on the test data. It loads the trained model, makes predictions on the test set, and generates evaluation metrics.

Usage: `python LSTM_Test.py`

### `Inference.py`

This script is used to generate a master table that summarizes the results of all the deep learning models (CNN, LSTM, and CNN+LSTM). It loads the trained models, makes predictions on the test set, and calculates evaluation metrics for each model. It also generates plots for comparison, such as AUCPR and AUCROC curves.

Usage: `python Inference.py`

## Usage

To train a specific deep learning model, navigate to the `deep_learning` folder and run the corresponding training script. For example, to train the CNN+LSTM hybrid model, use the following command:

```
python CNN_LSTM_Train.py
```

To evaluate a trained model on the test data, run the corresponding testing script. For example, to test the CNN model, use the following command:

```
python CNN_Test.py
```

To generate the master table and plots for all the models, run the `Inference.py` script:

```
python Inference.py
```

Make sure to have the required dependencies installed and the dataset properly prepared before running the scripts.
