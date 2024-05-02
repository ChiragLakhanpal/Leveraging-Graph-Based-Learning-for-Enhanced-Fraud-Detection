
## **Please note**:
All files in the `components` folder, except for `preprocess.py`, are not standalone files. They are designed to be called internally by other code files in the `main_code` directory.

## `preprocess.py`

This script performs data preprocessing on the input dataset. It provides various options for cleaning, transforming, and preparing the data for machine learning tasks.

### Usage

To use the `preprocess.py` script as a standalone script, run the following command:

```
python preprocess.py \
  --data_path /path/to/your/data.csv \
  --output_path /path/to/keep/processed/data.csv \
  --file_type "file type to use. CSV, Parquet, Xls"
  --test_size 0.2 \
  --no-val_data \
  --val_size 0.2 \
  --detect_binary \
  --no-numeric_dtype \
  --one_hot \
  --na_cleaner_mode "remove row" \
  --no-normalize \
  --no-balance \
  --sample \
  --sample_size 0.2 \
  --stratify_column 'target_column' \
  --datetime_columns 'datetime_column1,datetime_column2' \
  --clean_columns 'column1,column2' \
  --remove_columns 'column1,column2' \
  --consider_as_categorical 'category_column1,category_column2' \
  --target 'target_column' \
  --verbose
```

### Arguments

- `--data_path`: Path to the input data file (default: `"../../../Data/Raw Data/data.csv"`).
- `--output_path`: Path to save the preprocessed data (default: `"../../../Data/Processed Data/data.csv"`).
- `--file_type`: Type of the input file (default: `"csv"`).
- `--test_size`: Size of the test set (default: `0.2`).
- `--val_data`: If specified, the script will split the data into train, validation, and test sets.
- `--val_size`: Size of the validation set (default: `0.2`).
- `--detect_binary`: If specified, the script will automatically detect and encode columns based on the number of unique values.
- `--numeric_dtype`: If specified, columns will be converted to numeric data type when possible.
- `--one_hot`: If specified, non-numeric columns will be one-hot encoded.
- `--na_cleaner_mode`: Technique for handling missing values (default: `"remove row"`).
- `--normalize`: If specified, non-binary columns will be normalized.
- `--balance`: If specified, the dataset will be balanced using SMOTE.
- `--sample`: If specified, the script will sample the data.
- `--sample_size`: Size of the sample (default: `0.2`).
- `--stratify_column`: Column to stratify the sample (default: `None`).
- `--datetime_columns`: List of columns containing date/time values (default: `[]`).
- `--clean_columns`: List of columns to be cleaned by removing special characters (default: `[]`).
- `--remove_columns`: List of columns to be removed from the DataFrame (default: `[]`).
- `--consider_as_categorical`: List of columns to be considered as categorical (default: `[]`).
- `--target`: Target column (default: `None`).
- `--verbose`: If specified, progress will be printed.

### Functionality

The `preprocess.py` script performs the following tasks:

1. Reads the input data file.
2. Cleans column names and values by removing special characters.
3. Converts specified columns to datetime data type and creates new columns for datetime components.
4. Removes unwanted columns.
5. Handles missing values based on the specified `na_cleaner_mode`.
6. Detects and encodes binary, high cardinality, and label encoding columns.
7. Converts columns to numeric data type when possible.
8. Normalizes non-binary columns.
9. Balances the dataset using SMOTE if specified.
10. Samples the data if specified.
11. Saves the preprocessed data to the specified output path.

The script provides flexibility in configuring various preprocessing options based on the specific requirements of the dataset and the machine learning task at hand.

## `classical_machine_learning/utils.py`

The `utils.py` file within the `classical_machine_learning` directory provides utility functions specifically for classical machine learning models. These functions include data splitting, model evaluation metrics, hyperparameter tuning, and other common tasks related to training and evaluating classical machine learning algorithms.

## `deep_learning/dataloader.py`

The `dataloader.py` file in the `deep_learning` directory is responsible for loading and preprocessing data for deep learning models. It defines data loading pipelines, including data batch generation, and data normalization. The dataloader ensures that the data is efficiently loaded and fed into the deep learning models during training and inference.

## `deep_learning/models.py`

The `models.py` file contains the definition of various deep learning model architectures. It includes classes and functions that define the structure and layers of different neural networks, such as Convolutional Neural Networks (CNNs), Long short-term memory (LSTM), and a hybrid (LSTM + CNN) model. These models are used for the binary classification task.

## `deep_learning/utils.py`

Similar to the `utils.py` file in the `classical_machine_learning` directory, this file provides utility functions specific to deep learning models. It includes functions for model training, evaluation, checkpoint saving and loading, learning rate scheduling, and other common tasks related to deep learning workflows.

## `gnn/model.py`

The `model.py` file in the `gnn` directory defines the architecture of graph neural network (GNN) models. It includes classes and functions that specify the layers and structure of RGCN variant. This model is designed to operate on graph-structured data and learn node representations.

## `gnn/utils.py`

The `utils.py` file within the `gnn` directory contains utility functions specific to graph neural networks. It includes functions for graph data preprocessing, graph sampling, graph visualization, and evaluation metrics tailored for graph-based tasks. These utilities support the training, evaluation, and analysis of GNN models.
