import argparse 
import os
import pandas as pd
import numpy as np
import category_encoders as ce
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import re
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

# set random seed
seed = 42

np.random.seed(seed=seed)

# System path
parent_dir = os.getcwd()

def read_data(data_path, file_type="csv"):
    """
    Read data from the specified path and file type.
    
    :param data_path: path to the data file
    :param file_type: type of the file to be read
    :returns: Pandas DataFrame
    """
    try:
        if file_type == "csv":
            return pd.read_csv(data_path)
        elif file_type == "parquet":
            return pd.read_parquet(data_path)
        elif file_type == "xls" or file_type == "xlsx":
            pd_df = pd.read_excel(data_path)
            return pd.from_pandas(pd_df)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at specified path: {data_path}")
    except Exception as e:
        raise Exception(f"Error reading {data_path}: {e}")


def save_data(data, output_path, file_type="csv"):
    """
    Save preprocessed data to the specified path and file type.
    
    :param data: preprocessed Pandas DataFrame
    :param output_path: path to save the preprocessed data
    :param file_type: type of the file to be saved
    :returns: None
    """
    # Extract directory part from output_path
    output_dir = os.path.dirname(output_path)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if file_type == "csv":
            data.to_csv(output_path, index=False)
        elif file_type == "parquet":
            data.to_parquet(output_path, index=False)
        elif file_type == "xls" or file_type == "xlsx":
            data_pd = data.to_pandas()
            data_pd.to_excel(output_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {output_path}: {e}")


na_cleaner_modes = ["remove row", "mean", "mode"]
def preprocess_data(dataframe, 
            detect_binary=True, 
            numeric_dtype=True, 
            one_hot=True, 
            na_cleaner_mode="remove row", 
            normalize=True,
            balance=False,
            sample = False,
            sample_size = 0.2,
            stratify_column = None, 
            datetime_columns=[],
            clean_columns=[], 
            remove_columns=[], 
            consider_as_categorical=[],
            target='',
            verbose=True):

    """
    Performs data preprocessing on a given Pandas DataFrame.

    Parameters:
    :param dataframe: Pandas DataFrame to be preprocessed.
    :param detect_binary: If True, the function will automatically detect and encode 
                     columns based on the number of unique values:
        - 2 unique values: Binary encoding (0 and 1).
        - 3 to 10 unique values: Label encoding.
        - More than 10 unique values: Target encoding.
      Default is True.
    :param numeric_dtype: If True, columns will be converted to numeric data type 
                     when possible. Default is True.
    :param one_hot: If True, non-numeric columns will be one-hot encoded. 
               Default is True.
    :param na_cleaner_mode: Technique for handling missing values. Options:
        - False: Do not clean missing values.
        - 'remove row': Remove rows with missing values.
        - 'mean': Replace missing values with the column mean.
        - 'mode': Replace missing values with the column mode.
        - '*': Replace missing values with the specified value.
      Default is 'mode'.
    :param normalize: If True, non-binary columns will be normalized. 
                 Default is True.
    :param balance: If True, the dataset will be balanced using SMOTE. 
               Default is False.
    :param sample: If True, the function will sample the data. Default is False.
    :param sample_size: Size of the sample. Default is 0.2.
    :param stratify_column: Column to stratify the sample. Default is None.
    :param datetime_columns: List of columns containing date/time values.
                        Default is ['Time'].
    :param clean_columns: List of columns to be cleaned (removing special characters).
                  Default is ['Amount'].
    :param remove_columns: List of columns to be removed from the DataFrame.
                      Default is [].
    :param consider_as_categorical: List of columns to be considered as categorical.
                               Default is ['Use Chip', 'Merchant City', 
                               'Merchant State', 'Zip', 'MCC', 'Errors?'].
    :param target: Target column for balancing (if balance is True).
              Default is 'Is Fraud?'.
    :param verbose: If True, progress will be printed. Default is True.
    
    :returns: Preprocessed Pandas DataFrame.
    """
    
    df = dataframe.copy()

    if verbose:
        print("*" * 30, " Data Cleaning Started ", "*" * 30)
    
    assert type(one_hot) == type(True), "Please ensure that one_hot param is bool (True/False)"
    assert type(df) == type(pd.DataFrame()), "Parameter 'df' should be Pandas DataFrame"
    
    # cleaning columns --------------------------------------------------------------------------------------------------

    if verbose: 
            print("=  Cleaning columns... ")
            
    if datetime_columns        : datetime_columns        = [clean_col_name(col) for col in datetime_columns]
    if clean_columns           : clean_columns           = [clean_col_name(col) for col in clean_columns]
    if remove_columns          : remove_columns          = [clean_col_name(col) for col in remove_columns]
    if consider_as_categorical : consider_as_categorical = [clean_col_name(col) for col in consider_as_categorical]
    if stratify_column         : stratify_column         = clean_col_name(stratify_column)
    if target                  : target                  = clean_col_name(target)
    
    df = clean_columns_df(df, clean_columns, verbose)
 
    # casting datetime columns to datetime dtypes -----------------------------------------------------------------------
    if len(datetime_columns) > 0:
        if verbose: 
            print("=  Casting datetime columns to datetime dtype... ")
        df = convert_datetime_dtype_df(df, datetime_columns, verbose)

    # removing unwanted columns  ----------------------------------------------------------------------------------------
    if len(remove_columns) > 0: 
        if verbose: 
            print("=  Performing removal of unwanted columns... ")
        df = remove_columns_df(df, remove_columns, verbose)
        
    # sample data -------------------------------------------------------------------------------------------------------
    if sample:
        if verbose: 
            print("=  Sampling data... ")
        df = sample_data(df, sample_size, stratify_column, verbose)

    # clean None (na) values --------------------------------------------------------------------------------------------
    if na_cleaner_mode != False: 
        if verbose: 
            print("=  Performing None/NA/Empty values cleaning... ")
        df = clean_na_df(df, na_cleaner_mode, verbose)

    # detecting binary columns ------------------------------------------------------------------------------------------
    if detect_binary: 
        if verbose: 
            print("=  Performing encoding based on unique values...")
        df = encode_categorical_columns(df, consider_as_categorical ,target, binary_threshold=2, high_cardinality_threshold=10, verbose=verbose)

    # checking if any columns can be converted to numeric dtypes  -------------------------------------------------------
    if numeric_dtype: 
        if verbose: 
            print("= Converting columns to numeric dtypes when possible...")
        df = convert_numeric_df(df, exclude=datetime_columns, force=False, verbose=verbose)

    # normalize all columns (binary 0,1 columns are excluded) -----------------------------------------------------------
    if normalize: 
        if verbose: 
            print("=  Performing dataset normalization... ")
        df = normalize_df(df, exclude=datetime_columns, verbose=verbose)
    
    if balance: 
        if verbose: 
            print("=  Balancing dataset... ")
        df = balance_df(df, target, verbose)
        
    if verbose: 
        print("*" * 30, " Data Cleaning Finished ", "*" * 30)

    return df




""" ------------------------------------------------------------------------------------------------------------------------- """


def sample_data(df, sample_size, stratify_column=None, verbose=True):
    """
    sample_data function samples the data based on the specified sample size and stratify column
    :param df: input Pandas DataFrame
    :param sample_size: size of the sample.(default: 0.2)
    :param stratify_column: column to stratify the sample (default: None)
    :param verbose: print progress in terminal/cmd (default: True)
    :returns: sampled Pandas DataFrame
    """
    try:
        if isinstance(sample_size, float) and 0.0 <= sample_size <= 1.0:
            sample_frac = sample_size

        else:
            raise ValueError(" - Invalid sample_size. Provide either a percentage (0.0-1.0)")
        
        if stratify_column is not None and stratify_column not in df.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found in DataFrame.")
        
        if stratify_column is not None:
            if verbose:
                print(" + sampling data with stratification")
            df = df.groupby(stratify_column, group_keys=False).apply(lambda x: x.sample(frac=sample_frac,  random_state=42))
        else:
            if verbose:
                print(" + sampling data")
            df = df.sample(frac=sample_frac, random_state=42)
            df = df.reset_index(drop=True)
        
        if verbose:
            print(" + sampled data successfully. Number of samples: {}. Number of stratified samples: {}".format(df.shape[0], df[stratify_column].value_counts().ser if stratify_column else "N/A"))
        
        return df
    
    except Exception as e:
        print(" ERROR: {}".format(e))

def encode_categorical_columns(df, consider_as_categorical, target, binary_threshold=2, label_threshold=3, high_cardinality_threshold=10, one_hot=True, verbose=True):
    """
    encode_categorical_columns function detects binary columns, high cardinality columns, and columns for label encoding
    based on the specified cutoffs, and applies the appropriate encoding to the categorical columns in the DataFrame.
    
    :param df: input Pandas DataFrame
    :param target: target column to be excluded from encoding
    :param binary_threshold: threshold value to determine if a column is binary (default: 2)
    :param label_threshold: threshold value to determine if a column should be label encoded (default: 3)
    :param high_cardinality_threshold: threshold value to determine if a column has high cardinality (default: 10)
    :param one_hot: flag to indicate whether to apply one-hot encoding to columns with less than 8 unique values (default: False)
    :param verbose: print progress in terminal/cmd
    
    :returns: DataFrame with encoded categorical columns
    """
    try:
        if consider_as_categorical:
            categorical_cols = consider_as_categorical 
        else:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
        
        binary_cols = []
        high_cardinality_cols = []
        label_encoding_cols = []
        
        if target in df.columns:
            if len(df[target].unique()) == 2:
                if 'Yes' in df[target].unique() and 'No' in df[target].unique():
                    df[target] = df[target].map({'No': 0, 'Yes': 1})
                elif 'True' in df[target].unique() and 'False' in df[target].unique():
                    df[target] = df[target].map({'False': 0, 'True': 1})
                else:
                    df[target] = df[target].map({df[target].unique()[0]: 0, df[target].unique()[1]: 1})
            elif (len(df[target].unique()) > 2) and (len(df[target].unique()) <= 25):
                label_encoder = ce.OrdinalEncoder(cols=target)
                df[target] = label_encoder.fit_transform(df[target])                

        for col in categorical_cols:
            if col != target:
                unique_values = len(df[col].unique())
                if unique_values <= binary_threshold:
                    binary_cols.append(col)
                elif unique_values > high_cardinality_threshold:
                    high_cardinality_cols.append(col)
                elif unique_values <= label_threshold:
                    label_encoding_cols.append(col)
        
        if verbose:
            print(" + detected {} binary columns: {}".format(len(binary_cols), binary_cols))
            print(" + detected {} high cardinality columns: {}".format(len(high_cardinality_cols), high_cardinality_cols))
            print(" + detected {} columns for label encoding: {}".format(len(label_encoding_cols), label_encoding_cols))
        
        for col in binary_cols:
            if one_hot:
                print(col)
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            else:
                if 'Yes' in df[col].unique() and 'No' in df[col].unique():
                    df[col] = df[col].map({'No': 0, 'Yes': 1})
                elif 'True' in df[col].unique() and 'False' in df[col].unique():
                    df[col] = df[col].map({'False': 0, 'True': 1})
                else:
                    df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})
        
        target_encoder = ce.TargetEncoder(cols=high_cardinality_cols, return_df=True)
        df[high_cardinality_cols] = target_encoder.fit_transform(df[high_cardinality_cols], df[target])
        
        if one_hot:
            if verbose:
                print(" + Performing one-hot encoding to columns: {}".format(label_encoding_cols))
            df = pd.get_dummies(df, columns=label_encoding_cols)

        else:
            label_encoder = ce.OrdinalEncoder(cols=label_encoding_cols, return_df=True)
            df[label_encoding_cols] = label_encoder.fit_transform(df[label_encoding_cols])
        
        return df

    except Exception as e:
        print(" ERROR {}".format(e))

def clean_col_name(col):
    
    '''
    clean_col_name function cleans column names by removing special characters and replacing spaces with underscores
    
    :param col: column name to be cleaned
    :returns: cleaned column name    
    '''
    
    col = re.sub(r'[^a-zA-Z0-9\s]', '', col)
    
    col = re.sub(r'\s+', '_', col)
    
    return col
    
    

def clean_columns_df(df, clean_columns=None, verbose=True):
    """
    clean_columns_df function cleans column names and values in the DataFrame by removing special characters and replacing spaces with underscores
    
    :param df: input Pandas DataFrame
    :param clean_columns: list of columns to be cleaned (default: None)
    :param verbose: print progress in terminal/cmd
    :returns: cleaned Pandas DataFrame
    
    """
    for col in df.columns:

        cleaned_col_name = clean_col_name(col)
        
        df = df.rename(columns={col: cleaned_col_name})
        
        if clean_columns:
            if col in clean_columns:
                df[col] = df[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s\.]', '', str(x)))
    
    return df



def convert_datetime_dtype_series(series, verbose=True):
    """
    convert_datetime_dtype_series function attempts to cast a given column in the DataFrame to datetime dtype using multiple formats 
    and creates new columns for datetime components
    
    :param series: input Pandas Series
    :param verbose: print progress in terminal/cmd (default: True)
    :returns: processed Pandas Series and datetime format
    
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S.%f',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y',
        '%m/%d/%y %H:%M:%S',
        '%m/%d/%y %H:%M:%S.%f',
        '%m/%d/%y %H:%M',
        '%m/%d/%y',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S.%f',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y',
        '%d/%m/%y %H:%M:%S',
        '%d/%m/%y %H:%M:%S.%f',
        '%d/%m/%y %H:%M',
        '%d/%m/%y',
        '%H:%M:%S',
        '%H:%M:%S.%f',
        '%H:%M'
    ]

    for fmt in formats:
        try:
            series = pd.to_datetime(series, format=fmt)
            if verbose:
                print(f" + converted column {series.name} to datetime dtype using format {fmt}")
            return series, fmt
        except (ValueError, TypeError):
            pass

    if verbose:
        print(f" - unable to convert column {series.name} to datetime dtype")
    return series

def convert_datetime_dtype_df(df, cols, verbose=True):
    """
    convert_datetime_dtype_df function attempts to cast given columns in dataframe to datetime dtype
    and creates new columns for datetime components
    
    :param df: input Pandas DataFrame
    :param cols: list of column names to convert to datetime dtype
    :param verbose: print progress in terminal/cmd (default: True)
    :returns: processed Pandas DataFrame
    """
    for c in cols:
        converted_series, fmt = convert_datetime_dtype_series(df[c], verbose)
        if fmt is not None:
            df[c] = converted_series
            dt = converted_series.dt
            if '%Y' in fmt:
                df[f'{c}_year'] = dt.year
            if '%m' in fmt:
                df[f'{c}_month'] = dt.month
            if '%d' in fmt:
                df[f'{c}_day'] = dt.day
            if '%H' in fmt:
                df[f'{c}_hour'] = dt.hour
            if '%M' in fmt:
                df[f'{c}_minute'] = dt.minute
            if '%S' in fmt:
                df[f'{c}_second'] = dt.second
            if '.%f' in fmt:
                df[f'{c}_microsecond'] = dt.microsecond

    df = df.drop(columns=cols)
    return df


def remove_columns_df(df, remove_columns, verbose=True): 
    """
    remove_columns_df function removes columns in 'remove_columns' param list and returns df 
    
    :param df: input Pandas DataFrame 
    :param remove_columns: list of columns to be removed from the dataframe 
    :param verbose: print progress in terminal/cmd
    :returns: processed Pandas DataFrame 
    """
    stat = 0
    for col in remove_columns: 
        assert col in df.columns.to_list(), "{} is marked to be removed, but it does not exist in the dataset/dataframe".format(col)
        
        df.drop(columns=col, inplace=True)
        stat += 1
    if verbose: 
        print("  + removed {} columns successfully.".format(stat))
    return df

def convert_numeric_series(series, force=False, verbose=True): 
    """
    convert_numeric_series function converts columns of dataframe to numeric dtypes when possible safely 
    if the values that cannot be converted to numeric dtype are minority in the series (< %25), then
    these minority values will be converted to NaN and the series will be forced to numeric dtype 

    :param series: input Pandas Series
    :param force: if True, values which cannot be casted to numeric dtype will be replaced with NaN 'see pandas.to_numeric() docs' (be careful with force=True)
    :param verbose: print progress in terminal/cmd
    :returns: Pandas series
    """
    stats = 0
    if force: 
        stats += series.shape[0]
        return pd.to_numeric(series, errors='coerce'), stats
    else: 
        non_numeric_count = pd.to_numeric(series, errors='coerce').isna().sum()
        if non_numeric_count/series.shape[0] < 0.25: 

            stats += series.shape[0]
            if verbose and non_numeric_count != 0: 
                print("  + {} minority (minority means < %25 of '{}' entries) values that cannot be converted to numeric dtype in column '{}' have been set to NaN, nan cleaner function will deal with them".format(non_numeric_count, series.name, series.name))
            return pd.to_numeric(series, errors='coerce'), stats
        else: 
            return series, stats


def convert_numeric_df(df, exclude=[], force=False, verbose=True):
    """
    convert_numeric_df function converts dataframe columns to numeric dtypes when possible safely 
    if the values in a particular columns that cannot be converted to numeric dtype are minority in that column (< %25), then
    these minority values will be converted to NaN and the column will be forced to numeric dtype 

    :param df: input Pandas DataFrame
    :param exclude: list of columns to be excluded whice converting dataframe columns to numeric dtype (usually datetime columns)
    :param force: if True, values which cannot be casted to numeric dtype will be replaced with NaN 'see pandas.to_numeric() docs' (be careful with force=True)
    :param verbose: print progress in terminal/cmd
    :returns: Pandas DataFrame
    """
    stats = 0
    for col in df.columns.to_list(): 
        if col in exclude:
            continue
        df[col], stats_temp = convert_numeric_series(df[col], force, verbose)
    stats += stats_temp
    if verbose: 
        print("  + converted {} cells to numeric dtypes".format(stats))
    return df 



def clean_na_series(series, na_cleaner_mode): 
    """ 
    clean_nones function manipulates None/NA values in a given panda series according to cleaner_mode parameter
        
    :param series: the Panda Series in which the cleaning will be performed 
    :param na_cleaner_mode: what cleaning technique to apply, 'na_cleaner_modes' for a list of all possibilities 
    :returns: cleaned version of the passed Series
    """
    if na_cleaner_mode == 'remove row': 
        return series.dropna()
    elif na_cleaner_mode == 'mean':
        mean = series.mean()
        return series.fillna(mean)
    elif na_cleaner_mode == 'mode':
        mode = series.mode()[0]
        return series.fillna(mode)
    elif na_cleaner_mode == False: 
        return series
    else: 
        return series.fillna(na_cleaner_mode)


def clean_na_df(df, na_cleaner_mode, verbose=True): 
    """
    clean_na_df function cleans all columns in DataFrame as per given na_cleaner_mode
    
    :param df: input DataFrame
    :param na_cleaner_mode: what technique to apply to clean na values 
    :param verbose: print progress in terminal/cmd
    :returns: cleaned Pandas DataFrame 
    """
    stats = {}
    for col in df.columns.to_list(): 
        if df[col].isna().sum() > 0: 
            stats[col + " NaN Values"] = df[col].isna().sum()
            try:
                df[col] = clean_na_series(df[col], na_cleaner_mode)
            except: 
                print("  + could not find {} for column {}, will use fill NaN values".format(na_cleaner_mode, col))
                df[col] = clean_na_series(df[col], 'remove row')
    if verbose: 
        print("  + cleaned the following NaN values: {}".format(stats))
    return df

def normalize_df(df, exclude=[], verbose=True): 
    """
    normalize_df function performs normalization to all columns of dataframe excluding binary (1/0) columns 
    
    :param df: input Pandas DataFrame
    :param exclude: list of columns to be excluded when performing normalization (usually datetime columns)
    :param verbose: print progress in terminal/cmd
    :returns: normalized Pandas DataFrame 
    """
    stats = 0
    for col in df.columns.to_list(): 
        if col in exclude: 
            continue
        col_unique = df[col].unique().tolist()
        if len(col_unique) == 2 and 0 in col_unique and 1 in col_unique: 
            continue
        else: 
            df[col] = (df[col]-df[col].mean())/df[col].std()
            stats += df.shape[0]
    if verbose: 
        print("  + normalized {} cells".format(stats))
    return df

def balance_df(df, target, verbose=True): 
    """
    balance_df function balances the dataset by oversampling the minority class using SMOTE
    
    :param df: input Pandas DataFrame
    :param target: target column to be balanced
    :param verbose: print progress in terminal/cmd
    :returns: balanced Pandas DataFrame
    """
    try:
        smote = SMOTE()
        X = df.drop(columns=[target])
        y = df[target]
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
        if verbose:
            print("  + balanced dataset using SMOTE")
        return df_resampled
    except Exception as e:
        print(f"  + An error occurred while balancing the dataset: {e}")
        return df

    
def split_data(X, y, data=None, val_data = False, val_size = 0.2 ,test_size=0.2):
    
    '''
    Split the data into train and test sets.
    
    :param X: input features
    :param y: target variable
    :param data: data dictionary containing the target variable
    :param test_size: size of the test set
    :returns: train and test sets
    '''
    
    try:
        if data is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
            if val_data:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
                return X_train, X_val, X_test, y_train, y_val, y_test
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
            if val_data:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
                return X_train, X_val, X_test, y_train, y_val, y_test
            return X_train, X_test, y_train, y_test

    except Exception as e:
        raise Exception(f"Error splitting data: {e}")



        
def main():

    default_data_path = "Data/Raw Data/data.csv"
    default_output_path = "Data/Processed Data/data.csv"

    parser = argparse.ArgumentParser(description="Preprocess data")
    
    
    parser.add_argument("--data_path", type=str, default=default_data_path, 
                        help=f"Path to the data file. Default: {default_data_path}")
    parser.add_argument("--output_path", type=str, default=default_output_path,
                        help=f"Path to the output file. Default: {default_output_path}")
    parser.add_argument("--file_type", type=str, default="csv",
                        help="Type of the file to be read. Options: csv, parquet, xls, etc. Default: csv")
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set. Default: 0.2')
    parser.add_argument('--val_data', type=bool, action=argparse.BooleanOptionalAction, default=False, help='If True, the function will split the data into train, validation, and test sets. Default: False')   
    parser.add_argument('--val_size', type=float, default=0.2, help='Size of the validation set. Default: 0.2')
    parser.add_argument('--detect_binary', type=bool, action=argparse.BooleanOptionalAction, default=True, help='If True, the function will automatically detect and encode columns based on the number of unique values. Default: True')
    parser.add_argument('--numeric_dtype', type=bool, action=argparse.BooleanOptionalAction, default=True, help='If True, columns will be converted to numeric data type when possible. Default: True')
    parser.add_argument('--one_hot', type=bool, action=argparse.BooleanOptionalAction, default=True, help='If True, non-numeric columns will be one-hot encoded. Default: True')
    parser.add_argument('--na_cleaner_mode', type=str, default="remove row", help='Technique for handling missing values. Options: remove row, mean, mode, *. Default: mode')
    parser.add_argument('--normalize', type=bool, action=argparse.BooleanOptionalAction, default=True, help='If True, non-binary columns will be normalized. Do not use it with numeric_dtype = True. Default: True')
    parser.add_argument('--balance', type=bool, action=argparse.BooleanOptionalAction, default=False, help='If True, the dataset will be balanced using SMOTE. Default: False')
    parser.add_argument('--sample', type=bool, action=argparse.BooleanOptionalAction, default=False, help='If True, the function will sample the data. Default: False')
    parser.add_argument('--sample_size', type=float, default=0.2, help='Size of the sample. Default: 0.2')
    parser.add_argument('--stratify_column', type=str, default=None, help='Column to stratify the sample. Default: None')
    parser.add_argument('--datetime_columns', nargs='*', default=[], help='List of columns containing date/time values. Default: []')
    parser.add_argument('--clean_columns', nargs='*', default=[], help='List of columns to be cleaned (removing special characters). Default: []')
    parser.add_argument('--remove_columns', nargs='*', default=[], help='List of columns to be removed from the DataFrame. Default: []')
    parser.add_argument('--consider_as_categorical', nargs='*', default=[], help='List of columns to be considered as categorical. Default: []')
    parser.add_argument('--target', type=str, default=None, help='Target column')
    parser.add_argument('--verbose', type=bool, action=argparse.BooleanOptionalAction, default=True, help='If True, progress will be printed. Default: True')
   
    args = parser.parse_args()   
    
    data_path = os.path.join(parent_dir, args.data_path) if args.data_path == default_data_path else args.data_path
    output_path = os.path.join(parent_dir, args.output_path) if args.output_path == default_output_path else args.output_path
    args.datetime_columns = args.datetime_columns[0].split(',') if args.datetime_columns else []
    args.clean_columns = args.clean_columns[0].split(',') if args.clean_columns else []
    args.remove_columns = args.remove_columns[0].split(',') if args.remove_columns else []
    args.consider_as_categorical = args.consider_as_categorical[0].split(',') if args.consider_as_categorical else []
    
    try:
        data = read_data(data_path=data_path, file_type=args.file_type)
        print(data.head())

        data_processed = preprocess_data(dataframe=data, 
                                         detect_binary=args.detect_binary, 
                                         numeric_dtype=args.numeric_dtype, 
                                         one_hot=args.one_hot, 
                                         na_cleaner_mode=args.na_cleaner_mode, 
                                         normalize=args.normalize,
                                         balance=args.balance,
                                         sample=args.sample,
                                         sample_size=args.sample_size,
                                         stratify_column=args.stratify_column,
                                         datetime_columns=args.datetime_columns,
                                         clean_columns=args.clean_columns,
                                         remove_columns=args.remove_columns,
                                         consider_as_categorical=args.consider_as_categorical,
                                         target=args.target,
                                         verbose=args.verbose)
        print(data_processed.head())
        
        print(f"Data shape before processing: {data.shape}")
        print(f"Data shape after processing: {data_processed.shape}")
        
        save_data(data=data_processed, output_path=output_path, file_type=args.file_type)
        print(f"Data successfully processed and saved to {output_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
