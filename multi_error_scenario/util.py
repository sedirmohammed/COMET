from typing import List, Tuple
from pathlib import Path
from datetime import datetime
from sys import stdout
from logging import DEBUG, FileHandler, StreamHandler, basicConfig
from pandas import DataFrame, concat
import category_encoders as ce
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def target_encode(df_train: DataFrame, df_test: DataFrame, categorical_columns: List[str], target_column: str)\
        -> Tuple[DataFrame, DataFrame]:
    """
    Apply target-encoding on categorical columns on train and test dataset.
    :param df_train:            Training dataset
    :param df_test:             Test dataset
    :param categorical_columns: List of names of the categorical columns
    :param target_column: Name of the target column
    """

    df_train = df_train.copy()
    df_test = df_test.copy()

    if df_train[target_column].dtype == 'O':
        le = preprocessing.LabelEncoder()
        df_train[target_column] = le.fit_transform(df_train[target_column])
        df_test[target_column] = le.transform(df_test[target_column])

    y_train = df_train[target_column].values
    y_test = df_test[target_column].values
    df_train = df_train.drop(target_column, axis=1)
    df_test = df_test.drop(target_column, axis=1)

    target_encoder = ce.TargetEncoder(cols=categorical_columns)
    df_train_encoded = target_encoder.fit_transform(df_train, y_train)
    df_test_encoded = target_encoder.transform(df_test)

    df_train_encoded[target_column] = y_train
    df_test_encoded[target_column] = y_test

    return df_train_encoded, df_test_encoded


def one_hot_encode(train_df: DataFrame, test_df: DataFrame, categorical_columns: List[str], target_column: str)\
        -> Tuple[DataFrame, DataFrame]:
    """
    Apply one-hot-encoding on categorical columns on train and test dataset.
    :param train_df:            Training dataset
    :param test_df:             Test dataset
    :param categorical_columns: List of names of the categorical columns
    :param target_column: Name of the target column
    """

    # Extract target and drop unnecessary columns
    X_train = train_df.drop(columns=target_column)
    X_test = test_df.drop(columns=target_column)
    for col in categorical_columns:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    y_train = train_df[target_column]
    y_test = test_df[target_column]

    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_cols_train = DataFrame(oh_encoder.fit_transform(X_train[categorical_columns]))
    oh_cols_train.columns = oh_encoder.get_feature_names_out(categorical_columns)
    oh_cols_valid = DataFrame(oh_encoder.transform(X_test[categorical_columns]))
    oh_cols_valid.columns = oh_encoder.get_feature_names_out(categorical_columns)
    # One-hot encoding removed index; put it back
    oh_cols_train.index = X_train.index
    oh_cols_valid.index = X_test.index

    numeric_X_train = X_train.drop(categorical_columns, axis=1)
    numeric_X_valid = X_test.drop(categorical_columns, axis=1)

    # Add one-hot encoded columns to numerical features
    train_df = concat([numeric_X_train, oh_cols_train, y_train], axis=1)
    test_df = concat([numeric_X_valid, oh_cols_valid, y_test], axis=1)

    train_df.columns = train_df.columns.astype(str)
    test_df.columns = test_df.columns.astype(str)

    return train_df, test_df


def start_logging(log_level=DEBUG, append=False, cmd_out=False, data_dir=Path('data/')):
    """
    Configures and starts logging for the project using the logging library.

    :param log_level: logging library's level of detail for log prints
    :type log_level: int (from logging log level enum)
    :param append: determines if log file should be opened in append mode, defaults to False
    :type append: bool
    :param cmd_out: whether to also print logs to commandline
    :type cmd_out: bool
    :param data_dir: path to data directory
    :type data_dir: pathlib.Path
    """
    log_file_path = data_dir / f'logs/experiment_{datetime.now():%Y_%m_%d_%H_%M_%S}.log'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = 'a' if append else 'w'
    handlers = [FileHandler(log_file_path, mode=file_mode)]
    if cmd_out:
        handlers.append(StreamHandler(stdout))

    basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=handlers
    )
