import pandas as pd
import os

from sklearn.discriminant_analysis import StandardScaler


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    return df


def preprocess_data(df: pd.DataFrame):
    # Convert the labels[-1] to integers
    df.iloc[:, -1] = pd.factorize(df.iloc[:, -1])[0]
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    return df


def save_processed_data(df: pd.DataFrame, file_path):
    df.to_csv(file_path, index=False, header=False)
