import pandas as pd
import os
from config import DATASET_PATH
from numpy import ndarray


def load_data(file_name: str) -> tuple[ndarray, ndarray]:
    """
    Load the data from the specified file.

    Parameters:
    file_name (str): The name of the file to load.

    Returns:
    X (numpy.ndarray): The feature matrix of the data.
    y (numpy.ndarray): The labels of the data.
    """
    if not os.path.exists(os.path.join(DATASET_PATH, file_name + ".csv")):
        raise FileNotFoundError(f"File {file_name} not found in {DATASET_PATH}")
    df = pd.read_csv(os.path.join(DATASET_PATH, file_name + ".csv"), header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values
