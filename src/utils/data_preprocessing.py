import pandas as pd
import os
from config import DATASET_PATH
from numpy import ndarray


def load_data(file_name: str) -> tuple[ndarray, ndarray]:
    """
    Load the data from the specified file.

    Parameters
    ----------
    file_name : str
        The name of the file to load.

    Returns
    -------
    X : numpy.ndarray
        The feature matrix of the data.
    y : numpy.ndarray
        The labels of the data.
    """
    if not os.path.exists(os.path.join(DATASET_PATH, file_name + ".csv")):
        raise FileNotFoundError(f"File {file_name} not found in {DATASET_PATH}")
    df = pd.read_csv(os.path.join(DATASET_PATH, file_name + ".csv"), header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def save_data(X: ndarray, y: ndarray, file_name: str) -> None:
    """
    Save the data to the specified file.

    Parameters
    ----------
    X : numpy.ndarray
        The feature matrix of the data.
    y : numpy.ndarray
        The labels of the data.
    file_name : str
        The name of the file to save.
    """
    df = pd.DataFrame(X)
    df["target"] = y
    df.to_csv(f"{file_name}.csv", index=False, header=False)


def under_sampling(
    X: ndarray, y: ndarray, ratio: float = None, n_samples: int = None
) -> tuple[ndarray, ndarray]:
    """
    Under sample the dataset to the specified ratio or number of samples.

    Parameters
    ----------
    X : numpy.ndarray
        The feature matrix of the data.
    y : numpy.ndarray
        The labels of the data.
    ratio : float, optional
        The ratio of the samples to keep, by default None.
    n_samples : int, optional
        The number of samples to keep, by default None.

    Returns
    -------
    X : numpy.ndarray
        The under sampled feature matrix of the data.
    y : numpy.ndarray
        The under sampled labels of the data.
    """
    if not ratio and not n_samples:
        raise ValueError("Either ratio or n_samples should be provided.")

    if ratio:
        n_samples = int(len(X) * ratio)

    classes = set(y)
    n_classes = len(classes)
    samples_per_class = n_samples // n_classes

    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(sampling_strategy={c: samples_per_class for c in classes})
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled
