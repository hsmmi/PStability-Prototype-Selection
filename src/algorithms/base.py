# Create base class for instance selection algorithms
from abc import ABC

import numpy as np


class BaseAlgorithm(ABC):
    """
    Base class for instance selection algorithms.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.sample_indices_ = None
        self.X_ = None
        self.y_ = None
        self.reduction_ratio = None

    def fit(self, X, y):
        """
        Fit the model using the training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        self: Fitted instance of the algorithm.
        """
        self.X = X
        self.y = y
        self.sample_indices_ = self.select(X, y)
        self.X_ = X[self.sample_indices_]
        self.y_ = y[self.sample_indices_]
        self.reduction_ratio = 1 - len(self.X_) / len(X)
        return self

    def select(self, X, y):
        """
        Select instances from the training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        np.ndarray: Indices of the selected instances.
        """
        raise NotImplementedError(
            "The select method must be implemented in a subclass."
        )

    def transform(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform the data by selecting instances.

        Parameters:
        X (np.ndarray): Data to transform.
        y (np.ndarray): Target values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed data and target values.
        """
        if self.sample_indices_ is None:
            raise ValueError("The model has not been fitted yet.")
        return X[self.sample_indices_], y[self.sample_indices_]

    def fit_transform(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the model and transform the data in a single step.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed data and target values.
        """
        self.fit(X, y)
        return self.transform(X, y)
