# src/algorithms/enn.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.algorithms.base import BaseAlgorithm


class ENN(BaseAlgorithm):
    """
    Edited Nearest Neighbors (ENN) algorithm for noise reduction.

    Parameters:
    k (int): Number of neighbors to use for the k-nearest neighbors algorithm. Default is 3.
    """

    def __init__(self, n_neighbors: int = 3, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.n_neighbors: int = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors, metric=metric)

    def _fit(self) -> np.ndarray:
        """
        Perform the Edited Nearest Neighbors algorithm.

        Returns:
        np.ndarray: Indices of the samples that were not misclassified.
        """
        self.classifier.fit(self.X, self.y)
        y_pred = self.classifier.predict(self.X)
        misclassified_indices = np.where(self.y != y_pred)[0]
        sample_indices = np.setdiff1d(np.arange(len(self.X)), misclassified_indices)
        return sample_indices
