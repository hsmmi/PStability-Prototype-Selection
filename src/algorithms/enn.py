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

    def __init__(self, k: int = 3):
        self.k = k
        self.sample_indices_ = []
        self.X_ = None
        self.y_ = None

    def select(self) -> np.ndarray:
        """
        Perform the Edited Nearest Neighbors algorithm.

        Returns:
        np.ndarray: Indices of the samples that were not misclassified.
        """
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(self.X, self.y)
        y_pred = knn.predict(self.X)
        misclassified_indices = np.where(self.y != y_pred)[0]
        sample_indices = np.setdiff1d(np.arange(len(self.X)), misclassified_indices)
        return sample_indices


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    X, y = X[y != 2], y[y != 2]  # Keep only two classes for the example

    enn = ENN(k=3)
    X_cleaned, y_cleaned = enn.fit(X, y).transform(X, y)

    print(f"Original size: {len(X)}")
    print(f"Cleaned size: {len(X_cleaned)}")
