# src/algorithms/enn.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def edited_nearest_neighbor(X, y, k=3):
    """
    Edited Nearest Neighbor algorithm to clean the training set.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    k (int): Number of neighbors to use for classification.

    Returns:
    numpy.ndarray: Cleaned feature matrix.
    numpy.ndarray: Cleaned labels.
    """
    # Create a KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Iterate until no more instances can be removed
    while True:
        knn.fit(X, y)
        y_pred = knn.predict(X)

        # Identify instances where the prediction does not match the actual label
        misclassified_indices = np.where(y != y_pred)[0]

        if len(misclassified_indices) == 0:
            break

        # Remove misclassified instances
        X = np.delete(X, misclassified_indices, axis=0)
        y = np.delete(y, misclassified_indices)

    return X, y


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    X, y = X[y != 2], y[y != 2]  # Keep only two classes for the example
    X_cleaned, y_cleaned = edited_nearest_neighbor(X, y)
    print(f"Original size: {len(X)}")
    print(f"Cleaned size: {len(X_cleaned)}")
