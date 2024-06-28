# src/algorithms/cnn.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random


def condensed_nearest_neighbor(X, y):
    """
    Condensed Nearest Neighbor algorithm to reduce the training set.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.

    Returns:
    numpy.ndarray: Reduced feature matrix.
    numpy.ndarray: Reduced labels.
    """
    # Initialize the set of prototypes with the first instance of each class
    X_prototypes = []
    y_prototypes = []

    # Start with the first instance of each class
    for label in np.unique(y):
        index = random.choice(np.where(y == label)[0])
        X_prototypes.append(X[index])
        y_prototypes.append(y[index])

    # Convert lists to numpy arrays
    X_prototypes = np.array(X_prototypes)
    y_prototypes = np.array(y_prototypes)

    # Create a KNN classifier with k=1
    knn = KNeighborsClassifier(n_neighbors=1)

    # Iterate through all instances in the training set
    changes = True
    while changes:
        changes = False
        for i in range(len(X)):
            knn.fit(X_prototypes, y_prototypes)
            if knn.predict([X[i]])[0] != y[i]:
                # Add the misclassified instance to the set of prototypes
                X_prototypes = np.vstack([X_prototypes, X[i]])
                y_prototypes = np.append(y_prototypes, y[i])
                changes = True
                break  # Restart the while loop to re-evaluate with the new prototype set
    return X_prototypes, y_prototypes


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    X_reduced, y_reduced = condensed_nearest_neighbor(X, y)
    print(f"Original size: {len(X)}")
    print(f"Reduced size: {len(X_reduced)}")
