import random
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def comparison(X: np.ndarray, y: np.ndarray, algorithm):
    """
    Compare the original and reduced dataset using the specified algorithm.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    algorithm (function): The algorithm to use for reducing the dataset.

    Returns:
    float: Original accuracy.
    float: Reduced accuracy.
    int: Original size.
    int: Reduced size.
    float: Reduction percentage.
    float: Execution time.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # set random seed to 42
    random.seed(42)

    # Train the KNN classifier on the reduced dataset
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Start timer
    start_time = time.time()

    # Apply CNN algorithm
    X_reduced, y_reduced = algorithm(X_train, y_train)

    # End timer
    end_time = time.time()

    # Train the KNN classifier on the reduced dataset
    knn_reduced = KNeighborsClassifier(n_neighbors=3)
    knn_reduced.fit(X_reduced, y_reduced)

    # Evaluate the classifier
    y_pred_reduced = knn_reduced.predict(X_test)
    accuracy_reduced = accuracy_score(y_test, y_pred_reduced)

    return (
        accuracy,
        accuracy_reduced,
        len(X_train),
        len(X_reduced),
        100 * (1 - len(X_reduced) / len(X_train)),
        end_time - start_time,
    )
