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


# Compare different prororype selection with n-fold cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
import time


def compare_prototype_selection(
    X: np.ndarray, y: np.ndarray, algorithms: list, k: int = 3, n_folds: int = 5
):
    """
    Compare different prototype selection algorithms using n-fold cross validation.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    algorithms (list): List of algorithms to use for reducing the dataset.
    k (int): Number of neighbors to use for classification.
    n_folds (int): Number of folds for cross validation.

    Returns:
    dict: Dictionary containing the results for each algorithm.
    """
    results = {algorithm.__name__: [] for algorithm in algorithms}
    results["Original"] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the KNN classifier on the reduced dataset
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Evaluate the classifier
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results["Original"].append(
            [
                accuracy,
                len(X_train),
                0,
                -1,
            ]
        )

        for algorithm in algorithms:
            # Start timer
            start_time = time.time()

            # Apply the algorithm
            X_reduced, y_reduced = algorithm(X_train, y_train)

            # End timer
            end_time = time.time()

            # Train the KNN classifier on the reduced dataset
            knn_reduced = KNeighborsClassifier(n_neighbors=k)
            knn_reduced.fit(X_reduced, y_reduced)

            # Evaluate the classifier
            y_pred_reduced = knn_reduced.predict(X_test)
            accuracy_reduced = accuracy_score(y_test, y_pred_reduced)

            results[algorithm.__name__].append(
                [
                    accuracy_reduced,
                    len(X_reduced),
                    100 * (1 - len(X_reduced) / len(X_train)),
                    end_time - start_time,
                ]
            )

    # Make average of the results
    for key in results.keys():
        results[key] = np.mean(results[key], axis=0)

    return results
