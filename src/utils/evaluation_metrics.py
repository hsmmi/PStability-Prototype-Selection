import random
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from config import RANDOM_SEED
from src.algorithms.stability.p_stability import PStability


# Compare different prororype selection with n-fold cross validation
def compare_prototype_selection(
    X: np.ndarray,
    y: np.ndarray,
    algorithms: dict[str, dict],
    k: int = 3,
    n_folds: int = 5,
    distance_metric: str = "euclidean",
):
    """
    Compare different prototype selection algorithms using n-fold cross validation.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    algorithms (dict): dict of algorithms containing the algorithm function and optional init_params.
    k (int): Number of neighbors to use for classification.
    n_folds (int): Number of folds for cross validation.

    Returns:
    dict: Dictionary containing the results for each algorithm.
    """
    results = {key: [] for key in algorithms.keys()}
    results["Original"] = []

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_index, test_index in tqdm(
        kf.split(X, y), total=n_folds, desc="K-Fold progress", leave=False
    ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the KNN classifier on the reduced dataset
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        knn.fit(X_train, y_train)

        # Evaluate the classifier
        accuracy_train = knn.score(X_train, y_train)
        accuracy_test = knn.score(X_test, y_test)

        p_stability = PStability(distance_metric)
        p_stability.fit(X_train, y_train)
        distorion = p_stability.run_fuzzy_distortion(1)
        objective_function = distorion + p_stability.n_misses

        results["Original"].append(
            [
                accuracy_train,
                accuracy_test,
                len(X_train),
                distorion,
                objective_function,
                0,
                0,
            ]
        )

        for key, value in tqdm(
            algorithms.copy().items(), desc="Algorithm progress", leave=False
        ):
            if key == "Original":
                continue

            algorithm, init_params = value["algorithm"], value.get("init_params", None)

            args = {
                "X": X_train,
                "y": y_train,
            }
            if init_params:
                args.update(init_params)

            # Reset random seed
            random.seed(RANDOM_SEED)

            # Start timer
            start_time = time.time()

            # Apply Algorithm
            X_reduced, y_reduced = algorithm(**args)

            # End timer
            end_time = time.time()

            # Check if the reduced dataset has at least k samples
            if len(X_reduced) < k:
                # Remove the algorithm from the results
                results.pop(key)
                algorithms.pop(key)
                continue

            # Train the KNN classifier on the reduced dataset
            knn_reduced = KNeighborsClassifier(n_neighbors=k)
            knn_reduced.fit(X_reduced, y_reduced)

            # Evaluate the classifier
            accuracy_reduced_train = knn_reduced.score(X_train, y_train)
            accuracy_reduced_test = knn_reduced.score(X_test, y_test)

            psm = PStability(distance_metric)
            # psm.fit(X_reduced, y_reduced)

            psm.fit(X_train, y_train)

            # Remove prototypes
            removed_indices = []
            for idx in range(psm.X.shape[0]):
                if np.any(np.all(psm.X[idx] == X_reduced, axis=1)):
                    continue
                removed_indices.append(idx)
            for idx in removed_indices:
                psm.remove_point(idx, update_nearest_enemy=True)

            distorion_reduced = psm.run_fuzzy_distortion(1)
            objective_function_reduced = distorion_reduced + psm.n_misses

            results[key].append(
                [
                    accuracy_reduced_train,
                    accuracy_reduced_test,
                    len(X_reduced),
                    distorion_reduced,
                    objective_function_reduced,
                    (1 - len(X_reduced) / len(X_train)),
                    end_time - start_time,
                ]
            )

    # Make average of the results
    for key in results.keys():
        results[key] = np.mean(results[key], axis=0)

    return results
