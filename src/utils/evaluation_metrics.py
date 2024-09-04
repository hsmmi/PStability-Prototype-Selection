import random
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from config import RANDOM_SEED
from src.algorithms.stability.p_stability import PStability
from src.utils.result import DatasetResult, AlgorithmResult, RunResult


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
    algorithms["Original"] = {"algorithm": None}

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    dataset_res = DatasetResult("Dataset", n_folds, k)
    for algorithm in algorithms.keys():
        dataset_res.add_result(AlgorithmResult(algorithm))

    for train_index, test_index in tqdm(
        kf.split(X, y), total=n_folds, desc="K-Fold progress", leave=False
    ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for algorithm_name, value in tqdm(
            algorithms.copy().items(), desc="Algorithm progress", leave=False
        ):
            if algorithm_name == "Original":
                X_reduced, y_reduced = X_train, y_train
                start_time, end_time = 0, 0
            else:
                algorithm, init_params = value["algorithm"], value.get(
                    "init_params", None
                )
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
                algorithms.pop(algorithm_name)
                dataset_res.results.pop(algorithm_name)
                continue

            # Train the KNN classifier on the reduced dataset
            knn_reduced = KNeighborsClassifier(n_neighbors=k)
            knn_reduced.fit(X_reduced, y_reduced)

            # Evaluate the classifier
            accuracy_reduced = knn_reduced.score(X_test, y_test)

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

            run_result = RunResult(
                size=len(X_reduced),
                accuracy=accuracy_reduced,
                reduction=1 - len(X_reduced) / len(X_train),
                distortion=distorion_reduced,
                objective_function=objective_function_reduced,
                time=end_time - start_time,
            )

            dataset_res.results[algorithm_name].add_result(run_result)

    return dataset_res
