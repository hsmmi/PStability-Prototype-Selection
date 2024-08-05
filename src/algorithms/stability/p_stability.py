import itertools
from typing import Iterator, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import time
import multiprocessing as mp
from src.algorithms.stability.combination_generator import CombinationGenerator
from src.utils.data_preprocessing import load_data
from config.log import get_logger

logger = get_logger("mylogger")


class PStability:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_neighbors=5,
        show_progress=True,
        n_jobs=None,
        batch_size=None,
    ):
        """
        Initializes the PStability object.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
            X_test (np.ndarray): Test data features.
            y_test (np.ndarray): Test data labels.
            n_neighbors (int): Number of neighbors for KNeighborsClassifier.
            show_progress (bool): Whether to show the tqdm progress bar.
            n_jobs (int): Number of parallel jobs to run. Default is None which uses all CPU cores.
            batch_size (int): Size of each batch for parallel processing. Default is None which sets it to 1000 or number of samples.
        """
        # Initialize training and test data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set KNeighborsClassifier parameters
        self.n_neighbors = n_neighbors
        self.n_samples = X_train.shape[0]

        # Determine unique classes in the training data
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)

        # Initialize KNN classifier
        self.model = KNeighborsClassifier

        # Initialize the combination generator for removing samples
        self.combination_generator = CombinationGenerator()

        # Keep track of currently selected indices
        self._selected_indices: np.ndarray = np.arange(self.n_samples)

        # Calculate and store the base accuracy
        self.base_accuracy = self._get_accuracy()

        # Set the show_progress flag
        self.show_progress = show_progress

        # Set the number of parallel jobs
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()

        # Set the batch size for parallel processing
        self.batch_size = (
            batch_size if batch_size is not None else min(1000, self.n_samples)
        )

    def _get_accuracy(self):
        """
        Train the KNN model on the selected indices and return the accuracy on the test set.

        Returns:
            float: Accuracy of the model on the test set.
        """
        knn = self.model(n_neighbors=self.n_neighbors)
        knn.fit(
            self.X_train[self._selected_indices], self.y_train[self._selected_indices]
        )
        return knn.score(self.X_test, self.y_test)

    def _check_accuracy_drop(self, selected_indices, epsilon):
        """
        Check if the accuracy drops below a specified epsilon for the given selected indices.

        Args:
            selected_indices (np.ndarray): Indices of the samples to use for training.
            epsilon (float): Allowed drop in accuracy.

        Returns:
            bool: True if accuracy drops below epsilon, False otherwise.
        """
        self._selected_indices = selected_indices
        return self._get_accuracy() < self.base_accuracy - epsilon

    def _worker_find_maximum_p(self, batch, epsilon) -> int:
        """
        Worker function to find the maximum p value.
        For each batch, check if the accuracy drops below epsilon.
        Find the first batch where the accuracy drops below epsilon.
        If return batch size means no accuracy drop.
        """
        try:
            for selected_indices in batch:
                dropped = self._check_accuracy_drop(selected_indices, epsilon)
                if dropped:
                    return -1  # Indicate that the accuracy drop was detected
            return len(batch)
        except Exception as e:
            logger.error(f"Error in worker: {e}")
            return -1

    def find_maximum_p(self, epsilon: float = 0.0, max_limit: int = None) -> int:
        """
        Find the maximum p value for which the accuracy does not drop below epsilon.

        Args:
            epsilon (float): Allowed drop in accuracy.
            max_limit (int, optional): Maximum p value to evaluate. Default is the number of samples.

        Returns:
            int: Maximum p value.
        """
        if max_limit is None:
            max_limit = self.n_samples

        terminate_flag = mp.Value("i", 0)

        def update_callback(ret, pbar, terminate_flag):
            if ret == -1:
                with terminate_flag.get_lock():
                    terminate_flag.value = 1
                return
            pbar.update(ret)

        for p in range(1, max_limit + 1):
            if self._evaluate_p_value(p, epsilon, terminate_flag, update_callback):
                return p - 1

        return max_limit

    def _evaluate_p_value(self, p, epsilon, terminate_flag, update_callback) -> bool:
        """
        Evaluate a specific p value by generating combinations and checking accuracy drop.

        Args:
            p (int): Number of samples to remove.
            epsilon (float): Allowed drop in accuracy.
            terminate_flag (mp.Value): Flag to indicate termination.
            update_callback (function): Callback function to update progress bar.

        Returns:
            bool: True if termination is needed, False otherwise.
        """
        logger.debug(f"Finding Maximum p, Evaluating p={p}")
        combination_generator = CombinationGenerator().configure(
            self.n_samples, self.n_samples - p
        )
        total_combinations = len(combination_generator)

        with mp.Pool(processes=self.n_jobs) as pool:
            with tqdm(
                total=total_combinations,
                desc=f"Finding Maximum p, Evaluating p={p}",
                leave=False,
                disable=not self.show_progress,
            ) as pbar:
                for batch in self._batch_combinations(
                    combination_generator, min(self.batch_size, 1000)
                ):
                    if terminate_flag.value == 1:
                        pool.terminate()
                        pool.join()
                        return True

                    pool.apply_async(
                        self._worker_find_maximum_p,
                        (batch, epsilon),
                        callback=lambda ret, pbar=pbar, terminate_flag=terminate_flag: update_callback(
                            ret, pbar, terminate_flag
                        ),
                    )

                pool.close()
                pool.join()
        return False

    def _compute_decrease(self, selected_indices):
        """
        Compute the decrease in accuracy for the given selected indices.

        Args:
            selected_indices (np.ndarray): Indices of the samples to use for training.

        Returns:
            float: Decrease in accuracy.
        """
        self._selected_indices = selected_indices
        accuracy = self._get_accuracy()
        return self.base_accuracy - accuracy

    def _worker_find_epsilon(self, batch, results_dict, process_id, lock) -> list:
        """
        Worker function to find the epsilon value for a batch of combinations.

        Args:
            batch (list): Batch of combinations to evaluate.
            results_dict (dict): Dictionary to store results.
            process_id (int): ID of the process.
            lock (mp.Lock): Lock for synchronizing access to the results dictionary.

        Returns:
            list: Batch of combinations.
        """
        batch_max_decrease = -1
        batch_sum_decrease = 0.0
        batch_count = 0

        for selected_indices in batch:
            decrease = self._compute_decrease(selected_indices)
            batch_sum_decrease += decrease
            batch_count += 1
            if decrease > batch_max_decrease:
                batch_max_decrease = decrease

        with lock:
            results = results_dict[process_id]
            results["sum_decrease"] += batch_sum_decrease
            results["count"] += batch_count
            if batch_max_decrease > results["max_decrease"]:
                results["max_decrease"] = batch_max_decrease
            results_dict[process_id] = results

        return batch

    def _batch_combinations(self, iterable: Iterator, batch_size: int) -> Iterator:
        """
        Batch data into tuples of length batch_size. The last batch may be shorter.

        Args:
            iterable (Iterator): An iterator over the combinations.
            batch_size (int): The size of each batch.

        Yields:
            Iterator: An iterator over the batches.
        """
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, batch_size))
            if not batch:
                break
            yield batch

    def find_epsilon(self, p: int) -> tuple[float, float]:
        """
        Find the maximum and average epsilon values for a given p value.

        Args:
            p (int): Number of samples to remove.

        Returns:
            tuple[float, float]: Maximum and average epsilon values.
        """
        combination_generator = CombinationGenerator().configure(
            self.n_samples, self.n_samples - p
        )
        total_combinations = len(combination_generator)

        manager = mp.Manager()
        results_dict = manager.dict(
            {
                i: {"max_decrease": 0, "sum_decrease": 0.0, "count": 0}
                for i in range(self.n_jobs)
            }
        )
        lock = manager.Lock()

        with mp.Pool(processes=self.n_jobs) as pool:
            with tqdm(
                total=total_combinations,
                desc=f"Finding epsilons, Evaluating p={p}",
                leave=False,
                disable=not self.show_progress,
            ) as pbar:
                for idx, batch in enumerate(
                    self._batch_combinations(combination_generator, self.batch_size)
                ):
                    process_id = idx % self.n_jobs
                    pool.apply_async(
                        self._worker_find_epsilon,
                        (batch, results_dict, process_id, lock),
                        callback=lambda ret_batch: pbar.update(len(ret_batch)),
                    )
                    logger.debug(
                        f"Batch {idx} sent to process {process_id}",
                        extra={"use_tqdm": True},
                    )
                pool.close()
                pool.join()

        # Aggregate results
        total_sum_decrease = 0
        total_count = 0
        overall_max_decrease = -1
        for res in results_dict.values():
            total_sum_decrease += res["sum_decrease"]
            total_count += res["count"]
            if res["max_decrease"] > overall_max_decrease:
                overall_max_decrease = res["max_decrease"]

        avg_decrease = total_sum_decrease / total_count if total_count > 0 else 0
        max_decrease = overall_max_decrease

        return max_decrease, avg_decrease


class PStabilityResults:
    def __init__(
        self,
        max_p: Optional[int],
        max_epsilon: Optional[list[float]],
        avg_epsilon: Optional[list[float]],
        time: Optional[float] = None,
    ):
        """
        Initialize the PStabilityResults object.

        Args:
            max_p (int, optional): Maximum p value found.
            max_epsilon (list[float], optional): List of maximum epsilon values for each p.
            avg_epsilon (list[float], optional): List of average epsilon values for each p.
            time (float, optional): Total time taken.
        """
        self.max_p = max_p
        self.max_epsilon = max_epsilon
        self.avg_epsilon = avg_epsilon
        self.time = time

    def __str__(self):
        return f"PStabilityResults(max_p={self.max_p}, max_epsilon={self.max_epsilon}, avg_epsilon={self.avg_epsilon}, time={self.time})"

    def get_max_epsilon_percentage(self) -> Optional[list[str]]:
        """
        Get the maximum epsilon values as percentages.

        Returns:
            list[str]: List of maximum epsilon values as percentages.
        """
        if self.max_epsilon:
            return [f"{epsilon:.2%}" for epsilon in self.max_epsilon]
        return None

    def get_avg_epsilon_percentage(self) -> Optional[list[str]]:
        """
        Get the average epsilon values as percentages.

        Returns:
            list[str]: List of average epsilon values as percentages.
        """
        if self.avg_epsilon:
            return [f"{epsilon:.2%}" for epsilon in self.avg_epsilon]
        return None


def run_p_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_neighbors: int = 5,
    find_max_p: bool = False,
    find_epsilon: Optional[list[int]] = None,
    show_progress: bool = False,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> PStabilityResults:
    """
    Run the PStability algorithm on the given dataset using k-fold cross-validation.
    The algorithm finds the maximum p value and epsilon values for each fold.

    Parameters:
    X (np.ndarray): Feature matrix of the dataset.
    y (np.ndarray): Labels of the dataset.
    n_folds (int): Number of folds for cross-validation. Default is 5.
    n_neighbors (int): Number of neighbors for k-NN. Default is 5.
    find_max_p (bool): Whether to find the maximum p value. Default is False.
    find_epsilon (list[int], optional): List of p values to find the epsilon value. Default is None.
    show_progress (bool): Whether to use tqdm for progress tracking. Default is False.
    n_jobs (int): Number of parallel jobs to run. Default is None which uses all CPU cores.
    batch_size (int): Size of each batch for parallel processing. Default is None which sets it to 1000 or number of samples.

    Returns:
    PStabilityResults: Object containing max_p, max_epsilon, and avg_epsilon values, along with the total time taken.
    """
    if find_epsilon is None:
        find_epsilon = []

    if not find_max_p and not find_epsilon:
        raise ValueError(
            "At least one of find_max_p or find_epsilon must be provided and valid."
        )

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    max_p = []
    max_epsilon = [[] for _ in find_epsilon]
    avg_epsilon = [[] for _ in find_epsilon]
    time_taken = []

    for train_index, test_index in tqdm(
        kf.split(X, y),
        total=n_folds,
        desc="K-Fold progress",
        leave=False,
        disable=not show_progress,
    ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        p_stability = PStability(
            X_train,
            y_train,
            X_test,
            y_test,
            n_neighbors=n_neighbors,
            show_progress=show_progress,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

        start_time = time.time()
        if find_max_p:
            max_p_value = p_stability.find_maximum_p()
            max_p.append(max_p_value)
            logger.debug(
                f"Fold p={max_p_value}: max_p={max_p_value}", extra={"use_tqdm": True}
            )

        for i, p in enumerate(find_epsilon):
            tmp_max, tmp_avg = p_stability.find_epsilon(p)
            max_epsilon[i].append(tmp_max)
            avg_epsilon[i].append(tmp_avg)
            logger.debug(
                f"Fold p={p}: max_epsilon={tmp_max}, avg_epsilon={tmp_avg}",
                extra={"use_tqdm": True},
            )
        time_taken.append(time.time() - start_time)

    max_p_value = np.min(max_p) if find_max_p else None
    max_epsilon_values = np.max(max_epsilon, axis=1) if find_epsilon else None
    avg_epsilon_values = np.mean(avg_epsilon, axis=1) if find_epsilon else None
    sum_time = np.sum(time_taken)

    return PStabilityResults(
        max_p=max_p_value,
        max_epsilon=max_epsilon_values,
        avg_epsilon=avg_epsilon_values,
        time=sum_time,
    )
