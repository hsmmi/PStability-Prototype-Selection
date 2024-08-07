import numpy as np
from tqdm import tqdm
from src.algorithms.stability.my_knn import KNN


class PStability(KNN):
    def __init__(self, metric: str = "euclidean"):
        """
        Initialize PStability with the given metric.

        Parameters:
        metric (str): Distance metric to use.
        """
        super().__init__(metric)
        self.mask: np.ndarray = None
        self.classify_correct: np.ndarray = None
        self.p: list[int] = None
        self.max_misses: list[int] = None

    def _classify(self, idx: int) -> int:
        """
        Classify an instance.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: The predicted class.
        """
        nearest_neighbor = self.nearest_neighbour(idx)
        return self.y[nearest_neighbor]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PStability":
        """
        Fit the model using the training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        self: Fitted instance of the algorithm.
        """
        super().fit(X, y)
        self.classify_correct = np.array(
            [self._classify(i) == y[i] for i in range(self.n_samples)]
        )
        self.nearest_enemy_sorted_index = np.argsort(self.nearest_enemy_pointer)

        return self

    def run(self, p: list[int]) -> list[int]:
        """
        Run the stability check for the given values of p.

        Parameters:
        p (list[int]): list of values of p to check as the number of points to remove.

        Returns:
        list[int]: list of maximum misclassifications found for each p value.
        """
        return self._run(p, self._check_combinations)

    def relaxed_run(self, p: list[int]) -> list[int]:
        """
        Run the relaxed stability check for the given values of p.

        Parameters:
        p (list[int]): list of values of p to check as the number of points to remove.

        Returns:
        list[int]: list of maximum misclassifications found for each p value.
        """
        return self._run(p, self._relaxed_check)

    def _run(self, p: list[int], check_fn: callable) -> list[int]:
        """
        Run the stability check using the specified function for the given values of p.

        Parameters:
        p (list[int]): list of values of p to check as the number of points to remove.
        check_fn (callable): Function to use for checking stability.

        Returns:
        list[int]: list of maximum misclassifications found for each p value.
        """
        self.mask = np.ones(self.n_samples, dtype=bool)
        ret = []
        for p_value in p:
            logger.info(f"Checking stability for p={p_value}")
            max_misses = check_fn(p_value)
            logger.info(f"Maximum misclassifications: {max_misses}")
            ret.append(max_misses)

        return ret

    def _relaxed_check(self, p: int) -> int:
        """
        Perform a relaxed check by removing points and counting misclassifications.

        Parameters:
        p (int): Number of points to remove.

        Returns:
        int: Number of misclassifications.
        """
        removed = 0
        misses = 0
        pointer = 0
        while pointer < self.n_samples:
            idx = self.nearest_enemy_sorted_index[pointer]
            if self.classify_correct[idx]:
                self.mask[idx] = False
                if removed + self.nearest_enemy_pointer[idx] > p:
                    break
                removed += self.nearest_enemy_pointer[idx]
                misses += 1

            pointer += 1

        return misses

    def _remove_point(self, idx: int) -> np.ndarray:
        """
        Remove a point from the mask and update the nearest pointer for the neighbours.

        Parameters:
        idx (int): Index of the point to remove.

        Returns:
        np.ndarray: Array of indices of the samples which had their nearest pointer updated.
        """
        self.mask[idx] = False
        changed = np.nonzero(self.nearest_neighbours[:, self.nearest_pointer] == idx)[0]
        self.nearest_pointer[changed] += 1
        return changed

    def _put_back_point(self, idx: int, changed: np.ndarray) -> None:
        """
        Put back a point to the mask and update the nearest pointer
        for the neighbours which had been changed.

        Parameters:
        idx (int): Index of the point to put back.
        changed (np.ndarray): Array of indices that had their nearest pointer updated.
        """
        self.mask[idx] = True
        self.nearest_pointer[changed] -= 1

    def _calculate_stability(self) -> int:
        """
        Check how many samples that classified correctly at first will misclassify
        after removing p points using the mask.

        Returns:
        int: The number of samples that classify correctly at first but misclassify
        after removing p points.
        """
        misclassifications = 0
        for i in range(self.n_samples):
            if (
                self.mask[i]
                and self.classify_correct[i]
                and self.y[i] != self._classify(i)
            ):
                misclassifications += 1
        return misclassifications

    def _check_combinations(self, p: int, start_index: int = 0) -> int:
        """
        Check all the combinations of p points to remove.

        Parameters:
        p (int): Number of points to remove.
        start_index (int): Starting index for combinations.

        Returns:
        int: Maximum number of misclassifications found.
        """
        if p == 0:
            return self._calculate_stability()
        if start_index >= self.n_samples:
            return 0
        max_misses = 0
        for idx in tqdm(
            range(start_index, self.n_samples),
            desc=f"Checking p={p}",
            leave=False,
            disable=p < 3,
        ):
            if self.mask[idx]:
                changed = self._remove_point(idx)
                misses = self._check_combinations(p - 1, idx + 1)
                max_misses = max(max_misses, misses)
                self._put_back_point(idx, changed)
        return max_misses
