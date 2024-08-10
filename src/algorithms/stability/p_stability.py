import numpy as np
from tqdm import tqdm
from src.algorithms.stability.my_knn import KNN
from config.log import get_logger

logger = get_logger("mylogger")


class PStability(KNN):
    def __init__(self, metric: str = "euclidean"):
        """
        Initialize PStability with the given metric.

        Parameters
        ----------
        metric : str
            Distance metric to use.
        """
        super().__init__(metric)
        self.mask: np.ndarray = None
        self.classify_correct: np.ndarray = None
        self.p: list[int] = None
        self.max_misses: list[int] = None

    def _calculate_stability(self) -> int:
        """
        Check how many samples that classified correctly at first will misclassify
        after removing p points using the mask.

        Returns
        -------
        int
            The number of samples that were correctly classified initially but are misclassified
            after removing points.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PStability":
        """
        Fit the model using the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.

        Returns
        -------
        self
            Fitted instance of the algorithm.
        """
        super().fit(X, y)
        self.classify_correct = np.array(
            [self._classify(i) == y[i] for i in range(self.n_samples)]
        )
        # Sort instances by the number of friends they have until the nearest enemy in descending order
        self.nearest_enemy_sorted_index = self._sort_by_nearest_enemy()

        return self

    def run_misses(self, p: list[int]) -> list[int]:
        """
        Run the stability check for the given values of p cheking all combinations.

        Parameters
        ----------
        p : list[int]
            List of values of p to check as the number of points to remove.

        Returns
        -------
        list[int]
            List of maximum misclassifications found for each p value.
        """
        return self._run(p, self._check_combinations)

    def _find_p(self, miss: int, start_index: int = 0) -> int:
        """
        Find the minimum p value that results in at most `miss` misclassifications.
        In each iteration, the algorithm assume that a point is the point that gonna be misclassified
        and check the for minimum p that not gonna misclassify more than miss points.

        Parameters
        ----------
        miss : int
            The maximum number of allowable misclassifications.
        start_index : int, optional
            The starting index for the search, by default 0.

        Returns
        -------
        int
            The minimum p value that results in at most `miss` misclassifications.
        """
        if miss == 0:
            for idx in self.nearest_enemy_sorted_index:
                if idx >= start_index and self.mask[idx] and self.classify_correct[idx]:
                    return self._number_of_friends_until_nearest_enemy(idx) - 1
        max_p = self.n_samples + 1
        for idx in range(start_index, self.n_samples):
            if self.mask[idx] and self.classify_correct[idx]:
                changes = self._remove_nearest_neighbours(idx)
                missed = self._calculate_stability()
                if missed <= miss:
                    res_max_p = self._find_p(miss - missed, idx + 1)
                    if res_max_p != -1:
                        max_p = min(max_p, res_max_p + len(changes["neighbours"]))

                self._put_back_nearest_neighbours(changes)
        if max_p == self.n_samples + 1:
            return 0
        return max_p

    def run_max_p(self, misses: list[int]) -> list[int]:
        """
        Find the maximum p value that results in no more than the given number of misclassifications.

        Parameters
        ----------
        misses : list[int]
            List of maximum allowable misclassifications for each p value.

        Returns
        -------
        list[int]
            List of maximum p values found for each number of allowable misclassifications.
        """
        return self._run(misses, self._find_p)

    def run_relaxed_misses(self, p: list[int]) -> list[int]:
        """
        Run the relaxed stability check for the given values of p.

        Parameters
        ----------
        p : list[int]
            List of values of p to check as the number of points to remove.

        Returns
        -------
        list[int]
            List of maximum misclassifications found for each p value.
        """
        return self._run(p, self._relaxed_check)

    def _run(self, p: list[int], check_fn: callable) -> list[int]:
        """
        Run the stability check using the specified function for the given values of p.

        Parameters
        ----------
        p : list[int]
            List of values of p to check as the number of points to remove.
        check_fn : callable
            Function to use for checking stability.

        Returns
        -------
        list[int]
            List of maximum misclassifications found for each p value.
        """
        # Check that model is fitted
        if self.n_samples is None:
            raise ValueError("Model is not fitted. Run fit method first.")
        self.mask = np.ones(self.n_samples, dtype=bool)
        ret = []
        for p_value in p:
            logger.debug(f"Checking {check_fn.__name__} for p={p_value}")
            max_misses = check_fn(p_value)
            logger.debug(f"Result: {max_misses}")
            ret.append(max_misses)

        return ret

    def _relaxed_check(self, p: int) -> int:
        """
        Perform a relaxed check by removing points and counting misclassifications.

        Parameters
        ----------
        p : int
            Number of points to remove.

        Returns
        -------
        int
            Number of misclassifications after the removal.
        """
        removed = 0
        misses = 0
        pointer = 0
        while pointer < self.n_samples:
            idx = self.nearest_enemy_sorted_index[pointer]
            if self.classify_correct[idx]:
                self.mask[idx] = False
                if removed + self._number_of_friends_until_nearest_enemy(idx) > p:
                    break
                removed += self._number_of_friends_until_nearest_enemy(idx)
                misses += 1

            pointer += 1

        return misses

    def _check_combinations(self, p: int, start_index: int = 0) -> int:
        """
        Check all combinations of p points to remove and determine maximum misclassifications.

        Parameters
        ----------
        p : int
            Number of points to remove.
        start_index : int, optional
            Starting index for combinations, by default 0.

        Returns
        -------
        int
            Maximum number of misclassifications found.
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
                changed = self._remove_point_update_neighbours(idx)
                misses = self._check_combinations(p - 1, idx + 1)
                max_misses = max(max_misses, misses)
                self._put_back_point(idx, changed)
        return max_misses
