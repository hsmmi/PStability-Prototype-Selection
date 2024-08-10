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
        self.nearest_enemy_sorted_index: list[int] = None

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

    def convert_misses_to_p_list(self, list_max_p: list) -> list:
        """
        Get the list which index is the number of misclassifications and the value is the maximum p value that
        results in that no more than that number of misclassifications. The returned the assoicated misclassifications
        for each p value.

        ! The index of list_max_p list should be the maximum number of misclassifications.

        Parameters
        ----------
        list_max_p : list
            List of maximum p values found for each number of allowable misclassifications.

        Returns
        -------
        list
            List of maximum misclassifications found for each p value.
        """
        list_max_miss = [0] * (max(list_max_p) + 1)
        pointer_list_max_p, pointer_list_max_miss = 0, 0
        while pointer_list_max_p < len(list_max_p):
            while pointer_list_max_miss <= list_max_p[pointer_list_max_p]:
                list_max_miss[pointer_list_max_miss] = pointer_list_max_p
                pointer_list_max_miss += 1
            pointer_list_max_p += 1
        return list_max_miss

    def find_instance_with_min_friends(self, start_index: int = 0) -> int:
        """
        Find the instance with the minimum number of friends until the nearest enemy.
        The instance must be classified correctly.

        Parameters
        ----------
        start_index : int, optional
            The starting index for the search, by default 0.

        Returns
        -------
        int
            The index of the instance with the minimum number of friends until the nearest enemy.
        """
        min_friends, min_idx = self.n_samples + 1, -1
        for idx in range(start_index, self.n_samples):
            if self.mask[idx] and self.classify_correct[idx]:
                friends = self._number_of_friends_until_nearest_enemy(idx)
                if friends < min_friends:
                    min_friends, min_idx = friends, idx
        return min_idx

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
        self.number_of_friends_sorted_index = self._sort_by_number_of_friends()

        return self

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

    def _find_exact_miss(self, p: int, start_index: int = 0) -> int:
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
                misses = self._find_exact_miss(p - 1, idx + 1)
                max_misses = max(max_misses, misses)
                self._put_back_point(idx, changed)
        return max_misses

    def run_exact_miss(self, max_p: int) -> list[int]:
        """
        Run the exact maximum misclassifications check for each p value in the range [0, max_p].
        By checking all possible combinations of removing p points.

        Parameters
        ----------
        max_p : int
            Maximum number of points to remove.

        Returns
        -------
        list[int]
            List of maximum misclassifications found for each p value in range[0, max_p].
        """
        return self._run(range(max_p + 1), self._find_exact_miss)

    def _find_exact_p(self, miss: int, start_index: int = 0) -> int:
        """
        Find the maximum p value that results no more than the given number of misclassifications
        in any combination of removing p points.
        By checking all possible combinations instances to be misclassified.

        Parameters
        ----------
        miss : int
            The maximum number of allowable misclassifications.
        start_index : int, optional
            The starting index for the search, by default 0.

        Returns
        -------
        int
            The maximum p value found for the given number of allowable misclassifications.
            returns -1 if no such p value found.
        """
        if miss == 0:
            idx_min_friends = self.find_instance_with_min_friends(start_index)
            if idx_min_friends == -1:
                return 0
            return self._number_of_friends_until_nearest_enemy(idx_min_friends) - 1
        max_p = self.n_samples + 1
        # Assume that each instance is misclassified by removing all its friends.
        for idx in range(start_index, self.n_samples):
            if self.mask[idx] and self.classify_correct[idx]:
                changes = self._remove_nearest_neighbours(idx)
                missed = self._calculate_stability()
                if missed <= miss:
                    res_max_p = self._find_exact_p(miss - missed, idx + 1)
                    if res_max_p != -1:
                        max_p = min(max_p, res_max_p + len(changes["neighbours"]))

                self._put_back_nearest_neighbours(changes)
        if max_p == self.n_samples + 1:
            return -1
        return max_p

    def run_exact_p(self, max_miss: int) -> list[int]:
        """
        Find the exact maximum p value that results in no more than the given number of misclassifications

        Parameters
        ----------
        max_miss : int
            Maximum allowable misclassifications.

        Returns
        -------
        list[int]
            List of maximum p values found for each number of allowable misclassifications in range[0, max_miss].
        """
        return self._run(range(max_miss + 1), self._find_exact_p)

    def _find_lower_bound_p(self, miss: int) -> int:
        """
        Find the lower bound of the maximum p value that results in no more
        than the given number of misclassifications in any combination of removing p points.
        Assume that the friends of each instance is completely on the rest of the
        instances after that in increasing friends size order in each class.
        And check for each class because to have a lower bound, removed instances should be
        from the same class.

        Parameters
        ----------
        miss : int
            The maximum number of allowable misclassifications.

        Returns
        -------
        int
            The maximum p value found for the given number of allowable misclassifications.

            # TODO: returns -1 if no such p value found that exactly results in the given number of misclassifications.
        """
        if miss == 0:
            return self.number_of_friends_sorted_index[0][1] - 1
        max_p = self.n_samples + 1
        for class_label in self.classes:
            # Select instances of the class in the order of increasing number of friends
            number_of_friends_sorted_index_class = [
                (idx, n_friends)
                for (idx, n_friends) in self.number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            # Check if exact miss is possible by checking the next instance in the order to
            # have more friends than the current instance.
            if (
                number_of_friends_sorted_index_class[miss - 1][1]
                == number_of_friends_sorted_index_class[miss][1]
            ):
                continue
            else:
                max_p = min(
                    max_p,
                    number_of_friends_sorted_index_class[miss][1] - 1,
                )
        if max_p == self.n_samples + 1:
            return -1
        return max_p

    def run_lower_bound_p(self, max_miss: int) -> list[int]:
        """
        Find the lower bound of the maximum p value that results in no more
        than the given number of misclassifications.
        Parameters
        ----------
        misses : list[int]
            List of maximum allowable misclassifications for each p value.

        Returns
        -------
        list[int]
            List of maximum p values found for each number of allowable
            misclassifications in range[0, max_miss].
        """
        return self._run(range(max_miss + 1), self._find_lower_bound_p)
