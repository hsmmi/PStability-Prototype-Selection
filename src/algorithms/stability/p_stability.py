from typing import Tuple
import numpy as np
from tqdm import tqdm
from src.algorithms.stability.my_knn import KNN
from config.log import get_logger

logger = get_logger("mylogger")


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
        Classify an instance with 1-NN.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: The predicted class.
        """
        nearest_neighbor_idx, _ = self.nearest_neighbour(idx)
        return self.y[nearest_neighbor_idx]

    def _number_of_friends_until_nearest_enemy(self, idx: int) -> int:
        """
        Calculate the number of friends an instance has until the nearest enemy.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Number of friends until the nearest enemy.
        """
        return max(
            0,
            (self.nearest_enemy_index(idx) - self.nearest_enemies_pointer[idx])
            - (self.nearest_friend_index(idx) - self.nearest_friends_pointer[idx]),
        )

    def _sort_by_nearest_enemy(self) -> np.ndarray:
        """
        Sort instances by the number of friends they have until the nearest enemy.

        Returns:
        np.ndarray: Indices of the instances sorted by the number of friends they have.
        """
        return np.argsort(
            [
                self._number_of_friends_until_nearest_enemy(idx)
                for idx in range(self.n_samples)
            ]
        )

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
        # Sort instances by the number of friends they have until the nearest enemy in descending order
        self.nearest_enemy_sorted_index = self._sort_by_nearest_enemy()

        return self

    def run_misses(self, p: list[int]) -> list[int]:
        """
        Run the stability check for the given values of p using all combinations.

        Parameters:
        p (list[int]): list of values of p to check as the number of points to remove.

        Returns:
        list[int]: list of maximum misclassifications found for each p value.
        """
        return self._run(p, self._check_combinations)

    def _remove_nearest_neighbours(self, idx: int) -> dict[int, list[int]]:
        """
        Remove the nearest neighbours of a point until nearest enemy is reached.

        Parameters:
        idx (int): Index of the point.

        Returns:
        dict[int, list[int]]: Dictionary of indices that had their nearest pointers updated containing
        the indices of the nearest neighbours and the indices of the nearest enemies.
        """
        nearest_friends_pointer = self.nearest_friend_index(idx)
        nearest_neighbours_enemy_pointer = self.nearest_enemy_index(idx)
        neighbour_idx = self.nearest_neighbours[idx][
            nearest_friends_pointer:nearest_neighbours_enemy_pointer
        ]
        changes = {}
        changes["neighbours"] = neighbour_idx
        self.mask[neighbour_idx] = False
        changes["update_nearest_friends"] = {}
        changes["update_nearest_enemies"] = {}
        for idx2 in range(self.n_samples):
            while self.nearest_friend(idx2) in neighbour_idx:
                changes["update_nearest_friends"][idx2] = (
                    changes["update_nearest_friends"].get(idx2, 0) + 1
                )
                self.nearest_friends_pointer[idx2] += 1
            while self.nearest_enemy(idx2) in neighbour_idx:
                changes["update_nearest_enemies"][idx2] = (
                    changes["update_nearest_enemies"].get(idx2, 0) + 1
                )
                self.nearest_enemies_pointer[idx2] += 1
        return changes

    def _put_back_nearest_neighbours(self, changed_list: dict[int, list[int]]) -> None:
        """
        Put back the nearest neighbours of a point that were removed.

        Parameters:
        changed_list (list[int]): List of indices that had their nearest pointers updated.
        """
        neighbours_idx = changed_list["neighbours"]
        self.mask[neighbours_idx] = True
        for idx2, count in changed_list["update_nearest_friends"].items():
            self.nearest_friends_pointer[idx2] -= count
        for idx2, count in changed_list["update_nearest_enemies"].items():
            self.nearest_enemies_pointer[idx2] -= count

    def _find_p(self, miss: int, start_index: int = 0) -> int:
        """
        Find the minimum p value that will result in at most miss misclassifications.
        In each iteration, the algorithm assume that a point is the point that gonna be misclassified
        and check the for minimum p that not gonna misclassify more than miss points.

        Parameters:
        miss (int): The (maximum) number of misclassifications.

        Returns:
        int: Maximum number of p that not gonna misclassify more than miss points.
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
        Find the maximum p value that will result no more than the given number of misclassifications.


        Parameters:
        p (list[int]): list of values of p to check as the number of points to remove.

        Returns:
        list[int]: list of maximum misclassifications found for each p value.
        """
        return self._run(misses, self._find_p)

    def run_relaxed_misses(self, p: list[int]) -> list[int]:
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
                if removed + self._number_of_friends_until_nearest_enemy(idx) > p:
                    break
                removed += self._number_of_friends_until_nearest_enemy(idx)
                misses += 1

            pointer += 1

        return misses

    def _remove_point_update_neighbours(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove a point from the mask and update the nearest pointer for the neighbours.

        Parameters:
        idx (int): Index of the point to remove.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of indices that had their nearest pointers updated.
        The first array contains the indices of the nearest neighbours and the second array
        contains the indices of the nearest enemies.
        """
        self.mask[idx] = False
        changed_nearest_neighbor = np.nonzero(
            [self.nearest_friend(idx2) == idx for idx2 in range(self.n_samples)]
        )[0]
        self.nearest_friends_pointer[changed_nearest_neighbor] += 1

        changed_nearest_enemy = np.nonzero(
            [self.nearest_enemy(idx2) == idx for idx2 in range(self.n_samples)]
        )[0]
        self.nearest_enemies_pointer[changed_nearest_enemy] += 1

        return [changed_nearest_neighbor, changed_nearest_enemy]

    def _put_back_point(self, idx: int, changed: np.ndarray) -> None:
        """
        Put back a point to the mask and update the nearest pointer
        for the neighbours which had been changed.

        Parameters:
        idx (int): Index of the point to put back.
        changed (np.ndarray): Array of indices that had their nearest pointers updated.
        """
        self.mask[idx] = True
        changed_nearest_neighbor, changed_nearest_enemy = changed
        self.nearest_friends_pointer[changed_nearest_neighbor] -= 1
        self.nearest_enemies_pointer[changed_nearest_enemy] -= 1

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
                changed = self._remove_point_update_neighbours(idx)
                misses = self._check_combinations(p - 1, idx + 1)
                max_misses = max(max_misses, misses)
                self._put_back_point(idx, changed)
        return max_misses
