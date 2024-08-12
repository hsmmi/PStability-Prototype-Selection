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

        Parameters
        ----------
        metric : str
            Distance metric to use.
        """
        super().__init__(metric)
        self.sorted_fuzzy_missclassification_score: list[Tuple[int, float]] = None

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

    def find_fuzzy_missclassification_score(self) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each instance.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance.
        """
        friends = self.compute_all_firends()

        scores = np.zeros(self.n_samples)
        for idx in range(self.n_samples):
            if self.mask[idx]:
                # Calculate the score for the instance
                # If the instance is in the friends of the other instances with friend size L,
                # then it gets 1/L score from that instance.
                for idx2 in range(self.n_samples):
                    if len(friends[idx2]) == 0:
                        continue
                    if idx in friends[idx2]:
                        scores[idx] += 1 / len(friends[idx2])
        scores = [(idx, score) for idx, score in enumerate(scores)]
        return scores

    def find_max_fuzzy_missclassification_score(self) -> Tuple[int, float]:
        """
        Find the instance with the maximum fuzzy misclassification score.

        Returns
        -------
        Tuple[int, float]
            The (index, score) pair for the instance with the
            maximum fuzzy misclassification score.
        """
        scores = self.find_fuzzy_missclassification_score()
        max_idx, max_score = -1, -1
        for idx, score in scores:
            if score > max_score:
                max_idx, max_score = idx, score
        return max_idx, max_score

    def find_sorted_fuzzy_missclassification_score(self) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each
        instance and sort them in descending order.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance sorted in descending order of score.
        """
        scores = self.find_fuzzy_missclassification_score()
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def reset(self) -> None:
        """
        Reset the model to the initial state.
        """
        super().reset()

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
            The lower bound of the maximum p value found for the given number of allowable misclassifications.
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
            List of lower bound p values found for each number of allowable
            misclassifications in range[0, max_miss].
        """
        self._set_number_of_friends_sorted_index()
        return self._run(range(max_miss + 1), self._find_lower_bound_p)

    def _find_upper_bound_p(self, miss: int) -> int:
        """
        Find the upper bound of the maximum p value that results in no more
        than the given number of misclassifications in any combination of removing p points.
        Assume that the enemies of each instance is UNIQELY among the rest of the instances.

        Parameters
        ----------
        miss : int
            The maximum number of allowable misclassifications.

        Returns
        -------
        int
            The upper bound of the maximum p value found for the given number
            of allowable misclassifications.
        """
        upper_bound = 0
        for i in range(miss + 1):
            upper_bound += self.number_of_friends_sorted_index[i][1]
        return upper_bound - 1

    def run_upper_bound_p(self, max_miss: int) -> list[int]:
        """
        Find the upper bound of the maximum p value that results in no more
        than the given number of misclassifications.
        Parameters
        ----------
        misses : list[int]
            List of maximum allowable misclassifications for each p value.

        Returns
        -------
        list[int]
            List of upper bound p values found for each number of allowable
            misclassifications in range[0, max_miss].
        """
        self._set_number_of_friends_sorted_index()
        return self._run(range(max_miss + 1), self._find_upper_bound_p)

    def _find_fuzzy_missclassification(self, p: int) -> float:
        """
        Find the maximum fuzzy misclassifications for the given p value.
        In this method, the missclsification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the p points with the highest score.
        The value is a upper bound of the misclassifications.

        Parameters
        ----------
        p : int
            Number of points to remove.

        Returns
        -------
        float
            Maximum fuzzy misclassifications found for the given p value.
        """
        # Fuzzy misclassification score has been calculated in the fit method

        # Remove the p instances with the highest score
        fuzzy_misses = 0
        for idx in range(p):
            fuzzy_misses += self.sorted_fuzzy_missclassification_score[idx][1]
        return fuzzy_misses

    def run_fuzzy_missclassification(self, max_p: int) -> list[Tuple[int, float]]:
        """
        Find the maximum fuzzy misclassifications for each p value in the range [0, max_p].
        In this method, the missclsification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the p points with the highest score.
        The value is a much better upper bound of the misclassifications.

        Parameters
        ----------
        max_p : int

        Returns
        -------
        list[Tuple[int, float]]
            List of maximum fuzzy misclassifications found for each p value in range[0, max_p].

        """
        self.reset()
        self.sorted_fuzzy_missclassification_score = (
            self.find_sorted_fuzzy_missclassification_score()
        )
        return self._run(range(max_p + 1), self._find_fuzzy_missclassification)

    def _calculate_fuzzy_stability(self) -> float:
        """
        Calculate the fuzzy stability of the model.

        Returns
        -------
        float
            The fuzzy stability of the model.
        """
        removed = np.where(self.mask == False)[0]
        fuzzy_misses = 0
        # Calculate the fuzzy missclassification score for each sample
        scores = self.find_fuzzy_missclassification_score()
