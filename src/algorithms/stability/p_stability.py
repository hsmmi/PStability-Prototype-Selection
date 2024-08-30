from typing import Tuple
import numpy as np
from tqdm import tqdm
from src.algorithms.stability.my_knn import KNN
from config.log import get_logger
from copy import deepcopy

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
        self.initial_classify_correct = None
        self.sorted_fuzzy_score_teain: list[Tuple[int, float]] = None

    def calculate_stability(self) -> int:
        """
        Check how many samples that classified correctly at first will misclassify
        now in the current state.

        Assumption1: most deviation comes from the errors not from the corrections.

        Assumption2: we ignorethe samples which classified incorrectly at first.

        Returns
        -------
        int
            The number of samples that were correctly classified initially but are misclassified now.
        """
        # Sum initial classification correct and current classification incorrect
        return sum(self.initial_classify_correct & ~self.classify_correct)

    def find_fuzzy_score_teain(self) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each training instance. Which is based on
        friends of test instances on the training instances.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance.
        """
        friends = self.compute_all_friends()
        train_indices = np.where(self.mask_train)[0]
        scores = []
        # find the score for each training instance
        for idx in train_indices:
            score = 0
            # Calculate the score for the instance
            # If the instance is in the friends of the other instances with friend size L,
            # then it gets 1/L score from that instance.
            for idx2 in range(self.n_samples):
                if len(friends[idx2]) == 0:
                    continue
                if idx in friends[idx2]:
                    score += 1 / len(friends[idx2])
            scores.append((idx, score))
        return scores

    def find_max_fuzzy_score_teain(self) -> Tuple[int, float]:
        """
        Find the instance with the maximum fuzzy misclassification score.

        Returns
        -------
        Tuple[int, float]
            The (index, score) pair for the instance with the
            maximum fuzzy misclassification score.
        """
        scores = self.find_fuzzy_score_teain()
        max_idx, max_score = -1, -1
        for idx, score in scores:
            if score > max_score:
                max_idx, max_score = idx, score
        return max_idx, max_score

    def find_sorted_fuzzy_score_teain(self) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each
        training instance and sort them in descending order.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance sorted in descending order of score.
        """
        scores = self.find_fuzzy_score_teain()
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
        self.initial_classify_correct = deepcopy(self.classify_correct)
        return self

    def _run(self, list_value: list[int] | int, check_fn: callable) -> list[int] | int:
        """
        Run the given check function for each value in the list or a single value.

        Parameters
        ----------
        list_value : list[int] or int
            List of values to check or a single integer value.
        check_fn : callable
            Function to use for checking.

        Returns
        -------
        list[int] or int
            List of results for each value in list_value if it's a list,
            or a single result if list_value is an integer.
        """
        # Check that model is fitted
        if self.n_samples is None:
            raise ValueError("Model is not fitted. Run fit method first.")

        single_value = False
        if isinstance(list_value, int):
            list_value = [list_value]
            single_value = True

        ret = []
        for value in list_value:
            logger.debug(f"Checking {check_fn.__name__} for p={value}")
            res = check_fn(value)
            logger.debug(f"Result: {res}")
            ret.append(res)

        if single_value:
            return ret[0]

        return ret

    def find_exact_p(self, stability: int, start_index: int = 0) -> int:
        """
        Find the maximum p value that results no more than the given number of misclassifications
        in any combination of removing p points.
        By checking all possible combinations instances to be misclassified.

        Parameters
        ----------
        stability : int
            The maximum number of allowable misclassifications.
        start_index : int, optional
            The starting index for the search, by default 0.

        Returns
        -------
        int
            The maximum p value found for the given number of allowable misclassifications.
            returns -1 if no such p value found.
        """
        if stability < 0:
            return -1

        if start_index >= self.n_samples:
            # can't continue return invalid value to prevent update
            return self.n_samples + 1

        if stability == 0:
            idx_min_friends, friends = self.find_instance_with_min_friends(start_index)
            if idx_min_friends == -1:
                return 0
            return len(friends) - 1

        max_p = self.n_samples + 1
        # Assume that each instance going to be misclassified
        # with removing it's friends
        for idx in tqdm(
            range(start_index, self.n_samples),
            desc=f"Checking stability={stability}",
            leave=False,
            disable=stability < 3,
        ):
            if self.initial_classify_correct[idx] and self.classify_correct[idx]:
                changes = self.remove_nearest_friends(idx, update_nearest_enemy=False)
                change_stability = len(changes["classify_incorrect"])

                # Optimization: Not gonna improve anyway
                if len(changes["friends"]) > max_p:
                    self.put_back_nearest_friends(changes)
                    continue

                res_max_p = self.find_exact_p(stability - change_stability, idx + 1)
                if res_max_p != -1:
                    max_p = min(max_p, res_max_p + len(changes["friends"]))
                else:
                    # Stability became less than zero
                    max_p = min(max_p, len(changes["friends"]) - 1)

                self.put_back_nearest_friends(changes)

        return max_p

    def run_exact_p(self, list_stability: list[int]) -> list[int]:
        """
        Find the exact maximum p value that results in no more than the given number of misclassifications

        Parameters
        ----------
        list_stability : list[int]
            Maximum allowable misclassifications.

        Returns
        -------
        list[int]
            List of maximum p values found for each number of allowable misclassifications in list_stability.
        """
        return self._run(list_stability, self.find_exact_p)

    def find_lower_bound_p(self, stability: int) -> int:
        """
        Find the lower bound of the maximum p value that results in no more
        than the given number of misclassifications in any combination of removing p points.
        Assume that the friends of each instance is completely on the rest of the
        instances after that in increasing friends size order in each class.
        And check for each class because to have a lower bound, removed instances should be
        from the same class.

        Parameters
        ----------
        stability : int
            The maximum number of allowable misclassifications.

        Returns
        -------
        int
            The lower bound of the maximum p value found for the given number of allowable misclassifications.
        """
        if stability == 0:
            return self.number_of_friends_sorted_index[0][1] - 1
        if stability >= self.n_samples - self.n_misses:
            return self.n_samples
        number_of_friends_sorted_classes = [[]]
        for class_label in self.classes:
            # Select instances of the class in the order of increasing number of friends
            number_of_friends_sorted = [0] + [
                n_friends
                for idx, n_friends in self.number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            number_of_friends_sorted_classes.append(number_of_friends_sorted)

        # DP[i][j] =  minimum p possible by missing j instance from the first i classes.
        dp = [[float("inf")] * (stability + 2) for _ in range(self.n_classes + 1)]
        dp[0][0] = 0
        for i in range(1, self.n_classes + 1):
            len_class_i = len(number_of_friends_sorted_classes[i])
            for j in range(stability + 2):
                for x in range(min(j + 1, len_class_i)):
                    # Miss x instance from class i of total j miss and j - x from forst i - 1 class
                    dp[i][j] = min(
                        dp[i][j],
                        dp[i - 1][j - x] + number_of_friends_sorted_classes[i][x],
                    )

        return (
            int(dp[self.n_classes][stability + 1] - 1)
            if dp[self.n_classes][stability + 1] != float("inf")
            else -1
        )

    def run_lower_bound_p(self, list_stability: list[int]) -> list[int]:
        """
        Find the lower bound of the maximum p value that results in no more
        than the given number of misclassifications.

        Parameters
        ----------
        list_stability : list[int]
            List of stability values to check.

        Returns
        -------
        list[int]
            List of lower bound p values found for each number of allowable
            misclassifications in list_stability.
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_stability, self.find_lower_bound_p)

    def find_better_upper_bound_p(self, stability: int) -> int:
        """
        Find the better upper bound of the maximum p value that results in no more
        than the given number of stability in any combination of removing p points.
        Assume that the friends of each instance can be among the rest of the instances.
        And we will remove it's friends from list of other instances friends.
        In each step, we greedily select the instance with the minimum number of friends
        and remove it's friends and update the friends of other instances.

        Parameters
        ----------
        stability : int
            The maximum number of allowable misclassifications.

        Returns
        -------
        int
            The better upper bound of p value found for the given stability.
        """
        if stability < 0:
            return -1
        # Greedily select the instance with the minimum number of friends
        idx_min_friends, friends = self.find_instance_with_min_friends()
        if idx_min_friends == -1:
            return 0

        if stability == 0:
            return len(friends) - 1

        changes = self.remove_nearest_friends(
            idx_min_friends, update_nearest_enemy=False
        )

        stability_changed = len(changes["classify_incorrect"])

        rest = self.find_better_upper_bound_p(stability - stability_changed)

        self.put_back_nearest_friends(changes)

        if rest == -1:
            return len(friends) - 1
        return len(friends) + rest

    def run_better_upper_bound_p(self, list_stability: int) -> list[int]:
        """
        Run the better upper bound of the maximum p value that results in no more
        than the given number of stability in any combination of removing p points.

        Parameters
        ----------
        list_stability : int
            List of stability values to check.

        Returns
        -------
        list[int]
            List of better upper bound p values found for each stability value in list_stability.
        """
        ret = self._run(list_stability, self.find_better_upper_bound_p)
        return ret

    def find_upper_bound_p(self, stability: int) -> int:
        """
        Find the upper bound of the maximum p value that results in no more
        than the given number of misclassifications in any combination of removing p points.
        Assume that the friends of each instance is UNIQELY among the rest of the instances.

        Parameters
        ----------
        stability : int
            The maximum number of allowable change in misclassifications.

        Returns
        -------
        int
            The upper bound of the maximum p value found for the given stability.
        """
        upper_bound = 0
        for i in range(stability + 1):
            upper_bound += self.number_of_friends_sorted_index[i][1]
        return upper_bound - 1

    def run_upper_bound_p(self, list_stability: list[int]) -> list[int]:
        """
        Find the upper bound of the maximum p value that results in no more
        than the given number of misclassifications.
        Parameters
        ----------
        list_stability : list[int]
            The list of stability to check for upper bound p

        Returns
        -------
        list[int]
            List of upper bound p values found for each number of allowable
            misclassifications in list_stability
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_stability, self.find_upper_bound_p)

    def find_exact_stability(self, p: int, start_index: int = 0) -> int:
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
            return 0
        if start_index >= self.n_samples:
            return 0
        max_misses = 0
        for idx in tqdm(
            range(start_index, self.n_samples),
            desc=f"Checking p={p}",
            leave=False,
            disable=p < 3,
        ):
            if self.mask_train[idx]:
                changed = self.remove_point(idx, update_nearest_enemy=False)
                stability_changed = len(changed["classify_incorrect"])
                stability = self.find_exact_stability(p - 1, idx + 1)
                max_misses = max(max_misses, stability_changed + stability)
                self.put_back_point(idx, changed)
        return max_misses

    def run_exact_stability(self, p_list: list[int]) -> list[int]:
        """
        Run the exact maximum misclassifications check for each p value in the range [0, max_p].
        By checking all possible combinations of removing p points.

        Parameters
        ----------
        p_list : list[int]
            List of p values to check.

        Returns
        -------
        list[int]
            List of maximum misclassifications found for each p value in range[0, max_p].
        """
        return self._run(p_list, self.find_exact_stability)

    def find_lower_bound_stability(self, p: int) -> int:
        """
        Find the lower bound of the stability that results of removing
        any combination of p points.
        Assume that the friends of each instance is UNIQELY among the
        rest of the instances.

        Parameters
        ----------
        p : int
            The number of points to remove.

        Returns
        -------
        int
            The lower bound of the stability found for the given p.
        """
        lower_bound = 0
        sum_friends = 0
        for i in range(self.n_samples - self.n_misses):
            sum_friends += self.number_of_friends_sorted_index[i][1]
            if sum_friends > p:
                break
            lower_bound = i + 1
        return lower_bound

    def run_lower_bound_stability(self, list_p: list[int]) -> list[int]:
        """
        Find the lower bound of stability for each p value in the list.

        Parameters
        ----------
        list_p : list[int]
            The list of stability to check for lower bound p

        Returns
        -------
        list[int]
            List of lower bound stability values found for each p value in list_p
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_p, self.find_lower_bound_stability)

    def find_better_lower_bound_stability(self, p: int) -> int:
        """
        Find the better lower bound of the stability that results of removing
        any combination of p points.
        Assume that the friends of each instance can be among the rest of the instances.
        And we will remove it's friends from list of other instances friends.
        In each step, we greedily select the instance with the minimum number of friends
        and remove it's friends and update the friends of other instances.

        Parameters
        ----------
        p : int
            The number of points to remove.

        Returns
        -------
        int
            The better lower bound of the maximum p value found for the given number
            of allowable misclassifications.
        """
        if p < 0:
            logger.warning(
                f"Value of p is negative: {p}. In find_better_lower_bound_p."
            )
            return -1
        if p == 0:
            return 0
        # Greedily select the instance with the minimum number of friends
        idx_min_friends, friends = self.find_instance_with_min_friends()

        if idx_min_friends == -1:
            return -1

        if p < len(friends):
            return 0

        changes = self.remove_nearest_friends(
            idx_min_friends, update_nearest_enemy=False
        )

        stability_changed = len(changes["classify_incorrect"])

        rest = self.find_better_lower_bound_stability(p - len(friends))

        self.put_back_nearest_friends(changes)

        if rest == -1:
            return 0
        return rest + stability_changed

    def run_better_lower_bound_stability(self, list_p: int) -> list[int]:
        """
        Run the better lower bound for stability for each p value in the list.

        Parameter
        ----------
        list_p : int
            List of p values to check.

        Returns
        -------
        list[int]
            List of better lower bound stability values found for each p value in list_p.
        """
        ret = self._run(list_p, self.find_better_lower_bound_stability)
        return ret

    def find_upper_bound_stability(self, p: int) -> int:
        """
        Find the upper bound of stability for the given p value.
        Assume that the friends of each instance is completely on the rest of the
        instances after that instance in increasing friends size order in each class.
        """
        number_of_friends_sorted_classes = [[]]
        for class_label in self.classes:
            # Select instances of the class in the order of increasing number of friends
            number_of_friends_sorted = [
                n_friends
                for idx, n_friends in self.number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            number_of_friends_sorted_classes.append(number_of_friends_sorted)

        # DP[i][j] =  Maximum stability possible by remove j instance from the first i classes.
        dp = [[float("-inf")] * (p + 1) for _ in range(self.n_classes + 1)]
        dp[0][0] = 0
        for i in range(1, self.n_classes + 1):
            for j in range(p + 1):
                class_miss = 0
                len_class_i = len(number_of_friends_sorted_classes[i])
                for x in range(j + 1):
                    while (
                        class_miss < len_class_i
                        and number_of_friends_sorted_classes[i][class_miss] <= x
                    ):
                        class_miss += 1
                    # Remove x sample from class i of j total remove
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - x] + class_miss)

        return (
            int(dp[self.n_classes][p]) if dp[self.n_classes][p] != float("-inf") else -1
        )

    def run_upper_bound_stability(self, list_p: list[int]) -> list[int]:
        """
        Find the upper bound of stability for each p value in the list.
        Stability is the maximum number of misclassifications that can
        be achieved by removing any combination of p points.


        Parameters
        ----------
        list_p : list[int]
            List of p values to check.

        Returns
        -------
        list[int]
            List of upper bound stability values found for each p value in list_p.
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_p, self.find_upper_bound_stability)

    ####################################################################################

    def find_fuzzy_stability(self, p: int) -> float:
        """
        Find the maximum fuzzy stability for the given p value.
        In this method, the misclassification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the p points with the highest score.
        The value is a upper bound of the stability.

        Parameters
        ----------
        p : int
            Number of points to remove.

        Returns
        -------
        float
            Maximum fuzzy stability score.
        """
        # Remove the p instances with the highest score
        fuzzy_score = 0
        for i in range(p):
            fuzzy_score += self.sorted_fuzzy_score_teain[i][1]
        return fuzzy_score

    def run_fuzzy_stability(self, p_list: list[int] | int) -> list[float] | float:
        """
        Find the maximum fuzzy stability of removing p points.
        In this method, the missclsification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the p points with the highest score.
        The value is a much better upper bound of the stability.

        Parameters
        ----------
        p_list : list[int] or int
            List of p values to check or a single integer value.

        Returns
        -------
        list[float] or float
            List of maximum fuzzy stability score for each p value in list_p
            or a single result if list_p is an integer
        """
        self.sorted_fuzzy_score_teain = self.find_sorted_fuzzy_score_teain()
        return self._run(p_list, self.find_fuzzy_stability)

    def find_crisped_stability(self, p: int) -> int:
        """
        Find the maximum crisped stability for the given p value.
        In this method, the misclassification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the p points with the highest score.
        And the check how many of samples gonna misclasify completely (sample with miss degree 1)
        and count it as crisped stability.
        The value is a lower bound of the stability.

        Parameters
        ----------
        p : int
            Number of points to remove.

        Returns
        -------
        int
            Maximum crisped stability score.
        """
        # Remove the p instances with the highest score
        removed_sample = set(self.sorted_fuzzy_score_teain[i][0] for i in range(p))

        # Check each sample if all it's friends remove to count as crisped
        crisped_stability = 0
        for idx in np.where(self.classify_correct)[0]:
            friends_of_instance = self.friends[idx]
            if all(friend in removed_sample for friend in friends_of_instance):
                crisped_stability += 1

        return crisped_stability

    def run_crisped_stability(self, p_list: list[int] | int) -> list[int] | int:
        """
        Find the maximum crisped stability of removing p points.
        In this method, the missclsification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the p points with the highest score.
        The value is a much better lower bound of the stability.

        Parameters
        ----------
        p_list : list[int] or int
            List of p values to check or a single integer value.

        Returns
        -------
        list[int] or int
            List of maximum crisped stability score for each p value in list_p
            or a single result if list_p is an integer
        """
        self.sorted_fuzzy_score_teain = self.find_sorted_fuzzy_score_teain()
        self.friends = self.compute_all_friends()
        return self._run(p_list, self.find_crisped_stability)
