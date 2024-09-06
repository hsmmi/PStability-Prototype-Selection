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
        self.sorted_fuzzy_score_train: list[Tuple[int, float]] = None

    def calculate_distortion(self) -> int:
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

    def find_fuzzy_score_train(self) -> list[Tuple[int, float]]:
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
        scores_idx = np.full(self.n_samples, -1)
        scores_idx[train_indices] = np.arange(len(train_indices))
        scores = [0 for _ in range(len(train_indices))]

        # For each sample in test data
        for idx in range(self.n_samples):
            len_friends = len(friends[idx])
            # Add fuzzy score to each of it's friends
            for idx2 in friends[idx]:
                if scores_idx[idx2] != -1:
                    scores[scores_idx[idx2]] += 1 / len_friends
        for idx in range(len(scores)):
            scores[idx] = (train_indices[idx], scores[idx])

        return scores

    def find_max_fuzzy_score_train(self) -> Tuple[int, float]:
        """
        Find the instance with the maximum fuzzy misclassification score.

        Returns
        -------
        Tuple[int, float]
            The (index, score) pair for the instance with the
            maximum fuzzy misclassification score.
        """
        scores = self.find_fuzzy_score_train()
        max_idx, max_score = -1, -1
        for idx, score in scores:
            if score > max_score:
                max_idx, max_score = idx, score
        return max_idx, max_score

    def find_sorted_fuzzy_score_train(self) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each
        training instance and sort them in descending order.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance sorted in descending order of score.
        """
        scores = self.find_fuzzy_score_train()
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
            logger.debug(f"Checking {check_fn.__name__} for stability={value}")
            res = check_fn(value)
            logger.debug(f"Result: {res}")
            ret.append(res)

        if single_value:
            return ret[0]

        return ret

    def find_exact_stability(
        self, distortion: int, start_index: int = 0, max_stability: int = None
    ) -> int:
        """
        Find the maximum stability value that results no more than the given number of distortion
        in any combination of removing stability points.
        By checking all possible combinations instances to be misclassified.

        Parameters
        ----------
        distortion : int
            The maximum number of allowable misclassifications.
        start_index : int, optional
            The starting index for the search, by default 0.

        Returns
        -------
        int
            The maximum stability value found for the given number of allowable misclassifications.
            returns -1 if no such stability value found.
        """
        if distortion < 0:
            return -1

        if start_index >= self.n_samples:
            # can't continue return invalid value to prevent update
            return self.n_samples + 1

        if distortion == 0:
            idx_min_friends, friends = self.find_instance_with_min_friends(start_index)
            if idx_min_friends == -1:
                return 0
            return len(friends) - 1

        if max_stability == None:
            max_stability = self.n_samples + 1
        # Assume that each instance going to be misclassified
        # with removing it's friends
        for idx in tqdm(
            range(start_index, self.n_samples),
            desc=f"Checking distortion={distortion}",
            leave=False,
            disable=distortion < 3,
        ):
            if self.initial_classify_correct[idx] and self.classify_correct[idx]:
                changes = self.remove_nearest_friends(idx, update_nearest_enemy=False)
                change_distortion = len(changes["classify_incorrect"])
                len_friends = len(changes["friends"])

                # Optimization: Not gonna improve anyway
                if len_friends > max_stability:
                    self.put_back_nearest_friends(changes)
                    continue

                # pass max_stability to tarnsform before knowledge for optimization
                res_max_stability = self.find_exact_stability(
                    distortion - change_distortion, idx + 1, max_stability - len_friends
                )
                if res_max_stability != -1:
                    max_stability = min(max_stability, res_max_stability + len_friends)
                else:
                    # distortion became less than zero
                    max_stability = min(max_stability, len_friends - 1)

                self.put_back_nearest_friends(changes)

        return max_stability

    def run_exact_stability(self, list_distortion: list[int]) -> list[int]:
        """
        Find the exact maximum stability value that results in no more than the given number of misclassifications

        Parameters
        ----------
        list_distortion : list[int]
            Maximum allowable misclassifications.

        Returns
        -------
        list[int]
            List of maximum stability values found for each number of allowable misclassifications in list_distortion.
        """
        return self._run(list_distortion, self.find_exact_stability)

    def find_same_friend_stability(self, distortion: int) -> int:
        """
        Find the lower bound of the maximum stability value that results in no more
        than the given number of misclassifications in any combination of removing stability points.
        Assume that the friends of each instance is completely on the rest of the
        instances after that in increasing friends size order in each class.
        And check for each class because to have a lower bound, removed instances should be
        from the same class.

        Parameters
        ----------
        distortion : int
            The maximum number of allowable misclassifications.

        Returns
        -------
        int
            The lower bound of the maximum stability value found for the given number of allowable misclassifications.
        """
        if distortion == 0:
            return self.number_of_friends_sorted_index[0][1] - 1
        if distortion >= self.n_samples - self.n_misses:
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

        # DP[i][j] =  minimum stability possible by missing j instance from the first i classes.
        dp = [[float("inf")] * (distortion + 2) for _ in range(self.n_classes + 1)]
        dp[0][0] = 0
        for i in range(1, self.n_classes + 1):
            len_class_i = len(number_of_friends_sorted_classes[i])
            for j in range(distortion + 2):
                for x in range(min(j + 1, len_class_i)):
                    # Miss x instance from class i of total j miss and j - x from forst i - 1 class
                    dp[i][j] = min(
                        dp[i][j],
                        dp[i - 1][j - x] + number_of_friends_sorted_classes[i][x],
                    )

        return (
            int(dp[self.n_classes][distortion + 1] - 1)
            if dp[self.n_classes][distortion + 1] != float("inf")
            else -1
        )

    def run_same_friend_stability(self, list_distortion: list[int]) -> list[int]:
        """
        Find the lower bound of the maximum stability value that results in no more
        than the given number of misclassifications.

        Parameters
        ----------
        list_distortion : list[int]
            List of distortion values to check.

        Returns
        -------
        list[int]
            List of same friend stability values found for each number of allowable
            misclassifications in list_distortion.
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_distortion, self.find_same_friend_stability)

    def find_greedy_stability(self, distortion: int) -> int:
        """
        Find the better upper bound of the maximum stability value that results in no more
        than the given number of distortion in any combination of removing stability points.
        Assume that the friends of each instance can be among the rest of the instances.
        And we will remove it's friends from list of other instances friends.
        In each step, we greedily select the instance with the minimum number of friends
        and remove it's friends and update the friends of other instances.

        Parameters
        ----------
        distortion : int
            The maximum number of allowable misclassifications.

        Returns
        -------
        int
            The better upper bound of stability value found for the given distortion.
        """
        if distortion < 0:
            return -1
        # Greedily select the instance with the minimum number of friends
        idx_min_friends, friends = self.find_instance_with_min_friends()
        if idx_min_friends == -1:
            return 0

        if distortion == 0:
            return len(friends) - 1

        changes = self.remove_nearest_friends(
            idx_min_friends, update_nearest_enemy=False
        )

        distortion_changed = len(changes["classify_incorrect"])

        rest = self.find_greedy_stability(distortion - distortion_changed)

        self.put_back_nearest_friends(changes)

        if rest == -1:
            return len(friends) - 1
        return len(friends) + rest

    def run_greedy_stability(self, list_distortion: int) -> list[int]:
        """
        Run the better upper bound of the maximum stability value that results in no more
        than the given number of distortion in any combination of removing stability points.

        Parameters
        ----------
        list_distortion : int
            List of distortion values to check.

        Returns
        -------
        list[int]
            List of greedy stability values found for each distortion value in list_distortion.
        """
        ret = self._run(list_distortion, self.find_greedy_stability)
        return ret

    def find_unique_friend_stability(self, distortion: int) -> int:
        """
        Find the upper bound of the maximum stability value that results in no more
        than the given number of misclassifications in any combination of removing stability points.
        Assume that the friends of each instance is UNIQELY among the rest of the instances.

        Parameters
        ----------
        distortion : int
            The maximum number of allowable change in misclassifications.

        Returns
        -------
        int
            The upper bound of the maximum stability value found for the given distortion.
        """
        upper_bound = 0
        for i in range(distortion + 1):
            upper_bound += self.number_of_friends_sorted_index[i][1]
        return upper_bound - 1

    def run_unique_friend_stability(self, list_distortion: list[int]) -> list[int]:
        """
        Find the upper bound of the maximum stability value that results in no more
        than the given number of misclassifications.
        Parameters
        ----------
        list_distortion : list[int]
            The list of distortion to check for unique friend stability

        Returns
        -------
        list[int]
            List of unique friend stability values found for each number of allowable
            misclassifications in list_distortion
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        max_distortion = 0
        upper_bound = 0
        while upper_bound < self.n_samples:
            upper_bound += self.number_of_friends_sorted_index[max_distortion][1]
            max_distortion += 1
        list_distortion = list(range(min(max_distortion, list_distortion[-1] + 1)))
        return self._run(list_distortion, self.find_unique_friend_stability)

    def find_exact_distortion(self, stability: int, start_index: int = 0) -> int:
        """
        Check all combinations of stability points to remove and determine maximum misclassifications.

        Parameters
        ----------
        stability : int
            Number of points to remove.
        start_index : int, optional
            Starting index for combinations, by default 0.

        Returns
        -------
        int
            Maximum number of misclassifications found.
        """
        if stability == 0:
            return 0
        if start_index >= self.n_samples:
            return 0
        max_misses = 0
        for idx in tqdm(
            range(start_index, self.n_samples),
            desc=f"Checking stability={stability}",
            leave=False,
            disable=stability < 3,
        ):
            if self.mask_train[idx]:
                changed = self.remove_point(idx, update_nearest_enemy=False)
                distortion_changed = len(changed["classify_incorrect"])
                distortion = self.find_exact_distortion(stability - 1, idx + 1)
                max_misses = max(max_misses, distortion_changed + distortion)
                self.put_back_point(idx, changed)
        return max_misses

    def run_exact_distortion(self, p_list: list[int]) -> list[int]:
        """
        Run the exact maximum misclassifications check for each stability value in the range [0, max_stability].
        By checking all possible combinations of removing stability points.

        Parameters
        ----------
        p_list : list[int]
            List of stability values to check.

        Returns
        -------
        list[int]
            List of maximum misclassifications found for each stability value in range[0, max_stability].
        """
        return self._run(p_list, self.find_exact_distortion)

    def find_unique_friend_distortion(self, stability: int) -> int:
        """
        Find the lower bound of the distortion that results of removing
        any combination of stability points.
        Assume that the friends of each instance is UNIQELY among the
        rest of the instances.

        Parameters
        ----------
        stability : int
            The number of points to remove.

        Returns
        -------
        int
            The lower bound of the distortion found for the given stability.
        """
        lower_bound = 0
        sum_friends = 0
        for i in range(self.n_samples - self.n_misses):
            sum_friends += self.number_of_friends_sorted_index[i][1]
            if sum_friends > stability:
                break
            lower_bound = i + 1
        return lower_bound

    def run_unique_friend_distortion(self, list_stability: list[int]) -> list[int]:
        """
        Find the lower bound of distortion for each stability value in the list.

        Parameters
        ----------
        list_stability : list[int]
            The list of distortion to check for same friend stability

        Returns
        -------
        list[int]
            List of Unique Friend Distortion values found for each stability value in list_stability
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_stability, self.find_unique_friend_distortion)

    def find_greedy_distortion(self, stability: int) -> int:
        """
        Find the better lower bound of the distortion that results of removing
        any combination of stability points.
        Assume that the friends of each instance can be among the rest of the instances.
        And we will remove it's friends from list of other instances friends.
        In each step, we greedily select the instance with the minimum number of friends
        and remove it's friends and update the friends of other instances.

        Parameters
        ----------
        stability : int
            The number of points to remove.

        Returns
        -------
        int
            The better lower bound of the maximum stability value found for the given number
            of allowable misclassifications.
        """
        if stability < 0:
            logger.warning(
                f"Value of stability is negative: {stability}. In find_better_same_friend_stability."
            )
            return -1
        if stability == 0:
            return 0
        # Greedily select the instance with the minimum number of friends
        idx_min_friends, friends = self.find_instance_with_min_friends()

        if idx_min_friends == -1:
            return 0

        if stability < len(friends):
            return 0

        changes = self.remove_nearest_friends(
            idx_min_friends, update_nearest_enemy=False
        )

        distortion_changed = len(changes["classify_incorrect"])

        rest = self.find_greedy_distortion(stability - len(friends))

        self.put_back_nearest_friends(changes)

        if rest == -1:
            return 0
        return rest + distortion_changed

    def run_greedy_distortion(self, list_stability: int) -> list[int]:
        """
        Run the better lower bound for distortion for each stability value in the list.

        Parameter
        ----------
        list_stability : int
            List of stability values to check.

        Returns
        -------
        list[int]
            List of Greedy Distortion values found for each stability value in list_stability.
        """
        ret = self._run(list_stability, self.find_greedy_distortion)
        return ret

    def find_same_friend_distortion(self, stability: int) -> int:
        """
        Find the upper bound of distortion for the given stability value.
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

        # DP[i][j] =  Maximum distortion possible by remove j instance from the first i classes.
        dp = [[float("-inf")] * (stability + 1) for _ in range(self.n_classes + 1)]
        dp[0][0] = 0
        for i in range(1, self.n_classes + 1):
            for j in range(stability + 1):
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
            int(dp[self.n_classes][stability])
            if dp[self.n_classes][stability] != float("-inf")
            else -1
        )

    def run_same_friend_distortion(self, list_stability: list[int]) -> list[int]:
        """
        Find the upper bound of distortion for each stability value in the list.
        Distortion is the maximum number of misclassifications which classify
        correct at first that can be achieved by removing any combination of stability points.


        Parameters
        ----------
        list_stability : list[int]
            List of stability values to check.

        Returns
        -------
        list[int]
            List of Same Friend Distortion values found for each stability value in list_stability.
        """
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        return self._run(list_stability, self.find_same_friend_distortion)

    ####################################################################################

    def find_fuzzy_distortion(self, stability: int) -> float:
        """
        Find the maximum fuzzy distortion for the given stability value.
        In this method, the misclassification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the stability points with the highest score.
        The value is a upper bound of the distortion.

        Parameters
        ----------
        stability : int
            Number of points to remove.

        Returns
        -------
        float
            Maximum fuzzy distortion score.
        """
        # Remove the stability instances with the highest score
        fuzzy_score = 0
        for i in range(stability):
            fuzzy_score += self.sorted_fuzzy_score_train[i][1]
        return fuzzy_score

    def run_fuzzy_distortion(self, p_list: list[int] | int) -> list[float] | float:
        """
        Find the maximum fuzzy distortion of removing stability points.
        In this method, the missclsification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the stability points with the highest score.
        The value is a much better upper bound of the distortion.

        Parameters
        ----------
        p_list : list[int] or int
            List of stability values to check or a single integer value.

        Returns
        -------
        list[float] or float
            List of maximum fuzzy distortion score for each stability value in list_stability
            or a single result if list_stability is an integer
        """
        self.sorted_fuzzy_score_train = self.find_sorted_fuzzy_score_train()
        return self._run(p_list, self.find_fuzzy_distortion)

    def find_binary_distortion(self, stability: int) -> int:
        """
        Find the maximum binary distortion for the given stability value.
        In this method, the misclassification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the stability points with the highest score.
        And the check how many of samples gonna misclasify completely (sample with miss degree 1)
        and count it as binary distortion.
        The value is a lower bound of the distortion.

        Parameters
        ----------
        stability : int
            Number of points to remove.

        Returns
        -------
        int
            Maximum binary distortion score.
        """
        # Remove the stability instances with the highest score
        removed_sample = set(
            self.sorted_fuzzy_score_train[i][0] for i in range(stability)
        )

        # Check each sample if all it's friends remove to count as binary
        binary_distortion = 0
        for idx in np.where(self.classify_correct)[0]:
            friends_of_instance = self.friends[idx]
            if all(friend in removed_sample for friend in friends_of_instance):
                binary_distortion += 1

        return binary_distortion

    def run_binary_distortion(self, p_list: list[int] | int) -> list[int] | int:
        """
        Find the maximum binary distortion of removing stability points.
        In this method, the missclsification for a point is not considered as a binary value.
        We calculate a score for each point and then remove the stability points with the highest score.
        The value is a much better lower bound of the distortion.

        Parameters
        ----------
        p_list : list[int] or int
            List of stability values to check or a single integer value.

        Returns
        -------
        list[int] or int
            List of maximum binary distortion score for each stability value in list_stability
            or a single result if list_stability is an integer
        """
        self.sorted_fuzzy_score_train = self.find_sorted_fuzzy_score_train()
        self.friends = self.compute_all_friends()
        return self._run(p_list, self.find_binary_distortion)
