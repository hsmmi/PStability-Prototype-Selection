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
        self.sorted_fuzzy_score_teain: list[Tuple[int, float]] = None

    def calculate_stability(self) -> int:
        """
        Check how many samples that classified correctly at first will misclassify
        now in the current state.

        Returns
        -------
        int
            The number of samples that were correctly classified initially but are misclassified now.
        """
        return self.n_misses - self.n_misses_initial

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
            if self.classify_correct[idx]:
                changes = self.remove_nearest_friends(idx, update_nearest_enemy=True)
                change_stability = len(changes["classify_incorrect"]) - len(
                    changes["classify_correct"]
                )
                if change_stability <= stability:
                    res_max_p = self.find_exact_p(stability - change_stability, idx + 1)
                    if res_max_p != -1:
                        max_p = min(max_p, res_max_p + len(changes["friends"]))

                self.put_back_nearest_friends(changes)

        if max_p == self.n_samples + 1:
            return -1
        return max_p

    def find_lower_bound_p(self, stability: int) -> int:
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        number_of_friends_sorted_classes = [[]]
        for class_label in self.classes:
            number_of_friends_sorted = [0] + [
                n_friends
                for idx, n_friends in self.number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            number_of_friends_sorted_classes.append(number_of_friends_sorted)
        dp = [[float("inf")] * (stability + 2) for _ in range(self.n_classes + 1)]
        dp[0][0] = 0
        for i in range(1, self.n_classes + 1):
            len_class_i = len(number_of_friends_sorted_classes[i])
            for j in range(stability + 2):
                for x in range(min(j + 1, len_class_i)):
                    if (
                        x + 1 < len_class_i
                        and number_of_friends_sorted_classes[i][x]
                        == number_of_friends_sorted_classes[i][x + 1]
                    ):
                        continue
                    dp[i][j] = min(
                        dp[i][j],
                        dp[i - 1][j - x] + number_of_friends_sorted_classes[i][x],
                    )
        return int(dp[self.n_classes][stability + 1] - 1)

    def find_lower_bound_p_2(self, stability: int) -> int:
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
        self.number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        if stability == 0:
            return self.number_of_friends_sorted_index[0][1] - 1
        max_p = self.n_samples + 1
        for class_label in self.classes:
            # Select instances of the class in the order of increasing number of friends
            number_of_friends_sorted_index_class = [
                (idx, n_friends)
                for (idx, n_friends) in self.number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            # Check if exact stability is possible by checking the next instance in the order to
            # have more friends than the current instance.
            if (
                number_of_friends_sorted_index_class[stability - 1][1]
                == number_of_friends_sorted_index_class[stability][1]
            ):
                continue
            else:
                max_p = min(
                    max_p,
                    number_of_friends_sorted_index_class[stability][1] - 1,
                )

        if max_p == self.n_samples + 1:
            return -1
        return max_p

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
        # Greedily select the instance with the minimum number of friends
        idx_min_friends, friends = self.find_instance_with_min_friends()
        if stability < 0:
            logger.warning(
                f"Value of stability is negative: {stability}. In find_better_upper_bound_p."
            )
        if stability == 0:
            if idx_min_friends == -1:
                return 0
            return len(friends) - 1
        if idx_min_friends == -1:
            return -1
        changes = self.remove_nearest_friends(
            idx_min_friends, update_nearest_enemy=True
        )
        stability_changed = len(changes["classify_incorrect"]) - len(
            changes["classify_correct"]
        )
        if stability - stability_changed >= 0:
            rest = self.find_better_upper_bound_p(stability - stability_changed)
        else:
            rest = -1

        self.put_back_nearest_friends(changes)

        if rest == -1:
            return -1
        return rest + len(changes["friends"])

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
            return self.calculate_stability()
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
                changed = self.remove_point(idx, update_nearest_enemy=True)
                misses = self.find_exact_stability(p - 1, idx + 1)
                max_misses = max(max_misses, misses)
                self.put_back_point(idx, changed)
        return max_misses

    def find_lower_bound_stability(self, p: int) -> int:
        number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        lower_bound = 0
        sum_friends = 0
        for i in range(self.n_samples):
            sum_friends += number_of_friends_sorted_index[i][1]
            if sum_friends > p:
                break
            lower_bound = i + 1
        return lower_bound

    def find_better_lower_bound_stability(self, p: int) -> int:
        if p == 0:
            return 0
        idx_min_friends, friends = self.find_instance_with_min_friends()
        if p < len(friends):
            return 0
        changes = self.remove_nearest_friends(
            idx_min_friends, update_nearest_enemy=True
        )
        stability_changed = len(changes["classify_incorrect"]) - len(
            changes["classify_correct"]
        )
        rest = self.find_better_lower_bound_stability(p - len(friends))
        self.put_back_nearest_friends(changes)
        return rest + stability_changed

    def find_upper_bound_stability(self, p: int) -> int:
        number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        number_of_friends_sorted_classes = [[]]
        for class_label in self.classes:
            number_of_friends_sorted = [
                n_friends
                for idx, n_friends in number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            number_of_friends_sorted_classes.append(number_of_friends_sorted)
        dp = [[float("-inf")] * (p + 1) for _ in range(self.n_classes + 1)]
        dp[0][0] = 0
        for i in range(1, self.n_classes + 1):
            for j in range(p + 1):
                class_miss = 0
                len_class_i = len(number_of_friends_sorted_classes[i])
                for x in range(min(j, len_class_i) + 1):
                    while (
                        class_miss < len_class_i
                        and number_of_friends_sorted_classes[i][class_miss] <= x
                    ):
                        class_miss += 1
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - x] + class_miss)
        return int(dp[self.n_classes][p])

    def find_upper_bound_stability2(self, p: int) -> int:
        """
        Find the upper bound of stability for the given p value.
        Assume that the friends of each instance is completely on the rest of the
        instances after that instance in increasing friends size order in each class.
        """
        max_stability = 0
        for class_label in self.classes:
            # Select instances of the class in the order of increasing number of friends
            number_of_friends_sorted_index_class = [
                (idx, n_friends)
                for (idx, n_friends) in self.number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
            # Find the maximum stability for the class
            stability = 0
            for idx in range(len(number_of_friends_sorted_index_class)):
                if number_of_friends_sorted_index_class[idx][1] <= p:
                    stability = idx + 1
            max_stability = max(max_stability, stability)
        return max_stability

    ####################################################################################

    def find_sorted_fuzzy_score_teain(self) -> list[Tuple[int, float]]:
        friends = self.compute_all_friends()
        train_indices = np.where(self.mask_train)[0]
        scores = []
        for idx in train_indices:
            score = 0
            for idx2 in range(self.n_samples):
                if len(friends[idx2]) == 0:
                    continue
                if idx in friends[idx2]:
                    score += 1 / len(friends[idx2])
            scores.append((idx, score))
        sorted_fuzzy_score_teain = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_fuzzy_score_teain

    def find_fuzzy_stability(self, p: int) -> float:
        sorted_fuzzy_score_teain = self.find_sorted_fuzzy_score_teain()
        fuzzy_score = 0
        for i in range(p):
            fuzzy_score += sorted_fuzzy_score_teain[i][1]
        return fuzzy_score

    def find_crisped_stability(self, p: int) -> int:
        friends = self.compute_all_friends()
        sorted_fuzzy_score_teain = self.find_sorted_fuzzy_score_teain()
        removed_sample = set(sorted_fuzzy_score_teain[i][0] for i in range(p))
        crisped_stability = 0
        for idx in np.where(self.classify_correct)[0]:
            friends_of_instance = friends[idx]
            if all(friend in removed_sample for friend in friends_of_instance):
                crisped_stability += 1
        return crisped_stability

    def find_objective_function(self, p: int) -> float:
        fuzzy_stability = self.find_fuzzy_stability(p)
        return fuzzy_stability + self.n_misses

    def find_best_prototype(self, p: int) -> Tuple[int, float]:
        min_idx, min_objective_function_after_remove = -1, np.inf
        for idx in np.where(self.mask_train)[0]:
            changed = self.remove_point(idx, update_nearest_enemy=True)
            objective_function = self.find_objective_function(p)
            if objective_function < min_objective_function_after_remove:
                min_idx, min_objective_function_after_remove = idx, objective_function
            self.put_back_point(idx, changed)
        return min_idx, min_objective_function_after_remove

    def prototype_reduction(
        self, p: int, n_remove: int
    ) -> Tuple[list[int], list[float]]:
        removed_prototypes = []
        objective_functions = []
        for _ in range(n_remove):
            removed_prototype, objective_function_after_remove = (
                self.find_best_prototype(p)
            )
            if objective_function_after_remove <= min_objective_function:
                min_objective_function = objective_function_after_remove
            removed_prototypes.append(removed_prototype)
            objective_functions.append(objective_function_after_remove)
        return removed_prototypes, objective_functions
