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
        self.sorted_fuzzy_missclassification_score_teain: list[Tuple[int, float]] = None

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

    def find_fuzzy_missclassification_score_teain(
        self,
    ) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each training instance. Which is based on
        friends of test instances on the training instances.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance.
        """
        friends = self.compute_all_firends()
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

    def find_max_fuzzy_missclassification_score_teain(self) -> Tuple[int, float]:
        """
        Find the instance with the maximum fuzzy misclassification score.

        Returns
        -------
        Tuple[int, float]
            The (index, score) pair for the instance with the
            maximum fuzzy misclassification score.
        """
        scores = self.find_fuzzy_missclassification_score_teain()
        max_idx, max_score = -1, -1
        for idx, score in scores:
            if score > max_score:
                max_idx, max_score = idx, score
        return max_idx, max_score

    def find_sorted_fuzzy_missclassification_score_teain(
        self,
    ) -> list[Tuple[int, float]]:
        """
        Calculate the fuzzy misclassification score for each
        training instance and sort them in descending order.

        Returns
        -------
        list[Tuple[int, float]]
            The (index, score) pairs for each instance sorted in descending order of score.
        """
        scores = self.find_fuzzy_missclassification_score_teain()
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
        self.mask_train = np.ones(self.n_samples, dtype=bool)
        ret = []
        for p_value in p:
            logger.debug(f"Checking {check_fn.__name__} for p={p_value}")
            max_misses = check_fn(p_value)
            logger.debug(f"Result: {max_misses}")
            ret.append(max_misses)

        return ret

    def find_exact_miss(self, p: int, start_index: int = 0) -> int:
        if p == 0:
            return self.calculate_stability()
        max_misses = 0
        for idx in range(start_index, self.n_samples):
            changed = self.remove_point(idx, update_nearest_enemy=True)
            misses = self.find_exact_miss(p - 1, idx + 1)
            max_misses = max(max_misses, misses)
            self.put_back_point(idx, changed)
        return max_misses

    def find_exact_p(self, miss: int, start_index: int = 0) -> int:
        if miss == 0:
            friends = self.find_instance_with_min_friends(start_index)
            return len(friends) - 1
        max_p = self.n_samples + 1
        for idx in range(start_index, self.n_samples):
            if self.classify_correct[idx]:
                changes = self.remove_nearest_friends(idx, update_nearest_enemy=True)
                missed = len(changes["classify_incorrect"])
                if missed <= miss:
                    res_max_p = self.find_exact_p(miss - missed, idx + 1)
                    max_p = min(max_p, res_max_p + len(changes["friends"]))
                self.put_back_nearest_friends(changes)
        return max_p

    def find_lower_bound_p(self, miss: int) -> int:
        number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        if miss == 0:
            return self.number_of_friends_sorted_index[0][1] - 1
        max_p = self.n_samples + 1
        for class_label in self.classes:
            number_of_friends_sorted_index_class = [
                (idx, n_friends)
                for (idx, n_friends) in number_of_friends_sorted_index
                if self.y[idx] == class_label
            ]
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
        return max_p

    def find_upper_bound_p(self, miss: int) -> int:
        number_of_friends_sorted_index = self.find_number_of_friends_sorted_index()
        upper_bound = 0
        for i in range(miss + 1):
            upper_bound += number_of_friends_sorted_index[i][1]
        return upper_bound - 1

    def find_better_upper_bound_p(self, miss: int) -> int:
        idx_min_friends, friends = self.find_instance_with_min_friends()
        if miss == 0:
            return len(friends) - 1
        changes = self.remove_nearest_friends(idx_min_friends)
        missed = len(changes["classify_incorrect"])
        if miss - missed >= 0:
            rest = self.find_better_upper_bound_p(miss - missed)
        else:
            rest = -1
        self.put_back_nearest_friends(changes)
        if rest == -1:
            return -1
        return rest + len(changes["friends"])

    def find_fuzzy_missclassification(self, p: int) -> float:
        sorted_fuzzy_missclassification_score_teain = (
            self.find_sorted_fuzzy_missclassification_score_teain()
        )
        fuzzy_misses = 0
        for i in range(p):
            fuzzy_misses += sorted_fuzzy_missclassification_score_teain[i][1]
        return fuzzy_misses

    def find_total_fuzzy_missclassification_score_teain(self, p: int) -> float:
        fuzzy_miss_score = self.find_fuzzy_missclassification(p)
        return fuzzy_miss_score + self.n_misses

    def find_best_prototype(self, p: int) -> Tuple[int, float]:
        min_idx, min_score = -1, np.inf
        for idx in np.where(self.mask_train)[0]:
            changed = self.remove_point(idx, update_nearest_enemy=True)
            score = self.find_total_fuzzy_missclassification_score_teain(p)
            if score < min_score:
                min_idx, min_score = idx, score
            self.put_back_point(idx, changed)
        return min_idx, min_score

    def prototype_selection(
        self, p: int, n_remove: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_total_score = self.find_total_fuzzy_missclassification_score_teain(p)
        removed_prototypes = [-1]
        total_scores = [base_total_score]
        for _ in range(1, n_remove + 1):
            best_remove_idx, best_total_score_after_remove = self.find_best_prototype(p)
            removed_prototypes.append(best_remove_idx)
            total_scores.append(best_total_score_after_remove)

        return total_scores, removed_prototypes

    def prototype_reduction(self, p: int) -> dict:
        base_total_score = self.find_total_fuzzy_missclassification_score_teain(p)
        removed_prototypes = [-1]
        total_scores = [base_total_score]
        accuracy = [self.accuracy()]
        idx_min_total_score, min_total_score = 0, base_total_score
        last_idx_under_base = 0
        size_one_class = np.sum(self.y == 1)
        list_changes = []
        for idx in range(1, size_one_class + 1):
            best_remove_idx, best_total_score_after_remove = self.find_best_prototype(p)
            if best_total_score_after_remove <= min_total_score:
                min_total_score = best_total_score_after_remove
                idx_min_total_score = idx
            if best_total_score_after_remove < base_total_score:
                last_idx_under_base = idx
            removed_prototypes.append(best_remove_idx)
            total_scores.append(best_total_score_after_remove)
            changes = self.remove_point(best_remove_idx, update_nearest_enemy=True)
            accuracy.append(self.accuracy())
            list_changes.append(changes)

        # put back points
        for idx in range(size_one_class, 0, -1):
            self.put_back_point(removed_prototypes[idx], list_changes[idx - 1])

        ret = {
            "removed_prototypes": removed_prototypes,
            "total_scores": total_scores,
            "accuracy": accuracy,
            "base_total_score": base_total_score,
            "idx_min_total_score": idx_min_total_score,
            "min_total_score": min_total_score,
            "last_idx_under_base": last_idx_under_base,
        }

        logger.debug(
            "Removed prototypes: %(removed_prototypes)s",
            ret,
        )

        return ret

    def prototype_selection(self, p: int) -> list[int]:
        """
        Select prototypes based on p-stability.

        ! Use prototype_reduction to get more details.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy missclassification score)

        Returns
        -------
        list[int]
            List of prototypes selected.
        """
        result = self.prototype_reduction(p)
        removed_prototypes = result["removed_prototypes"]
        remaining_prototypes = np.setdiff1d(
            np.arange(self.n_samples), removed_prototypes
        )
        return remaining_prototypes
