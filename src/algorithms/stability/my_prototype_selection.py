from typing import Tuple
import numpy as np
from config.log import get_logger
from src.algorithms.stability.p_stability import PStability
from src.utils.visualization import plot_algorithm_results

logger = get_logger("mylogger")


class PrototypeSelection(PStability):
    def __init__(self, metric: str = "euclidean"):
        """
        Initialize PrototypeSelection with the given metric.

        Parameters
        ----------
        metric : str
            Distance metric to use.
        """
        super().__init__(metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrototypeSelection":
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

    def find_total_fuzzy_missclassification_score_teain(self, p: int) -> float:
        """
        Find the total fuzzy missclassification score for the training data.
        Fuzzy missclassification score + number of misclassified instances.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy missclassification score)

        Returns
        -------
        float
            Total fuzzy missclassification score.
        """
        fuzzy_miss_score, _ = self.run_fuzzy_missclassification(p)
        return fuzzy_miss_score + self.n_misses

    def find_best_prototype(self, p: int) -> Tuple[int, float]:
        """
        Find the prototype which after removing gives the minimum
        total fuzzy missclassification score.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy missclassification score)

        Returns
        -------
        Tuple[int, float]
            A tuple containing the index of the prototype and the
            total fuzzy missclassification score.
        """
        min_idx, min_score = -1, np.inf
        for idx in np.where(self.mask_train)[0]:
            changed = self._remove_point(idx, update_nearest_enemy=True)
            score = self.find_total_fuzzy_missclassification_score_teain(p)
            if score < min_score:
                min_idx, min_score = idx, score
            self._put_back_point(idx, changed)
        return min_idx, min_score

    def prototype_reduction(self, p: int) -> dict:
        """
        The order of prototypes to remove based on p-stability.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy missclassification score)

        Returns
        -------
        dict
            A dictionary containing the following keys:

            - removed_prototypes: list,
                List of prototypes removed in order.

            - total_scores: list,
                List of total fuzzy missclassification scores after

            - accuracy: list,
                List of accuracy after removing each prototype.

            - base_total_score: float,
                Total fuzzy missclassification score before removing any prototype.

            - idx_min_total_score: int,
                Number of prototypes removed which gives the minimum
                total fuzzy missclassification score.

            - last_idx_under_base: int,
                Maximum number of prototypes removed which gives a total fuzzy
                missclassification score less than the base total score.
        """
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
            changes = self._remove_point(best_remove_idx, update_nearest_enemy=True)
            accuracy.append(self.accuracy())
            list_changes.append(changes)

        # put back points
        for idx in range(size_one_class, 0, -1):
            self._put_back_point(removed_prototypes[idx], list_changes[idx - 1])

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
