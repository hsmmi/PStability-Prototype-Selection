from typing import Tuple
import numpy as np
from config.log import get_logger
from src.algorithms.stability.p_stability import PStability

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

    def prototype_reduction(self, p: int) -> Tuple[np.ndarray, float]:
        """
        Reduce the number of samles by removing p prototypes.

        Parameters
        ----------
        p : int
            Number of prototypes to remove.

        Returns
        -------
        Tuple[np.ndarray, float]
            A tuple containing the indices of the removed prototypes and the number of misclassifications
        """

        if p == 0:
            return np.array([]), 0

        # Find instance with maximum fuzzy missclassification score
        idx, fuzzy_miss = self.find_max_fuzzy_missclassification_score()

        if idx == -1:
            return -1

        # Remove the instance with the maximum fuzzy missclassification score
        changed = self._remove_point_update_neighbours(idx)

        # Recursively call the prototype reduction function
        removed, misses = self.prototype_reduction(p - 1)

        # Put back the removed instance
        self._put_back_point(idx, changed)

        return np.concatenate(([idx], removed)), misses + fuzzy_miss
