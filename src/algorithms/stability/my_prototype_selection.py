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

    def find_total_distortion(self, p: int) -> float:
        """
        Find the total distortion for the training data.
        Fuzzy stability + number of misclassified instances.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability)

        Returns
        -------
        float
            Total total distortion value.
        """
        fuzzy_stability = self.run_fuzzy_distortion(p)
        return fuzzy_stability + self.n_misses

    def find_best_prototype(self, p: int) -> Tuple[int, float]:
        """
        Find the prototype which after removing gives the minimum
        total distortion value.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability score)

        Returns
        -------
        Tuple[int, float]
            A tuple containing the index of the prototype and the
            total_distortion.
        """
        min_idx, min_total_distortion = -1, np.inf
        for idx in np.where(self.mask_train)[0]:
            changed = self.remove_point(idx, update_nearest_enemy=True)
            total_distortion = self.find_total_distortion(p)
            if total_distortion < min_total_distortion:
                min_idx, min_total_distortion = idx, total_distortion
            self.put_back_point(idx, changed)
        return min_idx, min_total_distortion

    def extract_percentage(self, stop_condition: str) -> float:
        """
        Extract the percentage from the stop condition.

        Parameters
        ----------
        stop_condition : str
            The condition to stop removing prototypes.

        Returns
        -------
        float
            The percentage value extracted.
        """
        if len(stop_condition) == 0:
            return 0
        percentage = float(stop_condition) / 100
        if percentage < 0 or percentage > 1:
            raise ValueError("Invalid percentage")
        return percentage

    def prototype_reduction(self, p: int, stop_condition: str = "acdr") -> dict:
        """
        The order of prototypes to remove based on p-stability.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability score)

        stop_condition : str (default="acdr")
            The condition to stop removing prototypes.

            - "xx.xx%": Remove xx.xx% of the prototypes.(Reduction rate)

            - "xxpl": Remove until xx prototypes left.

            - "xxpr": Remove xx prototypes.

            - "one-class-left": Remove until size of one class is left.

            - "all": Remove all prototypes.

            - "xx.xxacdr": Remove prototypes until the accuracy drops by xx.xx.

            - "min_total_distortion": Remove until the minimum total distortion

            If remaining prototypes are less than p, then it will stop.

        Returns
        -------
        dict
            A dictionary containing the following keys:

            - removed_prototypes: list,
                List of prototypes removed in order.

            - total_distortions: list,
                List of total_distortions after

            - accuracy: list,
                List of accuracy after removing each prototype.

            - base_total_distortion: float,
                Total fuzzy stability score before removing any prototype.

            - idx_min_total_distortion: int,
                Number of prototypes removed which gives the minimum
                total_distortion.

            - last_idx_under_base: int,
                Maximum number of prototypes removed which gives a total fuzzy
                stability score less than the base total score.
        """
        base_total_distortion = self.find_total_distortion(p)
        removed_prototypes = [-1]
        total_distortions = [base_total_distortion]
        accuracy = [self.accuracy()]
        reduction_rate = [self.reduction_rate()]
        idx_min_total_distortion, min_total_distortion = 0, base_total_distortion
        last_idx_under_base = 0
        size_one_class = np.sum(self.y == self.classes[0])
        list_changes = []

        n_remove = 0
        if stop_condition[-1] == "%":
            n_remove = int(
                self.n_samples * (100 - self.extract_percentage(stop_condition[:-1]))
            )
        elif stop_condition[-1] == "pl":
            n_remove = self.n_samples - int(
                self.extract_percentage(stop_condition[:-1])
            )
        elif stop_condition[-1] == "pr":
            n_remove = int(self.extract_percentage(stop_condition[:-1]))
        elif stop_condition == "one-class-left":
            n_remove = self.n_samples - size_one_class
        elif stop_condition == "all":
            n_remove = self.n_samples
        elif stop_condition[-4:] == "acdr":
            n_remove = self.n_samples
            drop_allowed = self.extract_percentage(stop_condition[:-4])
            base_accuracy = self.accuracy()
        else:
            raise ValueError("Invalid stop_condition")

        if n_remove > self.n_samples - p:
            n_remove = self.n_samples - p
        for idx in range(1, n_remove + 1):
            best_remove_idx, best_total_distortion_after_remove = (
                self.find_best_prototype(p)
            )
            changes = self.remove_point(best_remove_idx, update_nearest_enemy=True)

            if best_total_distortion_after_remove <= min_total_distortion:
                min_total_distortion = best_total_distortion_after_remove
                idx_min_total_distortion = idx
            if best_total_distortion_after_remove < base_total_distortion:
                last_idx_under_base = idx

            list_changes.append(changes)
            if stop_condition[-4:] == "acdr":
                if self.accuracy() < base_accuracy - drop_allowed:
                    break
            removed_prototypes.append(best_remove_idx)
            total_distortions.append(best_total_distortion_after_remove)
            accuracy.append(self.accuracy())
            reduction_rate.append(self.reduction_rate())

        # put back points
        for idx in range(len(removed_prototypes) - 1, 0, -1):
            self.put_back_point(removed_prototypes[idx], list_changes[idx - 1])

        ret = {
            "removed_prototypes": removed_prototypes,
            "total_distortions": total_distortions,
            "accuracy": accuracy,
            "reduction_rate": reduction_rate,
            "base_total_distortion": base_total_distortion,
            "idx_min_total_distortion": idx_min_total_distortion,
            "min_total_distortion": min_total_distortion,
            "last_idx_under_base": last_idx_under_base,
        }

        logger.debug(
            "Removed prototypes: %(removed_prototypes)s",
            ret,
        )

        return ret

    def prototype_selection(self, p: int, stop_condition: str = "acdr") -> list[int]:
        """
        Select prototypes based on p-stability.

        ! Use prototype_reduction to get more details.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability)

        stop_condition : str (default="all")


        Returns
        -------
        list[int]
            List of indices of the remaining prototypes.
        """
        result = self.prototype_reduction(p)
        removed_prototypes = result["removed_prototypes"]
        remaining_prototypes = np.setdiff1d(
            np.arange(self.n_samples), removed_prototypes
        )
        return remaining_prototypes

    def fit_transform(self, X: np.ndarray, y: np.ndarray, p: int = 1) -> np.ndarray:
        """
        Fit the model using the training data and return the reduced dataset.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.

        Returns
        -------
        np.ndarray
            Reduced dataset.
        """
        self.fit(X, y)
        remaining_indices = self.prototype_selection(p)
        return X[remaining_indices], y[remaining_indices]
