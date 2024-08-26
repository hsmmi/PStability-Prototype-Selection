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

    def find_objective_function(self, p: int) -> float:
        """
        Find the objective function for the training data.
        Fuzzy stability + number of misclassified instances.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability)

        Returns
        -------
        float
            Total objective function value.
        """
        fuzzy_stability = self.run_fuzzy_stability(p)
        return fuzzy_stability + self.n_misses

    def find_best_prototype(self, p: int) -> Tuple[int, float]:
        """
        Find the prototype which after removing gives the minimum
        objective function value.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability score)

        Returns
        -------
        Tuple[int, float]
            A tuple containing the index of the prototype and the
            objective_function.
        """
        min_idx, min_objective_function = -1, np.inf
        for idx in np.where(self.mask_train)[0]:
            changed = self.remove_point(idx, update_nearest_enemy=True)
            objective_function = self.find_objective_function(p)
            if objective_function < min_objective_function:
                min_idx, min_objective_function = idx, objective_function
            self.put_back_point(idx, changed)
        return min_idx, min_objective_function

    def prototype_reduction(self, p: int, stop_condition: str = "all") -> dict:
        """
        The order of prototypes to remove based on p-stability.

        Parameters
        ----------
        p : int
            p-stability parameter to use.(For fuzzy stability score)

        stop_condition : str (default="all")
            The condition to stop removing prototypes.

            - "xx.xx%": Remove xx.xx% of the prototypes.(Reduction rate)

            - "xxpl": Remove until xx prototypes left.

            - "xxpr": Remove xx prototypes.

            - "one-class-left": Remove until size of one class is left.

            - "all": Remove all prototypes.

            - "xx.xxacdr": Remove prototypes until the accuracy drops by xx.xx.

            If remaining prototypes are less than p, then it will stop.

        Returns
        -------
        dict
            A dictionary containing the following keys:

            - removed_prototypes: list,
                List of prototypes removed in order.

            - objective_functions: list,
                List of objective_functions after

            - accuracy: list,
                List of accuracy after removing each prototype.

            - base_objective_function: float,
                Total fuzzy stability score before removing any prototype.

            - idx_min_objective_function: int,
                Number of prototypes removed which gives the minimum
                objective_function.

            - last_idx_under_base: int,
                Maximum number of prototypes removed which gives a total fuzzy
                stability score less than the base total score.
        """
        base_objective_function = self.find_objective_function(p)
        removed_prototypes = [-1]
        objective_functions = [base_objective_function]
        accuracy = [self.accuracy()]
        idx_min_objective_function, min_objective_function = 0, base_objective_function
        last_idx_under_base = 0
        size_one_class = np.sum(self.y == self.classes[0])
        list_changes = []

        n_remove = 0
        if stop_condition[-1] == "%":
            n_remove = int(self.n_samples * (1 - float(stop_condition[:-1]) / 100))
        elif stop_condition[-1] == "pl":
            n_remove = self.n_samples - int(stop_condition[:-1])
        elif stop_condition[-1] == "pr":
            n_remove = int(stop_condition[:-1])
        elif stop_condition == "one-class-left":
            n_remove = self.n_samples - size_one_class
        elif stop_condition == "all":
            n_remove = self.n_samples
        elif stop_condition[-4:] == "acdr":
            n_remove = self.n_samples
            drop_allowed = float(stop_condition[:-4]) / 100
            base_accuracy = self.accuracy()
        else:
            raise ValueError("Invalid stop_condition")

        if n_remove > self.n_samples - p:
            n_remove = self.n_samples - p
        for idx in range(1, n_remove + 1):
            best_remove_idx, best_objective_function_after_remove = (
                self.find_best_prototype(p)
            )
            if best_objective_function_after_remove <= min_objective_function:
                min_objective_function = best_objective_function_after_remove
                idx_min_objective_function = idx
            if best_objective_function_after_remove < base_objective_function:
                last_idx_under_base = idx
            removed_prototypes.append(best_remove_idx)
            objective_functions.append(best_objective_function_after_remove)
            changes = self.remove_point(best_remove_idx, update_nearest_enemy=True)
            accuracy.append(self.accuracy())
            list_changes.append(changes)
            if stop_condition[-4:] == "acdr":
                if accuracy[-1] < base_accuracy - drop_allowed:
                    break
        # put back points
        for idx in range(size_one_class, 0, -1):
            self.put_back_point(removed_prototypes[idx], list_changes[idx - 1])

        ret = {
            "removed_prototypes": removed_prototypes,
            "objective_functions": objective_functions,
            "accuracy": accuracy,
            "base_objective_function": base_objective_function,
            "idx_min_objective_function": idx_min_objective_function,
            "min_objective_function": min_objective_function,
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
            p-stability parameter to use.(For fuzzy stability)

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
