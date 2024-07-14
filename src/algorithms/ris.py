import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from src.algorithms.base import BaseAlgorithm


class RIS(BaseAlgorithm):
    def __init__(self, method="RIS1"):
        self.threshold = None
        self.method = method
        self.pairwise_distances = None

    def _scores_redius(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the relevance scores and the radius of each instance.
        Radius is defined as the distance to the nearest enemy instance.

        Returns:
            tuple[np.ndarray, np.ndarray]: Relevance scores and radius of each instance.
        """
        self.n_samples = len(self.X)
        scores = np.zeros(self.n_samples)
        radius = np.full(self.n_samples, np.inf)

        # TODO: Optimize this loop
        scores = np.zeros(self.n_samples)
        self.pairwise_distances = euclidean_distances(self.X)
        for i in range(self.n_samples):
            sm_denom = 0
            sm_num = 0
            for j in range(self.n_samples):
                dist = self.pairwise_distances[i, j]
                val = np.exp(-dist)
                sm_denom += val
                if self.y[i] == self.y[j]:
                    sm_num += val
                else:
                    sm_num -= val
                    if dist < radius[i]:
                        radius[i] = dist
            # exp(0) = 1
            sm_num -= 1
            sm_denom -= 1
            if sm_denom == 0:
                scores[i] = 0
            else:
                scores[i] = sm_num / sm_denom

        return scores, radius

    def _is_relevant(self, selected_indices, idx):
        for r in selected_indices:
            if (
                self.y[r] == self.y[idx]
                and self.pairwise_distances[r, idx] <= self.radius[r]
            ):
                return False
        return True

    def _select_instances(self, sorted_indices):
        selected_indices = []

        for idx in sorted_indices:
            if self.scores[idx] < self.threshold:
                break
            if self._is_relevant(selected_indices, idx):
                selected_indices.append(idx)

        return selected_indices

    def _RIS1(self):
        self.scores, self.radius = self._scores_redius()

        # Min-max normalization
        self.scores = MinMaxScaler().fit_transform(self.scores.reshape(-1, 1)).flatten()

        sorted_indices = np.argsort(self.scores)[::-1]
        selected_indices = self._select_instances(sorted_indices)

        return selected_indices

    def _scale_scores_per_class(self):
        for cls in self.classes_:
            class_indices = np.where(self.y == cls)[0]
            class_scores = self.scores[class_indices]
            class_scores = (
                MinMaxScaler().fit_transform(class_scores.reshape(-1, 1)).flatten()
            )
            self.scores[class_indices] = class_scores

    def _RIS2(self):
        self.scores, self.radius = self._scores_redius()

        self._scale_scores_per_class()

        sorted_indices = np.argsort(self.scores)[::-1]
        selected_indices = self._select_instances(sorted_indices)

        return selected_indices

    def _eleminate_instances(self):
        sorted_indices = np.argsort(self.scores)[::-1]
        for i, idx in enumerate(sorted_indices):
            if self.scores[idx] < self.threshold:
                return sorted_indices[:i]
        return sorted_indices

    def _recalculate_radius(self, selected_indices):
        radius = np.full(self.n_samples, np.inf)
        for idx in selected_indices:
            for r in selected_indices:
                if self.y[r] != self.y[idx]:
                    dist = self.pairwise_distances[r, idx]
                    if dist < radius[idx]:
                        radius[idx] = dist
        return radius

    def _RIS3(self):
        self.scores, self.radius = self._scores_redius()

        self._scale_scores_per_class()

        # Eliminate instances with lower than threshold scores
        sorted_remaining_indices = self._eleminate_instances()

        # Recalculate the radius
        self.radius = self._recalculate_radius(sorted_remaining_indices)

        selected_indices = self._select_instances(sorted_remaining_indices)

        return selected_indices

    def _run_method(self):
        if self.method == "RIS1":
            return self._RIS1()
        elif self.method == "RIS2":
            return self._RIS2()
        elif self.method == "RIS3":
            return self._RIS3()
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def _best_threshold(self):
        thresholds = np.arange(0.1, 1.1, 0.1)
        scores = []
        for threshold in thresholds:
            self.threshold = threshold
            self.sample_indices_ = self._run_method()
            scores.append(self.score()[0])

        # Select the best threshold base on highest accuracy
        best_threshold = thresholds[np.argmax(scores)]
        self.threshold = best_threshold

        return best_threshold

    def select(self):
        self.threshold = self._best_threshold()
        return self._run_method()
