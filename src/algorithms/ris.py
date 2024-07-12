import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from src.algorithms.base import BaseAlgorithm


class RIS(BaseAlgorithm):
    def __init__(self, method="RIS1", threshold=0.1):
        self.threshold = threshold
        self.method = method
        self.pairwise_distances = None

    def scores_redius(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the relevance scores and the radius of each instance.
        Radius is defined as the distance to the nearest enemy instance.

        Returns:
            tuple[np.ndarray, np.ndarray]: Relevance scores and radius of each instance.
        """
        m = len(self.X)
        scores = np.zeros(m)
        radius = np.full(m, np.inf)

        # TODO: Optimize this loop
        scores = np.zeros(m)
        self.pairwise_distances = euclidean_distances(self.X)
        for i in range(m):
            sm_denom = 0
            sm_num = 0
            for j in range(m):
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
            scores[i] = sm_num / sm_denom

        # Min-max normalization
        scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()

        return scores, radius

    def is_relevant(self, selected_indices, idx):
        for r in selected_indices:
            if (
                self.y[r] == self.y[idx]
                and self.pairwise_distances[r, idx] <= self.radius[r]
            ):
                return False
        return True

    def select_instances(self):
        sorted_indices = np.argsort(self.scores)[::-1]
        selected_indices = []

        for idx in sorted_indices:
            if self.scores[idx] < self.threshold:
                break
            if self.is_relevant(selected_indices, idx):
                selected_indices.append(idx)

        return selected_indices

    def RIS1(self):
        self.scores, self.radius = self.scores_redius()

        selected_indices = self.select_instances()

        return selected_indices

    def select(self):
        if self.method == "RIS1":
            return self.RIS1()
        else:
            raise ValueError(f"Invalid method: {self.method}")
