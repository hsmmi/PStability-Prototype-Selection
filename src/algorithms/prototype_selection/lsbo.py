import numpy as np
from sklearn.metrics import pairwise_distances
from src.algorithms.prototype_selection.base import BaseAlgorithm
from src.algorithms.prototype_selection.lssm import LSSm


class LSBo(BaseAlgorithm):
    def __init__(self, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.mask: np.ndarray = None
        self.nearest_enemy: np.ndarray = None
        self.distance_nearest_enemy: np.ndarray = None
        self.local_set: list[set] = None
        self.pairwise_distance: np.ndarray = None

    def _set_distance_nearest_enemy(self):
        """
        Set the minimum distance to the nearest enemy for each instance.
        """

        self.nearest_enemy = np.full(self.X_.shape[0], -1)
        self.distance_nearest_enemy = np.full(self.X_.shape[0], np.inf)

        for idx in range(self.X_.shape[0]):
            for idx2 in range(self.X_.shape[0]):
                if (
                    self.y_[idx] != self.y_[idx2]
                    and self.pairwise_distance[idx, idx2]
                    < self.distance_nearest_enemy[idx]
                ):
                    self.distance_nearest_enemy[idx] = self.pairwise_distance[idx, idx2]
                    self.nearest_enemy[idx] = idx2

    def _reachable(self, idx: int, idx2: int) -> bool:
        """
        Check if an instance is reachable from another instance.

        Parameters:
        idx (int): Index of the first instance.
        idx2 (int): Index of the second instance.

        Returns:
        bool: True if the instance is reachable, False otherwise.
        """
        return self.pairwise_distance[idx, idx2] < self.distance_nearest_enemy[idx]

    def _set_local_set(self):
        """
        Set the local set for each instance.
        """

        for idx in range(self.X_.shape[0]):
            for idx2 in range(self.X_.shape[0]):
                if self._reachable(idx, idx2):
                    self.local_set[idx].add(idx2)

    def _fit(self) -> np.ndarray:
        """
        Select instances from the training data.

        Returns:
        np.ndarray: Indices of the selected instances.
        """
        S = LSSm().fit(self.X, self.y).sample_indices_

        self.X_, self.y_ = self.X[S], self.y[S]

        self.mask = np.zeros(len(self.X_), dtype=bool)

        self.pairwise_distance = pairwise_distances(self.X_, metric=self.metric)

        self._set_distance_nearest_enemy()

        self.local_set = [set() for _ in range(self.X_.shape[0])]
        self._set_local_set()
        len_local_set = np.array([len(local_set) for local_set in self.local_set])

        serted_indices = np.argsort(len_local_set)

        for idx in serted_indices:
            # intersection of the local set idx and the mask is empty
            if not self.mask[list(self.local_set[idx])].any():
                self.mask[idx] = True

        return S[self.mask]
