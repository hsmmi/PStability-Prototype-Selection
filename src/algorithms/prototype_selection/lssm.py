import numpy as np
from sklearn.metrics import pairwise_distances
from src.algorithms.prototype_selection.base import BaseAlgorithm


class LSSm(BaseAlgorithm):
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

    def _get_u(self, idx: int) -> int:
        """
        Get the number of reachable instances from the local set.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Number of reachable instances.
        """
        len_u = 0
        for idx2 in range(self.X_.shape[0]):
            if idx2 != idx and idx in self.local_set[idx2]:
                len_u += 1
        return len_u

    def _get_h(self, idx: int) -> int:
        """
        Get the number of reachable instances from the local set that are not in the local set.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Number of reachable instances that are not in the local set.
        """
        return len(np.where(self.nearest_enemy == idx)[0])

    def _fit(self) -> np.ndarray:
        self.X_, self.y_ = self.X, self.y

        self.mask = np.zeros(self.X_.shape[0])

        self.pairwise_distance = pairwise_distances(self.X_, metric=self.metric)

        self._set_distance_nearest_enemy()

        self.local_set = [set() for _ in range(self.X_.shape[0])]
        self._set_local_set()

        u = np.zeros(self.X_.shape[0])
        h = np.zeros(self.X_.shape[0])

        for idx in range(self.X_.shape[0]):
            u[idx] = self._get_u(idx)
            h[idx] = self._get_h(idx)

            if u[idx] >= h[idx]:
                self.mask[idx] = 1

        return np.where(self.mask == 1)[0]
