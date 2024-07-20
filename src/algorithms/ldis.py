import numpy as np
from sklearn.metrics import pairwise_distances
from src.algorithms.base import BaseAlgorithm


class LDIS(BaseAlgorithm):
    """
    Local Density-based Instance Selection (LDIS) algorithm.
    DOI: 10.1109/ICTAI.2015.114
    """

    def __init__(self, n_neighbors: int = 3, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.n_neighbors: int = n_neighbors
        self.pairwise_distance: np.ndarray = None
        # partial k-neighborhood
        self.pkn: np.ndarray = None
        self.dencity: np.ndarray = None

    def _set_nearest_neighbors(self, class_members: np.ndarray):
        """
        Set the nearest neighbors using the KNN algorithm.
        """
        self.pairwise_distance = pairwise_distances(
            self.X[class_members], metric=self.metric
        )

        for i in range(self.pairwise_distance.shape[0]):
            self.pairwise_distance[i][i] = -1.0

        k = min(self.n_neighbors, len(class_members) - 1)

        self.pkn = np.zeros((len(class_members), k), dtype=int)

        for i in range(len(class_members)):
            self.pkn[i] = np.argsort(self.pairwise_distance[i])[1 : k + 1]

    def _set_density(self) -> float:
        """
        Calculate the density of an instance on its class.
        """
        len_class_members = len(self.pairwise_distance)
        self.dencity = np.zeros(len_class_members)
        for idx in range(len_class_members):
            for idx2 in range(len_class_members):
                if idx != idx2:
                    self.dencity[idx] += self.pairwise_distance[idx][idx2]
            self.dencity[idx] = -self.dencity[idx] / (len_class_members - 1)

    def _fit(self) -> np.ndarray:
        """
        Select instances from the training data.

        Returns:
        np.ndarray: Indices of the selected instances.
        """
        for cls in self.classes_:
            class_members = np.where(self.y == cls)[0]
            self._set_nearest_neighbors(class_members)
            self._set_density()
            for idx in range(len(class_members)):
                found_denser = False
                for idx2 in self.pkn[idx]:
                    if self.dencity[idx] < self.dencity[idx2]:
                        found_denser = True
                        break
                if not found_denser:
                    self.sample_indices_.append(class_members[idx])
        return np.sort(self.sample_indices_)
