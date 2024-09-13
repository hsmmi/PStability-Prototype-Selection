import collections
import numpy as np
from src.algorithms.prototype_selection.base import BaseAlgorithm
from src.algorithms.prototype_selection.enn import ENN
from sklearn.metrics.pairwise import pairwise_distances


class DROP3(BaseAlgorithm):
    def __init__(self, n_neighbors: int = 3, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.n_neighbors: int = n_neighbors
        self.pairwise_distance: np.ndarray = None
        self.nearest_neighbors_complete: np.ndarray = None
        self.nearest_neighbors: np.ndarray = None
        self.nearest_enemy: np.ndarray = None
        self.nearest_enemy_distance: np.ndarray = None
        self.associates: list[set] = None
        self.mask: np.ndarray = None

    def set_nearest_neighbors(self):
        self.pairwise_distance = pairwise_distances(self.X_, metric=self.metric)

        for i in range(self.pairwise_distance.shape[0]):
            self.pairwise_distance[i][i] = -1.0

        self.nearest_neighbors_complete = np.argsort(self.pairwise_distance)[:, 1:]
        self.nearest_neighbors = [
            x[: self.n_neighbors] for x in self.nearest_neighbors_complete
        ]

    def set_nearest_enemy(self):
        self.nearest_enemy = np.zeros(len(self.X_))
        self.nearest_enemy_distance = np.zeros(len(self.X_))

        for i in range(len(self.X_)):
            self.nearest_enemy_distance[i] = np.inf
            for j in range(len(self.X_)):
                if self.y_[i] != self.y_[j]:
                    if self.pairwise_distance[i][j] < self.nearest_enemy_distance[i]:
                        self.nearest_enemy_distance[i] = self.pairwise_distance[i][j]
                        self.nearest_enemy[i] = j

    def set_associates(self):
        self.associates = [set() for _ in range(len(self.X_))]
        for i in range(len(self.X_)):
            for j in self.nearest_neighbors[i]:
                self.associates[j].add(i)

    def get_nearest_neighbors(self, idx):
        nearest_neighbors = []
        for neighbor in self.nearest_neighbors_complete[idx]:
            if self.mask[neighbor]:
                nearest_neighbors.append(neighbor)
            if len(nearest_neighbors) == self.n_neighbors:
                return nearest_neighbors
        return nearest_neighbors

    def most_common(self, labels):
        try:
            counts = np.bincount(labels)
            return np.argmax(counts)
        except:
            return collections.Counter(labels).most_common()[0][0]

    def classify(self, associate):
        """
        Classify each instance which is in associate list and then
        return number of associates whihch classify correctly
        """
        ans = 0
        for idx in associate:
            idx_neighbors = self.get_nearest_neighbors(idx)
            pred_idx = self.most_common([self.y_[i] for i in idx_neighbors])
            if pred_idx == self.y_[idx]:
                ans += 1
        return ans

    def _fit(self):
        enn = ENN(self.n_neighbors)
        S = enn.fit(self.X, self.y).sample_indices_

        self.X_ = self.X[S]
        self.y_ = self.y[S]

        self.mask = np.ones(len(self.X_), dtype=bool)

        self.set_nearest_neighbors()
        self.set_nearest_enemy()
        self.set_associates()

        idx_order = np.argsort(self.nearest_enemy_distance)[::-1]

        for idx in idx_order:
            associate = list(self.associates[idx])

            self.mask[idx] = False
            withou_i = self.classify(associate)

            self.mask[idx] = True
            with_i = self.classify(associate)

            if withou_i >= with_i:
                self.mask[idx] = False

            for a in self.associates[idx]:
                self.nearest_neighbors[a] = self.get_nearest_neighbors(a)

                for neighbor in self.nearest_neighbors[a].copy():
                    self.associates[neighbor].add(a)

        return S[self.mask]
