import collections
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.algorithms.base import BaseAlgorithm
from src.algorithms.enn import ENN
from sklearn.metrics.pairwise import euclidean_distances


class DROP3(BaseAlgorithm):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.sample_indices_ = []
        self.X_ = None
        self.y_ = None
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def set_nearest_neighbors(self):
        self.pairwise_distances = euclidean_distances(self.X_)

        for i in range(self.pairwise_distances.shape[0]):
            self.pairwise_distances[i][i] = -1.0

        self.nearest_neighbors_complete = np.argsort(self.pairwise_distances)[:, 1:]
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
                    if self.pairwise_distances[i][j] < self.nearest_enemy_distance[i]:
                        self.nearest_enemy_distance[i] = self.pairwise_distances[i][j]
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
        ans = 0
        for idx in associate:
            idx_neighbors = self.get_nearest_neighbors(idx)
            pred_idx = self.most_common([self.y_[i] for i in idx_neighbors])
            if pred_idx == self.y_[idx]:
                ans += 1
        return ans

    def select(self, X, y):
        enn = ENN(k=self.n_neighbors)
        S = enn.fit(X, y).sample_indices_

        self.X_ = X[S]
        self.y_ = y[S]

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

        self.sample_indices_ = S[self.mask]
        self.X_ = X[self.sample_indices_]
        self.y_ = y[self.sample_indices_]

        return self.sample_indices_
