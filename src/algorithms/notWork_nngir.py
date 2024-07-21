import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree, KNeighborsClassifier
import random
from src.algorithms.base import BaseAlgorithm


class NNGIR(BaseAlgorithm):
    def __init__(self, n_neighbors: int = 1, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.n_neighbors: int = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors, metric=metric)
        self.edited_set = []
        self.core_instances = []
        self.Nb = None
        self.NaN = []
        self.hoe = None
        self.hee = None

    def _set_nan_search(self) -> None:
        n = self.X_.shape[0]
        r = 1
        self.Nb = np.zeros(n, dtype=int)
        self.NaN = [set() for _ in range(n)]
        self.hoe = [set() for _ in range(n)]
        self.hee = [set() for _ in range(n)]
        self.NNr = [set() for _ in range(n)]
        tree = KDTree(self.X_)
        prev_num = n

        while True:
            for idx in range(n):
                _, indices = tree.query(self.X_[idx].reshape(1, -1), r + 1)
                if r < len(indices[0]):
                    idx2 = indices[0][r]  # r-th nearest neighbor e(idx, idx2)
                    self.Nb[idx2] += 1
                    self.NNr[idx].add(idx2)
                    self.NaN[idx2].add(idx)
                    if self.y_[idx] == self.y_[idx2]:
                        self.hoe[idx2].add(idx)
                    else:
                        self.hee[idx2].add(idx)

            current_num = sum(1 for neighbors in self.Nb if neighbors == 0)

            if r > 1 and current_num == prev_num:
                break
            else:
                prev_num = current_num
                r += 1

    def _noise_filter(self) -> None:
        edited_set = []
        indices = np.where(self.Nb)[
            0
        ]  # Filter instances without natural neighbors(outliers)

        for idx in indices:
            # Then filter instances that are not correctly classified by their natural neighbors
            nn_labels = self.y_[list(self.NaN[idx])]
            predominant_class = np.argmax(np.bincount(nn_labels))
            # TODO: it's different from the original paper because it can be multiple classes
            if self.y_[idx] == predominant_class:
                edited_set.append(idx)

        self.S = self.S[edited_set]

    def _search_core_instances(self) -> None:
        self.core_instances = []
        candidates = np.arange(self.X_.shape[0])
        while True:
            # Check if Nb is all zeros
            if not self.Nb[candidates].any() or candidates.size == 0:
                break
            # Find the instance with the maximum number of natural neighbors
            candidate_max_idx = np.argmax(self.Nb[candidates])
            idx = candidates[candidate_max_idx]
            natural_neighbors = self.NaN[idx]
            if all(self.Nb[neighbour] > 0 for neighbour in natural_neighbors):
                self.core_instances.append(idx)
                self.Nb[idx] = 0
                for neighbour in natural_neighbors:
                    self.Nb[neighbour] = 0
            else:
                candidates = np.delete(candidates, candidate_max_idx)

        self.core_instances = np.array(self.core_instances)

    def _distance_to_nearest_enemy(self, idx: int) -> float:
        nearest_neighbour = self.NNr[idx]
        distances_to_neighbours = pairwise_distances(
            self.X_[idx].reshape(1, -1), self.X_[list(nearest_neighbour)]
        )
        return np.min(distances_to_neighbours)

    def _select_border_instances(self) -> None:
        self.border_set = set()
        noise = set()
        # if |hee[idx]| > 0
        candidates = [idx for idx in range(self.X_.shape[0]) if self.hee[idx]]
        for idx in candidates:
            if len(self.hoe[idx]) >= len(self.hee[idx]):
                for idx2 in self.hee[idx]:
                    self.border_set.add(idx2)
            else:
                noise.add(idx)

        self.border_set = np.array(list(self.border_set - noise))

        self.border_instances = []
        for idx in self.border_set:
            # Distance idx to nearest neighbor in border_instances
            distance = np.inf
            for border_instance in self.border_instances:
                distance = min(
                    distance, np.linalg.norm(self.X_[idx] - self.X_[border_instance])
                )
            if distance > self._distance_to_nearest_enemy(idx):
                self.border_instances.append(idx)

        self.border_instances = np.array(self.border_instances)

    def _fit(self) -> np.ndarray:
        self.S = np.arange(self.n_samples)
        self.X_, self.y_ = self.X, self.y
        self._set_nan_search()
        self._noise_filter()
        self.X_, self.y_ = self.X[self.S], self.y[self.S]
        self._set_nan_search()
        self._search_core_instances()
        self._select_border_instances()
        core_and_border = np.union1d(self.core_instances, self.border_instances)
        return self.S[core_and_border]
