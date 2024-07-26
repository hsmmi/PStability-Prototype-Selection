import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree, KNeighborsClassifier
import random
from src.algorithms.prototype_selection.base import BaseAlgorithm


class CNNIR(BaseAlgorithm):
    def __init__(self, n_neighbors: int = 1, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.n_neighbors: int = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors, metric=metric)

    def _set_nan_search(self) -> None:
        n = self.X.shape[0]
        r = 1
        self.Nb = np.zeros(n, dtype=int)
        self.NaN = [set() for _ in range(n)]
        NNr = [set() for _ in range(n)]
        tree = KDTree(self.X)
        prev_num = n

        while True:
            for idx in range(n):
                _, indices = tree.query(self.X[idx].reshape(1, -1), r + 1)
                if r < len(indices[0]):
                    idx2 = indices[0][r]  # r-th nearest neighbor
                    self.Nb[idx2] += 1
                    NNr[idx].add(idx2)
                    self.NaN[idx2].add(idx)

            current_num = sum(1 for neighbors in self.Nb if neighbors == 0)

            if r > 1 and current_num == prev_num:
                break
            else:
                prev_num = current_num
                r += 1

    def _noise_filter(self) -> None:
        edited_set = []

        indices = np.where(self.Nb)[0]  # Filter instances without natural neighbors

        for idx in indices:
            # Then filter instances that are not correctly classified by their natural neighbors
            nn_labels = self.y[list(self.NaN[idx])]
            predominant_class = np.argmax(np.bincount(nn_labels))

            if self.y[idx] == predominant_class:
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

    def _fit(self) -> np.ndarray:
        self.S = np.arange(self.X.shape[0])
        self._set_nan_search()
        self._noise_filter()
        self.X_, self.y_ = self.X[self.S], self.y[self.S]
        self._search_core_instances()
        reduced_set = np.union1d(self.edited_set, self.core_instances)
        return reduced_set
