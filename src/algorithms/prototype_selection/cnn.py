# src/algorithms/cnn.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
from src.algorithms.prototype_selection.base import BaseAlgorithm


class CNN(BaseAlgorithm):
    def __init__(self, n_neighbors: int = 1, metric="euclidean"):
        super().__init__()
        self.metric = metric
        self.n_neighbors: int = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors, metric=metric)
        self.mask: np.ndarray = None

    def _fit(self) -> np.ndarray:
        self.mask = np.zeros(len(self.X), dtype=bool)
        for cls in self.classes_:
            randam_index = random.choice(np.where(self.y == cls)[0])
            self.mask[randam_index] = True

        progress = True
        while progress:
            progress = False
            self.classifier.fit(self.X[self.mask], self.y[self.mask])
            for i in range(len(self.X)):
                if not self.mask[i]:
                    if self.classifier.predict([self.X[i]]) != self.y[i]:
                        self.mask[i] = True
                        progress = True

        return np.where(self.mask)[0]
