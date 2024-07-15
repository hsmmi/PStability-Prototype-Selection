# src/algorithms/cnn.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
from src.algorithms.base import BaseAlgorithm


class CNN(BaseAlgorithm):
    def __init__(self, n_neighbors=1):
        super().__init__()
        self.n_neighbors: int = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors)
        self.mask: np.ndarray = None

    def select(self) -> np.ndarray:
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
