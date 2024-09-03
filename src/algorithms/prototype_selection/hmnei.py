import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.algorithms.prototype_selection.base import BaseAlgorithm
from sklearn.metrics import pairwise_distances


class HMNEI(BaseAlgorithm):
    def __init__(self, epsilon=0.1, metric="euclidean"):
        self.epsilon = epsilon
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=1, metric=metric)
        self.nearest_in_class = None

    def _set_nearest_in_class(self):
        self.nearest_in_class = np.full((self.X_.shape[0], self.n_classes_), -1)
        # index of the nearest sample in the specified class
        for idx in range(self.X_.shape[0]):
            for class_idx, class_ in enumerate(self.classes_):
                class_indices = np.where(self.y_ == class_)[0]
                if class_ == self.y_[idx]:
                    class_indices = np.delete(
                        class_indices, np.where(class_indices == idx)
                    )
                nearest = np.argmin(self.pairwise_distance[idx][class_indices])
                self.nearest_in_class[idx][class_idx] = class_indices[nearest]

    def _set_hit_miss(self):
        self.hit, self.miss = np.zeros(self.X_.shape[0]), np.zeros(self.X_.shape[0])
        for idx in range(self.X_.shape[0]):
            for class_idx, class_ in enumerate(self.classes_):
                idx2 = self.nearest_in_class[idx][class_idx]
                if self.y_[idx] == class_:
                    self.hit[idx2] += 1
                else:
                    self.miss[idx2] += 1

    def _set_class_count_weight(self):
        self.class_weights_ = np.zeros(self.n_classes_)
        self.class_counts = np.zeros(self.n_classes_)
        for class_idx, class_ in enumerate(self.classes_):
            self.class_counts[class_idx] = np.sum(self.y_ == class_)
            self.class_weights_[class_idx] = (
                self.class_counts[class_idx] / self.n_samples
            )

    def _fit(self):
        S = np.arange(self.n_samples)
        self.X_, self.y_ = self.X, self.y

        accuracy = 0.0

        epoch = 0
        while True:
            epoch += 1
            prev_accuracy = accuracy

            self.pairwise_distance = pairwise_distances(self.X_, metric=self.metric)

            self._set_nearest_in_class()

            self._set_hit_miss()

            self._set_class_count_weight()

            mask = np.ones(self.X_.shape[0], dtype=bool)

            # Rule R1
            for idx in range(self.X_.shape[0]):
                class_idx = np.where(self.classes_ == self.y_[idx])[0]
                if (self.class_weights_[class_idx] * self.miss[idx] + self.epsilon) > (
                    (1 - self.class_weights_[class_idx]) * self.hit[idx]
                ):
                    mask[idx] = False  # R1

            unmarded_indices = np.where(mask == 0)[0]

            # Rule R2
            for class_ in range(self.n_classes_):
                left = np.sum(self.y_[mask] == class_)
                if left < 4:
                    for idx in unmarded_indices:
                        if (
                            self.y_[idx] == class_
                            and (self.hit[idx] + self.miss[idx]) > 0
                        ):
                            mask[idx] = True  # R2

            # Rule R3, R4
            for idx in unmarded_indices:
                if (
                    self.n_classes_ > 3
                    and self.miss[idx] < (self.n_classes_ / 2)
                    and (self.hit[idx] + self.miss[idx] > 0)
                ):
                    mask[idx] = True  # R3
                class_idx = np.where(self.classes_ == self.y_[idx])[0]
                if self.hit[idx] >= self.class_counts[class_idx] / 4:
                    mask[idx] = True  # R4

            S = S[mask]
            self.X_ = self.X_[mask]
            self.y_ = self.y_[mask]

            # Compute accuracy
            self.model.fit(self.X_, self.y_)
            y_pred = self.model.predict(self.X)
            accuracy = np.sum(y_pred == self.y)

            # stop if accuracy is not improved anymore
            # if accuracy + len(mask) < prev_accuracy + len(S) or len(S) == len(mask):
            if accuracy < prev_accuracy or len(S) == len(mask):
                break

        return S
