import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial import distance
from src.algorithms.enn import enn


class DROP(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1, method="DROP1"):
        self.k = k
        self.method = method

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.clean_data_ = X
        self.removed_indices_ = []
        if self.method == "DROP1":
            self._drop1()
        elif self.method == "DROP2":
            self._drop2()
        elif self.method == "DROP3":
            self._drop3()
        return self.clean_data_, self.y_[self._get_remaining_indices()]

    def _drop1(self):
        preds = self._get_knn_predictions(
            self.X_, self.y_, exclude_index=None, test_removed=False
        )
        current_accuracy = np.sum(preds == self.y_)
        was_correct = preds[0] == self.y_[0]
        to_remove = []
        for j in range(len(self.X_)):
            preds_in = self._get_knn_predictions(
                self.X_, self.y_, exclude_index=[j] + to_remove, test_removed=False
            )
            remain_y = self.y_[np.setdiff1d(np.arange(len(self.y_)), to_remove + [j])]
            new_accuracy = np.sum(preds_in == remain_y)
            if new_accuracy >= current_accuracy - was_correct:
                current_accuracy = new_accuracy
                to_remove.append(j)
                was_correct = (
                    self.y_[j + 1] == preds_in[j - (len(to_remove) - 1)]
                    if j + 1 < len(self.X_)
                    else False
                )
            else:
                was_correct = (
                    self.y_[j + 1] == preds_in[j - len(to_remove)]
                    if j + 1 < len(self.X_)
                    else False
                )
        self.clean_data_ = np.delete(self.X_, to_remove, axis=0)
        self.removed_indices_ = to_remove

    def _drop2(self):
        dists_enemy = self._dist_enemy(self.X_, self.y_)
        removal_order = np.argsort(dists_enemy)[::-1]

        preds = self._get_knn_predictions(
            self.X_, self.y_, exclude_index=None, test_removed=True
        )
        current_accuracy = np.sum(preds == self.y_)

        to_remove = []
        for j in removal_order:
            preds_in = self._get_knn_predictions(
                self.X_, self.y_, exclude_index=[j] + to_remove, test_removed=True
            )
            new_accuracy = np.sum(preds_in == self.y_)
            if new_accuracy >= current_accuracy:
                current_accuracy = new_accuracy
                to_remove.append(j)

        self.clean_data_ = np.delete(self.X_, to_remove, axis=0)
        self.removed_indices_ = to_remove

    def _drop3(self):
        # Run ENN to get the initially kept samples
        self.X_, self.y_ = enn(self.X_, self.y_, self.k)

        self._drop2()

    def _get_knn_predictions(self, X, y, exclude_index, test_removed):

        if exclude_index is not None:
            train_data = np.delete(X, exclude_index, axis=0)
            train_label = np.delete(y, exclude_index, axis=0)
            if test_removed:
                test_data = X
            else:
                test_data = np.delete(X, exclude_index, axis=0)
        else:
            train_data = X
            train_label = y
            test_data = X

        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(train_data, train_label)
        return knn.predict(test_data)

    def _dist_enemy(self, X, y):
        dists = np.zeros(len(X))
        for i in range(len(X)):
            enemy_mask = y != y[i]
            enemy_dists = distance.cdist([X[i]], X[enemy_mask], "euclideanß")
            dists[i] = np.min(enemy_dists)
        return dists

    def predict(self, X):
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(self.clean_data_, self.y_[self._get_remaining_indices()])
        return knn.predict(X)

    def _get_remaining_indices(self):
        return np.setdiff1d(np.arange(len(self.y_)), self.removed_indices_)
