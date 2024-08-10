from typing import Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from config.log import get_logger
from sklearn.utils.validation import check_X_y

logger = get_logger("mylogger")
logger.setLevel("INFO")


class KNN:
    def __init__(
        self, metric: str = "euclidean", update_nearest_enemy: bool = False
    ) -> None:
        """
        Initialize the KNN model with the specified distance metric.

        Parameters
        ----------
        metric : str
            Distance metric to use (default is "euclidean").
        update_nearest_enemy : bool
            Whether to update the nearest enemies (default is False).
        """
        self.k: int = 1
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_samples: int = None
        self.n_features: int = None
        self.metric: str = metric
        self.distance_to_nearest_neighbour: np.ndarray = None
        self.nearest_neighbours: np.ndarray = None
        self.nearest_friends: np.ndarray = None
        self.nearest_friends_pointer: np.ndarray = None
        # pointers to the enemies of the nearest neighbours sorted by distance
        # ATTENTION: This is not index of the enemy, but pointer to the enemy in the list of nearest neighbours sorted by distance
        # So nearest enemy index is self.nearest_neighbours[self.nearest_enemies[idx]][self.nearest_enemies_pointer[idx]]
        self.nearest_enemies: np.ndarray = None
        self.nearest_enemies_pointer: np.ndarray = None
        self.update_nearest_enemy: bool = update_nearest_enemy

    def _classify(self, idx: int) -> int:
        """
        Classify an instance using 1-NN.

        Parameters
        ----------
        idx : int
            The index of the instance to classify.

        Returns
        -------
        int
            The predicted class of the instance.
        """
        nearest_neighbor_idx, _ = self.nearest_neighbour(idx)
        return self.y[nearest_neighbor_idx]

    def nearest_friend_index(self, idx: int) -> int:
        """
        Get the index of the nearest friend of the instance in the list of nearest neighbours.

        Parameters
        ----------
        idx : int
            The index of the instance.

        Returns
        -------
        int
            Index of the nearest friend in `nearest_neighbours`.
        """
        pointer = self.nearest_friends_pointer[idx]
        if pointer >= len(self.nearest_friends[idx]):
            return self.n_samples + 1
        return self.nearest_friends[idx][pointer]

    def nearest_friend(self, idx: int) -> int:
        """
        Get the nearest friend of the instance.

        Parameters
        ----------
        idx : int
            The index of the instance.

        Returns
        -------
        int
            Index of the nearest friend.
        """
        pointer = self.nearest_friend_index(idx)
        if pointer == self.n_samples + 1:
            return -1
        return self.nearest_neighbours[idx][pointer]

    def nearest_enemy_index(self, idx: int) -> int:
        """
        Get the index of the nearest enemy of the instance in the list of nearest neighbours.

        Parameters
        ----------
        idx : int
            The index of the instance.

        Returns
        -------
        int
            Index of the nearest enemy in `nearest_neighbours`.
        """
        pointer = self.nearest_enemies_pointer[idx]
        if pointer >= len(self.nearest_enemies[idx]):
            return self.n_samples + 1
        return self.nearest_enemies[idx][pointer]

    def nearest_enemy(self, idx: int) -> int:
        """
        Get the nearest enemy of the instance.

        Parameters
        ----------
        idx : int
            The index of the instance.

        Returns
        -------
        int
            Index of the nearest enemy.
        """
        pointer = self.nearest_enemy_index(idx)
        if pointer == self.n_samples + 1:
            return -1
        return self.nearest_neighbours[idx][pointer]

    def nearest_neighbour(self, idx: int) -> Tuple[int, bool]:
        """
        Get the nearest neighbour of the instance and whether it's a friend or enemy.

        Parameters
        ----------
        idx : int
            The index of the instance.

        Returns
        -------
        Tuple[int, bool]
            Index of the nearest neighbour and a boolean indicating
            whether it's a friend (1) or enemy (0).
        """
        nearest_friend_idx = self.nearest_friend_index(idx)
        nearest_enemy_idx = self.nearest_enemy_index(idx)

        if nearest_friend_idx < nearest_enemy_idx:
            return self.nearest_friend(idx), True
        else:
            return self.nearest_enemy(idx), False

    def r_nearest_neighbour(self, idx: int, r: int) -> int:
        """
        Get the r-th nearest neighbour of the instance.

        Parameters
        ----------
        idx : int
            The index of the instance.
        r : int
            The r-th nearest neighbour.

        Returns
        -------
        int
            Index of the r-th nearest neighbour.
        """
        return self.nearest_neighbours[idx][r]

    def _set_nearest_enemy_pointer(self):
        """
        Set the pointers to the nearest enemies and friends for each instance.
        """
        self.nearest_friends = []
        self.nearest_friends_pointer = np.zeros(self.n_samples, dtype=int)
        self.nearest_enemies = []
        self.nearest_enemies_pointer = np.zeros(self.n_samples, dtype=int)

        logger.debug("Setting nearest enemy and friend pointers.")
        for idx in range(self.n_samples):
            is_enemy = self.y[self.nearest_neighbours[idx]] != self.y[idx]
            enemy_indices = np.nonzero(is_enemy)[0]
            self.nearest_enemies.append(enemy_indices)
            self.nearest_friends.append(np.where(~is_enemy)[0])

    def _number_of_friends_until_nearest_enemy(self, idx: int) -> int:
        """
        Calculate the number of friends an instance has until the nearest enemy.

        Parameters
        ----------
        idx : int
            Index of the instance.

        Returns
        -------
        int
            Number of friends until the nearest enemy.
        """
        return max(
            0,
            (self.nearest_enemy_index(idx) - self.nearest_enemies_pointer[idx])
            - (self.nearest_friend_index(idx) - self.nearest_friends_pointer[idx]),
        )

    def _sort_by_nearest_enemy(self) -> np.ndarray:
        """
        Sort instances by the number of friends they have until the nearest enemy.

        Returns
        -------
        np.ndarray
            Indices of the instances sorted by the number of friends they have.
        """
        return np.argsort(
            [
                self._number_of_friends_until_nearest_enemy(idx)
                for idx in range(self.n_samples)
            ]
        )

    def _remove_nearest_neighbours(self, idx: int) -> dict[int, list[int]]:
        """
        Remove the nearest neighbours of a point until the nearest enemy is reached.

        Parameters
        ----------
        idx : int
            Index of the point.

        Returns
        -------
        dict[int, list[int]]
            Dictionary of indices that had their nearest pointers updated containing
            the indices of the nearest neighbours and the indices of the nearest enemies.
        """
        nearest_friends_pointer = self.nearest_friend_index(idx)
        nearest_neighbours_enemy_pointer = self.nearest_enemy_index(idx)
        neighbour_idx = self.nearest_neighbours[idx][
            nearest_friends_pointer:nearest_neighbours_enemy_pointer
        ]
        changes = {}
        changes["neighbours"] = neighbour_idx
        self.mask[neighbour_idx] = False
        changes["update_nearest_friends"] = {}
        if self.update_nearest_enemy:
            changes["update_nearest_enemies"] = {}
        for idx2 in range(self.n_samples):
            while self.nearest_friend(idx2) in neighbour_idx:
                changes["update_nearest_friends"][idx2] = (
                    changes["update_nearest_friends"].get(idx2, 0) + 1
                )
                self.nearest_friends_pointer[idx2] += 1
            if self.update_nearest_enemy:
                while self.nearest_enemy(idx2) in neighbour_idx:
                    changes["update_nearest_enemies"][idx2] = (
                        changes["update_nearest_enemies"].get(idx2, 0) + 1
                    )
                    self.nearest_enemies_pointer[idx2] += 1
        return changes

    def _put_back_nearest_neighbours(self, changed_list: dict[int, list[int]]) -> None:
        """
        Put back the nearest neighbours of a point that were removed.

        Parameters
        ----------
        changed_list : dict[int, list[int]]
            Dictionary of indices that had their nearest pointers updated, including
            the indices of the nearest neighbours and the indices of the nearest enemies.
        """
        neighbours_idx = changed_list["neighbours"]
        self.mask[neighbours_idx] = True
        for idx2, count in changed_list["update_nearest_friends"].items():
            self.nearest_friends_pointer[idx2] -= count
        if self.update_nearest_enemy:
            for idx2, count in changed_list["update_nearest_enemies"].items():
                self.nearest_enemies_pointer[idx2] -= count

    def _remove_point_update_neighbours(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove a point from the mask and update the nearest pointer for the neighbours.

        Parameters
        ----------
        idx : int
            Index of the point to remove.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of indices that had their nearest pointers updated.
            The first array contains the indices of the nearest neighbours
            and the second array contains the indices of the nearest enemies.
        """
        self.mask[idx] = False
        changed_nearest_neighbor = np.nonzero(
            [self.nearest_friend(idx2) == idx for idx2 in range(self.n_samples)]
        )[0]
        self.nearest_friends_pointer[changed_nearest_neighbor] += 1
        if self.update_nearest_enemy:
            changed_nearest_enemy = np.nonzero(
                [self.nearest_enemy(idx2) == idx for idx2 in range(self.n_samples)]
            )[0]
            self.nearest_enemies_pointer[changed_nearest_enemy] += 1
        else:
            changed_nearest_enemy = None

        return [changed_nearest_neighbor, changed_nearest_enemy]

    def _put_back_point(self, idx: int, changed: np.ndarray) -> None:
        """
        Put back a point to the mask and update the nearest pointer
        for the neighbours which had been changed.

        Parameters
        ----------
        idx : int
            Index of the point to put back.
        changed : np.ndarray
            Array of indices that had their nearest pointers updated.
        """
        self.mask[idx] = True
        changed_nearest_neighbor, changed_nearest_enemy = changed
        self.nearest_friends_pointer[changed_nearest_neighbor] -= 1
        if self.update_nearest_enemy:
            self.nearest_enemies_pointer[changed_nearest_enemy] -= 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """
        Fit the KNN model using the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        update_nearest_enemy : bool
            Whether to update the nearest enemies (default is False).

        Returns
        -------
        KNN
            The fitted KNN instance.
        """
        try:
            X, y = check_X_y(X, y, ensure_2d=True, dtype=None)
        except ValueError as e:
            logger.error(f"Error in check_X_y: {e}")
            raise

        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        logger.debug(f"Fitting KNN with k={self.k} and metric={self.metric}.")
        knn = NearestNeighbors(n_neighbors=self.n_samples, metric=self.metric)
        knn.fit(X)
        self.nearest_neighbours = knn.kneighbors(
            X, n_neighbors=self.n_samples, return_distance=False
        )[:, 1:]

        self._set_nearest_enemy_pointer()

        logger.info("Nearest neighbours and enemies set.")
        logger.info("Model fitting complete.")
        return self
