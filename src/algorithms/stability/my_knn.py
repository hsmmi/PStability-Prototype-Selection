from typing import Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from config.log import get_logger
from sklearn.utils.validation import check_X_y

logger = get_logger("mylogger")
logger.setLevel("INFO")


class KNN:
    def __init__(self, metric: str = "euclidean"):
        """
        Initialize KNN with the given metric.

        Parameters:
        metric (str): Distance metric to use.
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

    def nearest_friend_index(self, idx: int) -> int:
        """
        Return index of the nearest friend of the instance in nearest_neighbours.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Index of the nearest friend in nearest_neighbours.
        """
        pointer = self.nearest_friends_pointer[idx]
        if pointer >= len(self.nearest_friends[idx]):
            return self.n_samples + 1
        return self.nearest_friends[idx][pointer]

    def nearest_friend(self, idx: int) -> int:
        """
        Get the nearest friend of the instance.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Index of the nearest friend.
        """
        return self.nearest_neighbours[idx][self.nearest_friend_index(idx)]

    def nearest_enemy_index(self, idx: int) -> int:
        """
        Return index of the nearest enemy of the instance in nearest_neighbours.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Index of the nearest enemy in nearest_neighbours.
        """
        return self.nearest_enemies[idx][self.nearest_enemies_pointer[idx]]

    def nearest_enemy(self, idx: int) -> int:
        """
        Get the nearest enemy of the instance.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Index of the nearest enemy.
        """
        return self.nearest_neighbours[idx][self.nearest_enemy_index(idx)]

    def nearest_neighbour(self, idx: int) -> Tuple[int, int]:
        """
        Get the nearest neighbour of the instance.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        Tuple[int, int]: Index of the nearest neighbour, status friend(1) or enemy(0).
        """
        nearest_friend_idx = self.nearest_friend_index(idx)
        nearest_enemy_idx = self.nearest_enemy_index(idx)

        if nearest_friend_idx < nearest_enemy_idx:
            return self.nearest_friend(idx), 1
        else:
            return self.nearest_enemy(idx), 0

    def r_nearest_neighbour(self, idx: int, r: int) -> int:
        """
        Get the r-th nearest neighbour of the instance.

        Parameters:
        idx (int): Index of the instance.
        r (int): The r-th nearest neighbour.

        Returns:
        int: Index of the r-th nearest neighbour.
        """
        return self.nearest_neighbours[idx][r]

    def _set_nearest_enemy_pointer(self):
        """
        Set the nearest enemy and friend pointers for each instance.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """
        Fit the model using the training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        self: Fitted instance of the algorithm.
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
