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
        self.nearest_pointer: np.ndarray = None
        self.nearest_enemy_pointer: np.ndarray = None

    def nearest_neighbour(self, idx: int) -> int:
        """
        Get the nearest neighbour of the instance.

        Parameters:
        idx (int): Index of the instance.

        Returns:
        int: Index of the nearest neighbour.
        """
        return self.nearest_neighbours[idx][self.nearest_pointer[idx]]

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
        Set the nearest enemy for each instance.
        """
        logger.info("Setting nearest enemy pointers.")
        for idx in range(self.n_samples):
            enemy_indices = np.where(
                self.y[idx] != self.y[self.nearest_neighbours[idx]]
            )[0]
            if enemy_indices.size > 0:
                self.nearest_enemy_pointer[idx] = enemy_indices[0]

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

        logger.info(f"Fitting KNN with k={self.k} and metric={self.metric}.")
        knn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        knn.fit(X)
        self.nearest_neighbours = knn.kneighbors(
            X, n_neighbors=self.n_samples, return_distance=False
        )[:, 1:]
        self.nearest_pointer = np.zeros(self.n_samples, dtype=int)
        self.nearest_enemy_pointer = np.full(self.n_samples, -1)
        self._set_nearest_enemy_pointer()

        logger.info("Nearest neighbours and enemies set.")
        logger.info("Model fitting complete.")
        return self
