from typing import Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from config.log import get_logger
from sklearn.utils.validation import check_X_y

logger = get_logger("mylogger")
logger.setLevel("INFO")


class KNN:
    def __init__(self, metric: str = "euclidean") -> None:
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
        self.n_classes: int = None
        self.classes: np.ndarray = None
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
        self.mask_train: np.ndarray = None
        self.classify_correct: np.ndarray = None
        self.friends: list[list[int]] = None
        self.number_of_friends_sorted_index: list[Tuple[int, int]] = None
        self.n_misses: int = None
        self.n_misses_initial: int = None

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

        # if there is no nearest friend i.e. index is out of bounds (-1)
        if nearest_friend_idx == -1:
            return nearest_enemy_idx, False
        if nearest_enemy_idx == -1:
            return nearest_friend_idx, True

        if nearest_friend_idx < nearest_enemy_idx:
            return self.nearest_friend(idx), True
        else:
            return self.nearest_enemy(idx), False

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

    def _set_nearest_friends_enemies(self):
        """
        Set the nearest friends and enemies for each instance.
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

    def find_friends_list(self, idx: int) -> list[int]:
        """
        Find the friends of an instance. "Friends" are the instances that have
        the same class label as the target instance and occur before the nearest enemy
        in the list of nearest neighbors.

        Instance is index of data test in dataset X. Neighbours are in train data which is in mask.

        Parameters
        ----------
        idx : int
            Index of the target instance.

        Returns
        -------
        list[int]
            List of indices of the friends of the instance. Returns an empty list
            if no friends are found or if the target instance is not in the mask.
        """
        # Find the indices of the nearest friend and nearest enemy
        nearest_friend_pointer = self.nearest_friend_index(idx)
        nearest_enemy_pointer = self.nearest_enemy_index(idx)

        # Fetch the class label of the target instance
        class_label = self.y[idx]

        # Retrieve the list of nearest neighbours for the instance
        nearest_neighbour = self.nearest_neighbours[idx]

        # Collect the friends that appear before the nearest enemy
        friends_list = []
        for pointer in range(nearest_friend_pointer, nearest_enemy_pointer):
            neighbour_idx = nearest_neighbour[pointer]

            # Check if the neighbour is in the mask and shares the same class label
            if self.mask_train[neighbour_idx] and self.y[neighbour_idx] == class_label:
                friends_list.append(neighbour_idx)

        if self.classify_correct[idx] == False and len(friends_list) > 0:
            # raise e ValueError("Instance is classified correctly and has friends.")
            logger.error(
                f"Instance {idx} is classified incorrectly but has friends: {friends_list}"
            )
        if self.classify_correct[idx] == True and len(friends_list) == 0:
            # raise e ValueError("Instance is classified incorrectly and has no friends.")
            logger.error(f"Instance {idx} is classified correctly but has no friends.")

        return friends_list

    def compute_all_firends(self) -> list[list[int]]:
        """
        Find the friends for each instance test(whole dataset) in train data(mask data).

        Returns
        -------
        list[list[int]]
            List of lists containing the indices of the friends for each instance.
            Index i in the list corresponds to the friends of instance i.
        """

        return [self.find_friends_list(idx) for idx in range(self.n_samples)]

    def find_instance_with_min_friends(
        self, start_index: int = 0
    ) -> Tuple[int, list[int]]:
        """
        Find the instance with the minimum number of friends until the nearest enemy.
        The instance must be classified correctly.

        ! If the friend list of instance be empty, it will be ignored.

        Parameters
        ----------
        start_index : int, optional
            The starting index for the search, by default 0.

        Returns
        -------
        int
            The index of the instance with the minimum number of friends until the nearest enemy.
        list[int]
            The list of friends of the instance.
        """
        min_friends, min_idx = self.n_samples + 1, -1
        min_friends_list = []
        for idx in range(start_index, self.n_samples):
            if self.classify_correct[idx]:
                friends = self.find_friends_list(idx)
                if len(friends) == 0:
                    logger.warning(
                        f"Instance {idx} has no friends and is classified correctly in find min friends."
                    )
                    self.find_friends_list(idx)
                if len(friends) < min_friends:
                    min_friends, min_idx = len(friends), idx
                    min_friends_list = friends
        return min_idx, min_friends_list

    def find_number_of_friends_sorted_index(self) -> "None":
        """
        Sort instances by the number of friends they have until the nearest enemy.
        Just return the instances that are in the mask.

        Returns
        -------
        list[Tuple[int, int]]
            List of (indices, number of friends until nearest enemy) sorted by
            the number of friends in ascending order.
        """
        return sorted(
            [
                (idx, len(self.find_friends_list(idx)))
                for idx in range(self.n_samples)
                if self.mask_train[idx]
            ],
            key=lambda x: x[1],
        )

    def remove_nearest_friends(
        self, idx: int, update_nearest_enemy: bool = False
    ) -> dict:
        """
        Remove the nearest friends of a point until the nearest enemy is reached.

        Parameters
        ----------
        idx : int
            Index of the point.
        update_nearest_enemy : bool
            Whether to update the nearest enemies (default is False).

        Returns
        -------
        dict
            A dictionary with the following structure:

            - "classify_incorrect": list[int]
                A list of indices of instances that become misclassified after removing the friends.

            - "classify_correct": list[int]
                A list of indices of instances that become classified correctly after removing the friends.

            - "friends": list[int]
                A list of indices of the nearest friends that were removed.

            - "update_nearest_friends": dict[int, int]
                A dictionary where the keys are the indices of instances whose nearest friend pointers were updated,
                and the values are the number of updates for each instance.

            - "update_nearest_enemies": dict[int, int], optional
                A dictionary where the keys are the indices of instances whose nearest enemy pointers were updated,
                and the values are the number of updates for each instance. This key is included only if
                `update_nearest_enemy` is True.
        """
        changes = {}
        changes["friends"] = self.find_friends_list(idx)
        self.mask_train[changes["friends"]] = False
        changes["update_nearest_friends"] = {}
        if update_nearest_enemy:
            changes["update_nearest_enemies"] = {}
        for idx2 in range(self.n_samples):
            nearest_friend_idx2 = self.nearest_friend(idx2)
            while (
                nearest_friend_idx2 in changes["friends"]
                or (self.mask_train[nearest_friend_idx2] == False)
            ) and nearest_friend_idx2 != -1:
                changes["update_nearest_friends"][idx2] = (
                    changes["update_nearest_friends"].get(idx2, 0) + 1
                )
                self.nearest_friends_pointer[idx2] += 1
                nearest_friend_idx2 = self.nearest_friend(idx2)
            if update_nearest_enemy:
                nearest_enemy_idx2 = self.nearest_enemy(idx2)
                while (
                    nearest_enemy_idx2 in changes["friends"]
                    or self.mask_train[nearest_enemy_idx2] == False
                ) and nearest_enemy_idx2 != -1:
                    changes["update_nearest_enemies"][idx2] = (
                        changes["update_nearest_enemies"].get(idx2, 0) + 1
                    )
                    self.nearest_enemies_pointer[idx2] += 1
                    nearest_enemy_idx2 = self.nearest_enemy(idx2)

        changes["classify_incorrect"] = []
        changes["classify_correct"] = []
        # Check if instance becomes misclassified after removing the friends of the point
        for idx2 in range(self.n_samples):
            if self.classify_correct[idx2] and self._classify(idx2) != self.y[idx2]:
                changes["classify_incorrect"].append(idx2)
                self.classify_correct[idx2] = False
                self.n_misses += 1
            if (
                self.classify_correct[idx2] == False
                and self._classify(idx2) == self.y[idx2]
            ):
                changes["classify_correct"].append(idx2)
                self.classify_correct[idx2] = True
                self.n_misses -= 1
        return changes

    def put_back_nearest_friends(self, changed_list: dict) -> None:
        """
        Put back the nearest friends of a point that were removed.

        Parameters
        ----------
        changed_list : dict
            A dictionary with the following structure:

            - "classify_incorrect": list[int]
                A list of indices of instances that become misclassified after removing the friends.

            - "classify_correct": list[int]
                A list of indices of instances that become classified correctly after removing the friends.

            - "friends": list[int]
                A list of indices of the nearest friends that were removed.

            - "update_nearest_friends": dict[int, int]
                A dictionary where the keys are the indices of instances whose nearest friend pointers were updated,

            - "update_nearest_enemies": dict[int, int], optional
                A dictionary where the keys are the indices of instances whose nearest enemy pointers were updated,
                and the values are the number of updates for each instance. This key is included only if
        """
        neighbours_idx = changed_list["friends"]
        self.classify_correct[changed_list["classify_incorrect"]] = True
        self.n_misses -= len(changed_list["classify_incorrect"])
        self.classify_correct[changed_list["classify_correct"]] = False
        self.n_misses += len(changed_list["classify_correct"])
        self.mask_train[neighbours_idx] = True
        for idx2, count in changed_list["update_nearest_friends"].items():
            self.nearest_friends_pointer[idx2] -= count
        if "update_nearest_enemies" in changed_list:
            for idx2, count in changed_list["update_nearest_enemies"].items():
                self.nearest_enemies_pointer[idx2] -= count

    def remove_point(self, idx: int, update_nearest_enemy: bool = False) -> dict:
        """
        Remove a point from the mask and update the nearest pointer for other instances.

        Parameters
        ----------
        idx : int
            Index of the point to remove.
        update_nearest_enemy : bool
            Whether to update the nearest enemies (default is False).

        Returns
        -------
        dict
            A dictionary with the following structure:

            - "classify_incorrect": list[int]
                A list of indices of instances that become misclassified after removing the point.

            - "classify_correct": list[int]
                A list of indices of instances that become classified correctly after removing the point.

            - "update_nearest_friends": dict[int, int]
                A dictionary where the keys are the indices of instances whose nearest friend pointers were updated,
                and the values are the number of updates for each instance.

            - "update_nearest_enemies": dict[int, int], optional
                A dictionary where the keys are the indices of instances whose nearest enemy pointers were updated,
                and the values are the number of updates for each instance. This key is included only if
                `update_nearest_enemy` is True.

        """
        self.mask_train[idx] = False
        changes = {}
        changes["classify_incorrect"] = []
        changes["classify_correct"] = []
        changes["update_nearest_friends"] = {}
        changes["update_nearest_enemies"] = {}
        for idx2 in range(self.n_samples):
            nearest_friend_idx = self.nearest_friend(idx2)
            while (
                nearest_friend_idx == idx
                or self.mask_train[nearest_friend_idx] == False
            ) and nearest_friend_idx != -1:
                changes["update_nearest_friends"][idx2] = (
                    changes["update_nearest_friends"].get(idx2, 0) + 1
                )
                self.nearest_friends_pointer[idx2] += 1
                nearest_friend_idx = self.nearest_friend(idx2)
            if update_nearest_enemy:
                nearest_enemy_idx = self.nearest_enemy(idx2)
                while (
                    nearest_enemy_idx == idx
                    or self.mask_train[nearest_enemy_idx] == False
                ) and nearest_enemy_idx != -1:
                    changes["update_nearest_enemies"][idx2] = (
                        changes["update_nearest_enemies"].get(idx2, 0) + 1
                    )
                    self.nearest_enemies_pointer[idx2] += 1
                    nearest_enemy_idx = self.nearest_enemy(idx2)
            # Check if instance becomes misclassified after removing the point and vice versa
            if self.classify_correct[idx2] and self._classify(idx2) != self.y[idx2]:
                changes["classify_incorrect"].append(idx2)
                self.classify_correct[idx2] = False
                self.n_misses += 1
            if (
                self.classify_correct[idx2] == False
                and self._classify(idx2) == self.y[idx2]
            ):
                changes["classify_correct"].append(idx2)
                self.classify_correct[idx2] = True
                self.n_misses -= 1
        return changes

    def put_back_point(
        self,
        idx: int,
        changed: Tuple[dict[int, int], dict[int, int]],
    ) -> None:
        """
        Put back a point to the mask and update the nearest pointer
        for the instances which had been changed.

        Parameters
        ----------
        idx : int
            Index of the point to put back.
        changed : dict
            A dictionary with the following structure:

            - "classify_incorrect": list[int]
                A list of indices of instances that become misclassified after removing the point.

            - "classify_correct": list[int]
                A list of indices of instances that become classified correctly after removing the point.

            - "update_nearest_friends": dict[int, int]
                A dictionary where the keys are the indices of instances whose nearest friend pointers were updated,
                and the values are the number of updates for each instance.

            - "update_nearest_enemies": dict[int, int], optional
                A dictionary where the keys are the indices of instances whose nearest enemy pointers were updated,
                and the values are the number of updates for each instance. This key is included only if
                `update_nearest_enemy` is True.

        """
        self.classify_correct[changed["classify_incorrect"]] = True
        self.n_misses -= len(changed["classify_incorrect"])
        self.classify_correct[changed["classify_correct"]] = False
        self.n_misses += len(changed["classify_correct"])
        self.mask_train[idx] = True
        for idx2, count in changed["update_nearest_friends"].items():
            self.nearest_friends_pointer[idx2] -= count
        if "update_nearest_enemies" in changed:
            for idx2, count in changed["update_nearest_enemies"].items():
                self.nearest_enemies_pointer[idx2] -= count

    def reset(self) -> None:
        """
        Reset the changes made.
        """
        logger.debug("Resetting changes in KNN.")
        self.mask_train = np.ones(self.n_samples, dtype=bool)
        self.nearest_friends_pointer = np.zeros(self.n_samples, dtype=int)
        self.nearest_enemies_pointer = np.zeros(self.n_samples, dtype=int)

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
        self.n_samples, self.n_features = X.shape
        self.y = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        logger.debug(f"Fitting KNN with k={self.k} and metric={self.metric}.")
        knn = NearestNeighbors(n_neighbors=self.n_samples, metric=self.metric)
        knn.fit(X)
        self.nearest_neighbours = knn.kneighbors(
            X, n_neighbors=self.n_samples, return_distance=False
        )[:, 1:]

        self._set_nearest_friends_enemies()

        self.mask_train = np.ones(self.n_samples, dtype=bool)
        self.classify_correct = np.array(
            [self._classify(i) == y[i] for i in range(self.n_samples)]
        )
        self.n_misses = self.n_samples - np.sum(self.classify_correct)
        self.n_misses_initial = self.n_misses

        logger.info("Nearest neighbours and enemies set.")
        logger.info("Model fitting complete.")
        return self

    def accuracy(self) -> float:
        """
        Calculate the accuracy of the KNN model.

        Returns
        -------
        float
            The accuracy of the model.
        """
        return 1 - self.n_misses / self.n_samples
