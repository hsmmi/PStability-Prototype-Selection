from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y


class BaseAlgorithm(ABC):
    """
    Base class for instance selection algorithms.
    """

    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.sample_indices_: np.ndarray = None
        self.X_: np.ndarray = None
        self.y_: np.ndarray = None
        self.reduction_ratio_: float = None
        self.n_samples: int = None
        self.n_features: int = None
        self.classes_: np.ndarray = None
        self.n_classes_: int = None
        self.metric: str = "euclidean"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseAlgorithm":
        """
        Fit the model using the training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        self: Fitted instance of the algorithm.
        """
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.sample_indices_ = []
        self.sample_indices_ = self._fit()
        self.X_ = X[self.sample_indices_]
        self.y_ = y[self.sample_indices_]
        self.reduction_ratio = 1 - len(self.X_) / len(X)
        return self

    @abstractmethod
    def _fit(self) -> np.ndarray:
        """
        Select instances from the training data.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        np.ndarray: Indices of the selected instances.
        """
        pass

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform the data by selecting instances.

        Parameters:
        X (np.ndarray): Data to transform.
        y (np.ndarray): Target values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed data and target values.
        """
        if self.sample_indices_ is None:
            raise ValueError("The model has not been fitted yet.")
        return X[self.sample_indices_], y[self.sample_indices_]

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the model and transform the data in a single step.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed data and target values.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def score(self) -> tuple[float, float]:
        """
        Return the accuracy and reduction ratio of the model.

        Returns:
        Tuple[float, float]: Accuracy and reduction ratio.
        """

        # Ckeck if the model has been fitted
        if self.sample_indices_ is None:
            raise ValueError("The model has not been fitted yet.")

        # If number of selected instances less than n_neighbors return 0
        if len(self.sample_indices_) < 5:
            return 0, 100

        # Fit the KNN classifier on the reduced dataset
        knn = KNeighborsClassifier()
        knn.fit(self.X[self.sample_indices_], self.y[self.sample_indices_])

        # Evaluate the classifier
        y_pred = knn.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)
        reduction_ratio = 1 - len(self.sample_indices_) / len(self.X)

        return accuracy, reduction_ratio
