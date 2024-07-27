import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from src.algorithms.stability.combination_generator import CombinationGenerator
from src.utils.data_preprocessing import load_data


class PStability:
    def __init__(self, X_train, y_train, X_test, y_test, n_neighbors=5):
        """
        Initializes the PStability object.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
            X_test (np.ndarray): Test data features.
            y_test (np.ndarray): Test data labels.
            n_neighbors (int): Number of neighbors for KNeighborsClassifier.
        """
        # Initialize training and test data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set KNeighborsClassifier parameters
        self.n_neighbors = n_neighbors
        self.n_samples = X_train.shape[0]

        # Determine unique classes in the training data
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)

        # Initialize KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Initialize the combination generator for removing samples
        self.combination_generator = CombinationGenerator()

        # Keep track of currently selected indices
        self._selected_indices: np.ndarray = np.arange(self.n_samples)

        # Calculate and store the base accuracy
        self.base_accuracy = self._get_accuracy()

    def _get_accuracy(self):
        """
        Trains the KNN model on the selected indices and computes the accuracy on the test set.

        Returns:
            float: Accuracy of the model on the test set.
        """
        # Train the KNN model using the selected indices in the training set
        self.knn.fit(
            self.X_train[self._selected_indices], self.y_train[self._selected_indices]
        )

        # Compute and return the accuracy on the test set
        return self.knn.score(self.X_test, self.y_test)

    def _find_maximum_p(self, epsilon: float = 0.0) -> int:
        """
        Finds the maximum p where accuracy is maintained within the threshold.

        Args:
            epsilon (float): Tolerance for accuracy drop.

        Returns:
            int: Maximum number of samples that can be removed without dropping accuracy
                 below the threshold.
        """
        # Iterate through different sizes of removed sets
        for p in range(1, self.n_samples + 1):

            # Display progress bar with tqdm, with leave=False to remove it after completion
            for self._selected_indices in tqdm(
                self.combination_generator.set_params(
                    self.n_samples, self.n_samples - p
                ),
                desc=f"Finding maximum p, Evaluating p={p}",  # Description of the current progress
                leave=False,
            ):

                # Calculate the accuracy with the current set of selected indices
                accuracy = self._get_accuracy()

                # Check if accuracy drops below the base accuracy minus epsilon
                if accuracy < self.base_accuracy - epsilon:
                    # Return the previous p value as the maximum where accuracy is maintained
                    return p - 1

        # If no significant drop in accuracy is found, return n_samples
        return self.n_samples

    def _find_epsilon(self, p: int):
        """
        Finds the appropriate epsilon value for a given p value that maintains accuracy.

        In other words, this method finds the maximum drop in accuracy.

        Args:
            p (int): Number of indices in the removed set.
        """
        # Initialize the maximum decrease in accuracy
        max_decrease = -1
        avg_decrease = 0

        # Iterate through all possible combinations of removed indices
        for self._selected_indices in tqdm(
            self.combination_generator.set_params(self.n_samples, self.n_samples - p),
            desc=f"Find epsilon, Evaluating p={p}",
            leave=False,
        ):

            # Calculate the accuracy with the current set of selected indices
            accuracy = self._get_accuracy()

            decreased = self.base_accuracy - accuracy
            avg_decrease += decreased

            # Update the maximum decrease in accuracy
            if decreased > max_decrease:
                max_decrease = decreased

        avg_decrease /= len(self.combination_generator)

        # Return the maximum and average decrease in accuracy
        return max_decrease, avg_decrease


if __name__ == "__main__":
    # Load the data
    X, y = load_data("wine")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize PStability with training and test sets
    p_stability = PStability(X_train, y_train, X_test, y_test)

    p_list = [1, 2, 3]
    for p in p_list:
        max_decrease, avg_decrease = p_stability._find_epsilon(p)
        print(
            f"For p={p}, max decrease: {max_decrease:.6f}, avg decrease: {avg_decrease:.6f}"
        )
