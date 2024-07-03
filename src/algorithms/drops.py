# src/algorithms/drop1.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.utils.data_preprocessing import load_data
from matplotlib import pyplot as plt
from src.utils.evaluation_metrics import comparison


def drop1(X, y, k=1):
    """
    DROP1 algorithm to reduce the training set.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    k (int): Number of neighbors to use for classification.

    Returns:
    numpy.ndarray: Reduced feature matrix.
    numpy.ndarray: Reduced labels.
    """
    knn = KNeighborsClassifier(
        n_neighbors=k + 1
    )  # k+1 because the instance itself will be in its neighbors
    knn.fit(X, y)

    # Find associates for each instance
    neighbors = knn.kneighbors(X, len(X), return_distance=False)
    associates = {i: set() for i in range(len(X))}
    neighbors_labels = []
    for i, nbrs in enumerate(neighbors):
        neighbors_labels.append(y[nbrs])
        for nbr in nbrs[1 : k + 1]:  # Exclude the instance itself
            associates[nbr].add(i)

    to_remove = set()
    for i in range(len(X)):
        correct_with = 0
        correct_without = 0
        # Check the associates of the instance for removal
        for associate in associates[i]:
            # Check if the associate is correctly classified with the instance
            # If count label of the associate in the neighbors_labels of the instance is greater than k/2
            count_label = np.count_nonzero(
                neighbors_labels[i][1 : k + 2] == y[associate]
            )
            label_next = neighbors_labels[i][k + 1]
            label_i = y[i]
            if count_label - (label_next == y[associate]) > k / 2:
                correct_with += 1
            # Check if the associate is correctly classified without the instance
            if count_label - (label_i == y[associate]) > k / 2:
                correct_without += 1
        # If removing the instance does not decrease the classification accuracy of its associates
        if correct_without >= correct_with:
            # Remove x_i
            to_remove.add(i)
            # Updating neighbors
            for j in range(len(X)):
                index = np.where(neighbors[j] == i)[0][0]
                if index == 0:
                    continue
                neighbors[j] = np.append(
                    np.delete(neighbors[j], index),
                    -1,
                )
                neighbors_labels[j] = np.append(
                    np.delete(neighbors_labels[j], index),
                    -1,
                )
            # Update associates i
            if len(associates[i]) > 0:
                tmp = associates[i].copy()
                for associate in associates[i]:
                    tmp.remove(associate)
                    next_neighbor = neighbors[associate][k]
                    associates[next_neighbor].add(associate)
                associates[i] = tmp
            # Update associates connected to i
            for j in range(len(X)):
                if i in associates[j]:
                    associates[j].remove(i)
    remaining_indices = list(set(range(len(X))) - to_remove)
    new_neighbors = neighbors[remaining_indices][: len(remaining_indices)]
    # fit the model with the reduced dataset
    knn.fit(X[remaining_indices], y[remaining_indices])

    n_edges = 0
    for i in range(len(X)):
        n_edges += len(associates[i])
    X_reduced = np.delete(X, list(to_remove), axis=0)
    y_reduced = np.delete(y, list(to_remove))

    return X_reduced, y_reduced


# Example usage
if __name__ == "__main__":
    data = load_data("data/raw/data.csv")
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    (
        accuracy,
        accuracy_reduced,
        len_X_train,
        len_X_reduced,
        reduction_percentage,
        execution_time,
    ) = comparison(X, y, drop1)
    print(f"Original accuracy: {accuracy}")
    print(f"Reduced accuracy: {accuracy_reduced}")
    print(f"Original size: {len_X_train}")
    print(f"Reduced size: {len_X_reduced}")
    print(f"Reduction percentage: {reduction_percentage:.2f}%")
    print(f"Execution time: {execution_time:.2f} seconds")
