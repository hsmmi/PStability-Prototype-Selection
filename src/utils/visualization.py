from matplotlib import pyplot as plt


def plot_algorithm_results(X, y, X_reduced, y_reduced, title="Algorithm Results"):
    """
    Plot the results of the specified algorithm.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    X_reduced (numpy.ndarray): Reduced feature matrix.
    y_reduced (numpy.ndarray): Reduced labels.
    title (str): Title of the plot.
    """

    # Compare the original and reduced dataset next to each other
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", color="red")
    axs[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", color="blue")
    axs[0].set_title("Original Dataset")
    axs[0].legend()
    X_interval = (X[:, 0].min(), X[:, 0].max())
    y_interval = (X[:, 1].min(), X[:, 1].max())
    axs[1].scatter(
        X_reduced[y_reduced == 0][:, 0],
        X_reduced[y_reduced == 0][:, 1],
        label="Class 0",
        color="red",
    )
    axs[1].scatter(
        X_reduced[y_reduced == 1][:, 0],
        X_reduced[y_reduced == 1][:, 1],
        label="Class 1",
        color="blue",
    )
    axs[1].set_xlim(X_interval)
    axs[1].set_ylim(y_interval)
    axs[1].set_title("Reduced Dataset")
    axs[1].legend()

    plt.suptitle(title)
    plt.show()
