from matplotlib import pyplot as plt


def plot_algorithm_results(
    X,
    y,
    X_,
    y_,
    title="Algorithm Results",
    save_plot=False,
    show_plot=True,
):
    """
    Plot the results of the specified algorithm.

    Parameters:
    X (numpy.ndarray): Feature matrix of the training data.
    y (numpy.ndarray): Labels of the training data.
    X_ (numpy.ndarray): Reduced feature matrix.
    y_ (numpy.ndarray): Reduced labels.
    title (str): Title of the plot.
    """

    # Compare the original and reduced dataset next to each other
    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", color="red")
    axs[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", color="blue")
    axs[0].set_title("Original Dataset")
    axs[0].legend()
    X_interval = (X[:, 0].min(), X[:, 0].max())
    y_interval = (X[:, 1].min(), X[:, 1].max())
    axs[1].scatter(
        X_[y_ == 0][:, 0],
        X_[y_ == 0][:, 1],
        label="Class 0",
        color="red",
    )
    axs[1].scatter(
        X_[y_ == 1][:, 0],
        X_[y_ == 1][:, 1],
        label="Class 1",
        color="blue",
    )
    axs[1].set_xlim(X_interval)
    axs[1].set_ylim(y_interval)
    axs[1].set_title("Reduced Dataset")
    axs[1].legend()
    plt.suptitle(title)

    if save_plot:
        plt.savefig(f"results/figures/{title}.png")
    if show_plot:
        plt.show()

    # Close the plot
    plt.close()
