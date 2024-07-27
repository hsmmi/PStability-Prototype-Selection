import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from src.utils.data_preprocessing import load_data
from src.algorithms.stability.p_stability import PStability

DATASET_NAME = "circles_0.05" + "_undersampled"

# Get file name
FILE_NAME = __file__.split("/")[-1].split(".")[0]

n_folds = 5

# k-fold cross validation
X, y = load_data(DATASET_NAME)


def run_p_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    find_max_p: bool = True,
    find_epsilon: list[int] = [],
):
    """
    Run the PStability algorithm on the given dataset.
    Using k-fold cross validation, the algorithm finds the maximum p value and epsilon value for each fold.

    Parameters:
    X (numpy.ndarray): Feature matrix of the dataset.
    y (numpy.ndarray): Labels of the dataset.
    n_folds (int): Number of folds for cross validation.
    find_max_p (bool): Whether to find the maximum p value.
    find_epsilon (list): List of p values to find the epsilon value.

    """
    if type(find_epsilon) != list:
        raise ValueError("find_epsilon must be a list of integers.")

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    max_p = []
    max_epsilon = [[] for _ in find_epsilon]
    avg_epsilon = [[] for _ in find_epsilon]

    for train_index, test_index in tqdm(
        kf.split(X, y),
        total=n_folds,
        desc="K-Fold progress",
        leave=False,
    ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        p_stability = PStability(X_train, y_train, X_test, y_test)
        if find_max_p:
            max_p.append(p_stability._find_maximum_p())

        for i, p in enumerate(find_epsilon):
            tmp_max, tmp_avg = p_stability._find_epsilon(p)
            max_epsilon[i].append(tmp_max)
            avg_epsilon[i].append(tmp_avg)

    max_p = np.min(max_p) if find_max_p else None
    max_epsilon = np.max(max_epsilon, axis=1) if find_epsilon else None
    avg_epsilon = np.mean(avg_epsilon, axis=1) if find_epsilon else None

    return max_p, max_epsilon, avg_epsilon


find_epsilon = [1, 2, 3]

max_p, max_epsilon, avg_epsilon = run_p_stability(
    X, y, n_folds=n_folds, find_max_p=True, find_epsilon=find_epsilon
)

print(f"Max p: {max_p}")
# print max and average epsilon values for each p with tabulated format
import tabulate

table = []
for i, p in enumerate(find_epsilon):
    table.append([p, f"{max_epsilon[i]:.2%}", f"{avg_epsilon[i]:.2%}"])

headers = ["p", "Max Epsilon", "Avg Epsilon"]

# Add padding to the headers :^10
headers = [f"{header:^10}" for header in headers]

print(
    tabulate.tabulate(
        table,
        headers=headers,
        tablefmt="fancy_grid",
        numalign="center",
        stralign="center",
    )
)
