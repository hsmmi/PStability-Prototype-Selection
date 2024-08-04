import time
from src.utils.data_preprocessing import load_data
from src.algorithms.stability.p_stability import run_p_stability
from src.utils.path import ProjectPath
from config.log import get_logger

if __name__ == "__main__":
    logger = get_logger("mylogger")
    # Set handelers level to INFO
    logger.setLevel("DEBUG")

    FILE_NAME = ProjectPath(__file__).get_safe_filename()
    DATASET_NAME = "wine"  # Dataset name for logging and data loading

    # Load data
    X, y = load_data(DATASET_NAME)

    # print("X shape: ", X.shape)
    # results = run_p_stability(
    #     X,
    #     y,
    #     n_folds=5,
    #     find_max_p=True,
    #     show_progress=True,
    #     n_jobs=1,
    # )

    # logger.info(f"Max p: {results.max_p} with time elapsed: {results.time}")

    find_epsilon = [3]

    results = run_p_stability(
        X, y, n_folds=5, find_epsilon=find_epsilon, show_progress=True, n_jobs=8
    )

    print("time elapsed: ", results.time, " seconds with n_jobs=1")

    # # Print max and average epsilon values for each p with tabulated format
    # from tabulate import tabulate

    # table = [
    #     [
    #         p,
    #         f"{results.max_epsilon[i]:.2%}",
    #         f"{results.avg_epsilon[i]:.2%}",
    #     ]
    #     for i, p in enumerate(find_epsilon)
    # ]
    # headers = ["p", "Max Epsilon", "Avg Epsilon"]
    # headers = [f"{header:^10}" for header in headers]

    # print(
    #     tabulate(
    #         table,
    #         headers=headers,
    #         tablefmt="fancy_grid",
    #         numalign="center",
    #         stralign="center",
    #     )
    # )
