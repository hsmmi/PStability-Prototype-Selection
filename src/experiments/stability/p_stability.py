from tqdm import tqdm
from src.algorithms.stability.p_stability import PStability
from src.utils.timer import measure_time
from config.log import get_logger
from src.utils.excel import save_to_excel
from src.utils.result import save_jsonl
from src.utils.data_preprocessing import load_data
from src.utils.visualization import plot_bounds

logger = get_logger("mylogger")


def run_dataset(dataset: str):
    X, y = load_data(dataset)

    excel_content = {}

    p_stability = PStability()
    with measure_time("fitting"):
        p_stability.fit(X, y)

    max_p = p_stability.n_samples
    max_stability = p_stability.n_samples - p_stability.n_misses

    with measure_time("Runtime: Exact p for each Stability"):
        list_stability = list(range(11))
        list_exact_p = p_stability.run_exact_p(list_stability)
        logger.info(f"Exact p: {list_exact_p}")
        excel_content["Exact p"] = {
            "stability": list_stability,
            "p": list_exact_p,
        }

    with measure_time("Runtime: Lower bound p for each stability"):
        list_stability = list(range(max_stability + 1))
        list_lower_bound_p = p_stability.run_lower_bound_p(list_stability)
        logger.info(f"Lower bound p for stability: {list_lower_bound_p}")
        excel_content["Lower Bound p"] = {
            "stability": list_stability,
            "p": list_lower_bound_p,
        }

    with measure_time("Runtime: Better upper bound p for each stability"):
        list_stability = list(range(max_stability + 1))
        list_better_upper_bound_p = p_stability.run_better_upper_bound_p(list_stability)
        logger.info(f"Better upper bound p for stability: {list_better_upper_bound_p}")
        excel_content["Better Upper Bound p"] = {
            "stability": list_stability,
            "p": list_better_upper_bound_p,
        }
    with measure_time("Runtime: Upper bound p for each stability"):
        list_stability = list(range(21))
        list_upper_bound_p = p_stability.run_upper_bound_p(list_stability)
        logger.info(f"Upper bound p for stability: {list_upper_bound_p}")
        excel_content["Upper Bound p"] = {
            "stability": list_stability,
            "p": list_upper_bound_p,
        }
    with measure_time("Runtime: Exact Stability for each p"):
        list_p = [0, 1, 2]
        list_exact_stability = p_stability.run_exact_stability(list_p)
        logger.info(f"Exact Stability: {list_exact_stability}")
        excel_content["Exact MissclasStabilityifications"] = {
            "p": list_p,
            "stability": list_exact_stability,
        }

    with measure_time("Runtime: Lower bound stability for each p"):
        list_p = list(range(max_p + 1))
        list_lower_bound_stability = p_stability.run_lower_bound_stability(list_p)
        logger.info(f"Lower bound stability for p: {list_lower_bound_stability}")
        excel_content["Lower Bound Stability"] = {
            "p": list_p,
            "stability": list_lower_bound_stability,
        }

    with measure_time("Runtime: Better lower bound stability for each p"):
        list_p = list(range(max_p + 1))
        list_better_lower_bound_stability = (
            p_stability.run_better_lower_bound_stability(list_p)
        )
        logger.info(
            f"Better lower bound stability for p: {list_better_lower_bound_stability}"
        )
        excel_content["Better Lower Bound Stability"] = {
            "p": list_p,
            "stability": list_better_lower_bound_stability,
        }

    with measure_time("Runtime: Upper bound stability for each p"):
        list_p = list(range(max_p + 1))
        list_upper_bound_stability = p_stability.run_upper_bound_stability(list_p)
        logger.info(f"Upper bound stability for p: {list_upper_bound_stability}")
        excel_content["Upper Bound Stability"] = {
            "p": list_p,
            "stability": list_upper_bound_stability,
        }

    with measure_time("Runtime: Crisped Stability for p"):
        list_p = list(range(max_p + 1))
        list_crisped_stability = p_stability.run_crisped_stability(list_p)
        logger.info(f"Crisped stabilityclassification score: {list_crisped_stability}")
        excel_content["Crisped Stability(lower)"] = {
            "p": list_p,
            "Crisped Miss": list_crisped_stability,
        }

    with measure_time("Runtime: Fuzzy Stability for p"):
        list_p = list(range(max_p + 1))
        list_fuzzy_stability = p_stability.run_fuzzy_stability(list_p)
        list_fuzzy_stability = [
            round(fuzzy_stability, 2) for fuzzy_stability in list_fuzzy_stability
        ]
        logger.info(f"Fuzzy stability score: {list_fuzzy_stability}")
        excel_content["Fuzzy Stability(upper)"] = {
            "p": list_p,
            "Fuzzy Stability": list_fuzzy_stability,
        }
    save_jsonl("p_stability", {"dataset": dataset, "results": excel_content})
    plot_bounds(excel_content, dataset, show_plot=False, save_plot=True)
    # save_to_excel(excel_content, f"p_stability {dataset} tmp", mode="horizontal")


if __name__ == "__main__":

    dataset_list = [
        "circles_0.05_undersampled",
        "moons_0.15_undersampled",
        "iris_undersampled",
        "wine_undersampled",
        "iris_0_1",
        "iris_0_2",
        "iris_1_2",
        "iris",
        "wine",
    ]

    for dataset in tqdm(dataset_list, desc="Datasets progress", leave=False):
        run_dataset(dataset)
