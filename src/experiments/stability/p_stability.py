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

    max_stability = p_stability.n_samples
    max_distortion = p_stability.n_samples - p_stability.n_misses

    with measure_time("Runtime: Exact stability for each distortion"):
        list_distortion = list(range(10))
        list_exact_stability = p_stability.run_exact_stability(list_distortion)
        logger.info(f"Exact stability: {list_exact_stability}")
        excel_content["Exact stability"] = {
            "distortion": list_distortion,
            "stability": list_exact_stability,
        }

    with measure_time("Runtime: Lower bound stability for each distortion"):
        list_distortion = list(range(max_distortion + 1))
        list_lower_bound_stability = p_stability.run_lower_bound_stability(
            list_distortion
        )
        logger.info(
            f"Lower bound stability for distortion: {list_lower_bound_stability}"
        )
        excel_content["Lower Bound stability"] = {
            "distortion": list_distortion,
            "stability": list_lower_bound_stability,
        }

    with measure_time("Runtime: Better upper bound stability for each distortion"):
        list_distortion = list(range(max_distortion + 1))
        list_better_upper_bound_stability = (
            p_stability.run_better_upper_bound_stability(list_distortion)
        )
        logger.info(
            f"Better upper bound stability for distortion: {list_better_upper_bound_stability}"
        )
        excel_content["Better Upper Bound stability"] = {
            "distortion": list_distortion,
            "stability": list_better_upper_bound_stability,
        }
    with measure_time("Runtime: Upper bound stability for each distortion"):
        list_distortion = list(range(21))
        list_upper_bound_stability = p_stability.run_upper_bound_stability(
            list_distortion
        )
        logger.info(
            f"Upper bound stability for distortion: {list_upper_bound_stability}"
        )
        excel_content["Upper Bound stability"] = {
            "distortion": list_distortion,
            "stability": list_upper_bound_stability,
        }
    with measure_time("Runtime: Exact distortion for each stability"):
        list_stability = [0, 1, 2]
        list_exact_distortion = p_stability.run_exact_distortion(list_stability)
        logger.info(f"Exact distortion: {list_exact_distortion}")
        excel_content["Exact Missclasdistortionifications"] = {
            "stability": list_stability,
            "distortion": list_exact_distortion,
        }

    with measure_time("Runtime: Lower bound distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_lower_bound_distortion = p_stability.run_lower_bound_distortion(
            list_stability
        )
        logger.info(
            f"Lower bound distortion for stability: {list_lower_bound_distortion}"
        )
        excel_content["Lower Bound distortion"] = {
            "stability": list_stability,
            "distortion": list_lower_bound_distortion,
        }

    with measure_time("Runtime: Better lower bound distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_better_lower_bound_distortion = (
            p_stability.run_better_lower_bound_distortion(list_stability)
        )
        logger.info(
            f"Better lower bound distortion for stability: {list_better_lower_bound_distortion}"
        )
        excel_content["Better Lower Bound distortion"] = {
            "stability": list_stability,
            "distortion": list_better_lower_bound_distortion,
        }

    with measure_time("Runtime: Upper bound distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_upper_bound_distortion = p_stability.run_upper_bound_distortion(
            list_stability
        )
        logger.info(
            f"Upper bound distortion for stability: {list_upper_bound_distortion}"
        )
        excel_content["Upper Bound distortion"] = {
            "stability": list_stability,
            "distortion": list_upper_bound_distortion,
        }

    with measure_time("Runtime: Crisped distortion for stability"):
        list_stability = list(range(max_stability + 1))
        list_crisped_distortion = p_stability.run_crisped_distortion(list_stability)
        logger.info(
            f"Crisped distortionclassification score: {list_crisped_distortion}"
        )
        excel_content["Crisped distortion(lower)"] = {
            "stability": list_stability,
            "Crisped Miss": list_crisped_distortion,
        }

    with measure_time("Runtime: Fuzzy distortion for stability"):
        list_stability = list(range(max_stability + 1))
        list_fuzzy_distortion = p_stability.run_fuzzy_distortion(list_stability)
        list_fuzzy_distortion = [
            round(fuzzy_distortion, 2) for fuzzy_distortion in list_fuzzy_distortion
        ]
        logger.info(f"Fuzzy distortion score: {list_fuzzy_distortion}")
        excel_content["Fuzzy distortion(upper)"] = {
            "stability": list_stability,
            "Fuzzy distortion": list_fuzzy_distortion,
        }
    save_jsonl("p_stability", {"dataset": dataset, "results": excel_content})
    plot_bounds(excel_content, dataset, show_plot=False, save_plot=True)
    save_to_excel(excel_content, f"p_stability {dataset} tmp", mode="horizontal")


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
