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
        list_distortion = list(range(14))
        list_exact_stability = p_stability.run_exact_stability(list_distortion)
        logger.info(f"Exact stability: {list_exact_stability}")
        excel_content["Exact stability"] = {
            "distortion": list_distortion,
            "stability": list_exact_stability,
        }

    with measure_time("Runtime: same friend stability for each distortion"):
        list_distortion = list(range(max_distortion + 1))
        list_same_friend_stability = p_stability.run_same_friend_stability(
            list_distortion
        )
        logger.info(
            f"same friend stability for distortion: {list_same_friend_stability}"
        )
        excel_content["same friend stability"] = {
            "distortion": list_distortion,
            "stability": list_same_friend_stability,
        }

    with measure_time("Runtime: greedy stability for each distortion"):
        list_distortion = list(range(max_distortion + 1))
        list_greedy_stability = p_stability.run_greedy_stability(list_distortion)
        logger.info(f"greedy stability for distortion: {list_greedy_stability}")
        excel_content["greedy stability"] = {
            "distortion": list_distortion,
            "stability": list_greedy_stability,
        }
    with measure_time("Runtime: unique friend stability for each distortion"):
        list_distortion = list(range(21))
        list_unique_friend_stability = p_stability.run_unique_friend_stability(
            list_distortion
        )
        logger.info(
            f"unique friend stability for distortion: {list_unique_friend_stability}"
        )
        excel_content["unique friend stability"] = {
            "distortion": list_distortion,
            "stability": list_unique_friend_stability,
        }
    with measure_time("Runtime: Exact distortion for each stability"):
        list_stability = [0, 1, 2]
        list_exact_distortion = p_stability.run_exact_distortion(list_stability)
        logger.info(f"Exact distortion: {list_exact_distortion}")
        excel_content["Exact Missclasdistortionifications"] = {
            "stability": list_stability,
            "distortion": list_exact_distortion,
        }

    with measure_time("Runtime: unique friend distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_unique_friend_distortion = p_stability.run_unique_friend_distortion(
            list_stability
        )
        logger.info(
            f"unique friend distortion for stability: {list_unique_friend_distortion}"
        )
        excel_content["unique friend distortion"] = {
            "stability": list_stability,
            "distortion": list_unique_friend_distortion,
        }

    with measure_time("Runtime: greedy distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_greedy_distortion = p_stability.run_greedy_distortion(list_stability)
        logger.info(f"greedy distortion for stability: {list_greedy_distortion}")
        excel_content["greedy distortion"] = {
            "stability": list_stability,
            "distortion": list_greedy_distortion,
        }

    with measure_time("Runtime: same friend distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_same_friend_distortion = p_stability.run_same_friend_distortion(
            list_stability
        )
        logger.info(
            f"same friend distortion for stability: {list_same_friend_distortion}"
        )
        excel_content["same friend distortion"] = {
            "stability": list_stability,
            "distortion": list_same_friend_distortion,
        }

    with measure_time("Runtime: binary distortion for stability"):
        list_stability = list(range(max_stability + 1))
        list_binary_distortion = p_stability.run_binary_distortion(list_stability)
        logger.info(f"binary distortionclassification score: {list_binary_distortion}")
        excel_content["binary distortion(lower)"] = {
            "stability": list_stability,
            "distortion": list_binary_distortion,
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
            "distortion": list_fuzzy_distortion,
        }
    save_jsonl("p_stability", {"dataset": dataset, "results": excel_content})
    plot_bounds(excel_content, dataset, show_plot=False, save_plot=True)
    save_to_excel(excel_content, f"p_stability {dataset} tmp", mode="horizontal")


if __name__ == "__main__":

    dataset_list = [
        "circles_0.05_150",
        "moons_0.15_150",
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
