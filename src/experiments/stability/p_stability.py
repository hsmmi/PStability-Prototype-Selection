import time
from tqdm import tqdm
from src.algorithms.stability.p_stability import PStability
from src.utils.timer import measure_time
from config.log import get_logger
from src.utils.excel import save_to_excel
from src.utils.result import save_jsonl
from src.utils.data_preprocessing import load_data
from src.utils.visualization import plot_bounds

logger = get_logger("mylogger")

datasets = [
    # "appendicitis",
    "bupa",
    # "ecoli",
    # "glass",
    "haberman",
    # "heart",
    # "ionosphere",
    # "iris",
    # "liver",
    "movement_libras",
    # "promoters",
    "sonar",
    # "wine",
    # "zoo",
]
# datasets = [
#     "appendicitis",
#     # "bupa",
#     # "ecoli",
#     # "glass",
#     # "haberman",
#     # "heart",
#     # "ionosphere",
#     "iris",
#     # "liver",
#     # "movement_libras",
#     "promoters",
#     # "sonar",
#     # "wine",
#     "zoo",
# ]


folder = f"p_stability/{time.strftime("%Y-%m-%d %H:%M:%S")}_{len(datasets)}-DS"+ "/"

def run_dataset(dataset: str):
    X, y = load_data(dataset)

    excel_content = {}

    p_stability = PStability()
    with measure_time("fitting"):
        p_stability.fit(X, y)

    max_stability = p_stability.n_samples
    max_distortion = p_stability.n_samples - p_stability.n_misses

    with measure_time("Runtime: Exact Stability for each distortion"):
        list_distortion = list(range(0))
        list_exact_stability = p_stability.run_exact_stability(list_distortion)
        logger.info(f"Exact Stability: {list_exact_stability}")
        excel_content["Exact Stability"] = {
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
        list_distortion = list(range(max_distortion + 1))
        list_unique_friend_stability = p_stability.run_unique_friend_stability(
            list_distortion
        )
        list_distortion = list(range(len(list_unique_friend_stability)))
        logger.info(
            f"unique friend stability for distortion: {list_unique_friend_stability}"
        )
        excel_content["unique friend stability"] = {
            "distortion": list_distortion,
            "stability": list_unique_friend_stability,
        }
    with measure_time("Runtime: Exact distortion for each stability"):
        list_stability = list(range(0))
        list_exact_distortion = p_stability.run_exact_distortion(list_stability)
        logger.info(f"Exact distortion: {list_exact_distortion}")
        excel_content["Exact Missclasdistortionifications"] = {
            "stability": list_stability,
            "distortion": list_exact_distortion,
        }

    with measure_time("Runtime: Unique Friend Distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_unique_friend_distortion = p_stability.run_unique_friend_distortion(
            list_stability
        )
        logger.info(
            f"Unique Friend Distortion for stability: {list_unique_friend_distortion}"
        )
        excel_content["Unique Friend Distortion"] = {
            "stability": list_stability,
            "distortion": list_unique_friend_distortion,
        }

    with measure_time("Runtime: Greedy Distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_greedy_distortion = p_stability.run_greedy_distortion(list_stability)
        logger.info(f"Greedy Distortion for stability: {list_greedy_distortion}")
        excel_content["Greedy Distortion"] = {
            "stability": list_stability,
            "distortion": list_greedy_distortion,
        }

    with measure_time("Runtime: Same Friend Distortion for each stability"):
        list_stability = list(range(max_stability + 1))
        list_same_friend_distortion = p_stability.run_same_friend_distortion(
            list_stability
        )
        logger.info(
            f"Same Friend Distortion for stability: {list_same_friend_distortion}"
        )
        excel_content["Same Friend Distortion"] = {
            "stability": list_stability,
            "distortion": list_same_friend_distortion,
        }

    with measure_time("Runtime: binary distortion for stability"):
        list_stability = list(range(max_stability + 1))
        list_binary_distortion = p_stability.run_binary_distortion(list_stability)
        logger.info(f"binary distortionclassification score: {list_binary_distortion}")
        excel_content["Binary Distortion"] = {
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
        excel_content["Fuzzy Distortion"] = {
            "stability": list_stability,
            "distortion": list_fuzzy_distortion,
        }
    save_jsonl(folder+"p_stability", {"dataset": dataset, "results": excel_content})
    bt_ds = dataset.replace("_", " ").capitalize()
    plot_bounds(excel_content, bt_ds, folder, show_plot=False, save_plot=True)
    save_to_excel(excel_content, folder+f"p_stability {dataset}", mode="horizontal")


if __name__ == "__main__":
    for dataset in tqdm(datasets, desc="Datasets progress", leave=False):
        run_dataset(dataset)
